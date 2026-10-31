#!/usr/bin/env python3
"""Train DESL on multiple-visit onset landmarks: first recorded HT onset within a window (2 or 3 years) after a negative anchor."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             brier_score_loss, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import train as original
from data import load_visits, load_splits, derive, audit, Preprocessor, make_graphs


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def metrics(y, p, pred):
    y, p, pred = np.asarray(y, int), np.asarray(p, float), np.asarray(pred, int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    divide = lambda a, b: float(a / b) if b else 0.0
    return {"ACC": float((y == pred).mean()),
            "AUC": float(roc_auc_score(y, p)) if len(set(y)) == 2 else np.nan,
            "AUPRC": float(average_precision_score(y, p)) if y.sum() else np.nan,
            "Precision": divide(tp, tp + fp), "Recall": divide(tp, tp + fn),
            "F1": divide(2 * tp, 2 * tp + fp + fn), "Specificity": divide(tn, tn + fp),
            "Brier": float(brier_score_loss(y, p)), "ECE": original.compute_ece(y, p),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


class Platt:
    """Positive-slope Platt scaling, preserving within-fold ranking."""
    def fit(self, probabilities, y):
        x = logit(np.clip(probabilities, 1e-6, 1 - 1e-6))
        def objective(ab):
            z = ab[0] * x + ab[1]
            residual = expit(z) - y
            return np.mean(np.logaddexp(0, z) - y * z), np.array([np.mean(residual * x), np.mean(residual)])
        result = minimize(objective, [1., 0.], method="L-BFGS-B", jac=True,
                          bounds=[(1e-6, 100), (-100, 100)])
        if not result.success:
            raise RuntimeError(f"Calibration failed: {result.message}")
        self.a, self.b = map(float, result.x)
        return self

    def predict(self, probabilities):
        return expit(self.a * logit(np.clip(probabilities, 1e-6, 1 - 1e-6)) + self.b)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    result = []
    for batch in loader:
        result.extend(model(batch.to(device)).softmax(1)[:, 1].cpu().numpy())
    return np.asarray(result)


def prepare(cfg, output):
    visits, images = load_visits(cfg)
    assignments = load_splits(cfg, visits)
    samples = derive(visits, cfg["horizon_days"], cfg["min_history_visits"])
    audit_result = audit(samples, assignments, visits, cfg)
    assignments.to_csv(output / "patient_split_assignments.csv", index=False)
    samples.to_csv(output / "samples_metadata.csv", index=False)
    dump(output / "leakage_audit.json", audit_result)
    counts = []
    for fold, group in assignments.groupby("fold"):
        for split, g in group.groupby("split"):
            subset = samples[samples.patient_id.isin(g.patient_id)]
            eligible = subset[subset.censored == 0]
            counts.append({"fold": fold, "split": split, "assigned_patients": len(g),
                           "candidate_anchors": len(subset), "eligible_patients": eligible.patient_id.nunique(),
                           "eligible_samples": len(eligible), "positive": int(eligible.label.sum()),
                           "negative": int((eligible.label == 0).sum()), "censored": int(subset.censored.sum()),
                           "positive_rate": float(eligible.label.mean())})
    pd.DataFrame(counts).to_csv(output / "sample_derivation.csv", index=False)
    eligible = samples[samples.censored == 0]
    overview = {"patients_in_source": visits.patient_id.nunique(), "candidate_anchors": len(samples),
                "eligible_anchors": len(eligible), "eligible_patients": eligible.patient_id.nunique(),
                "positive_anchors": int(eligible.label.sum()), "negative_anchors": int((eligible.label == 0).sum()),
                "censored_anchors": int(samples.censored.sum()),
                "censored_fraction": float(samples.censored.mean()), "positive_fraction": float(eligible.label.mean())}
    dump(output / "sample_overview.json", overview)
    print(json.dumps(overview), flush=True)
    return visits, images, assignments, eligible


def run_fold(fold, cfg, output, visits, images, assignments, eligible, device):
    directory = output / f"fold_{fold}"
    directory.mkdir(exist_ok=True)
    if (directory / "complete.json").exists():
        print(f"Fold {fold}: completed, reuse saved predictions", flush=True)
        return
    seed = cfg["seed"] + fold
    original.set_seed(seed)
    random.seed(seed)
    pool = assignments[assignments.fold == fold]
    parts = {}
    for split in ["train", "val", "test"]:
        frame = eligible[eligible.patient_id.isin(pool.loc[pool.split == split, "patient_id"])].copy()
        frame = frame.sort_values(["patient_id", "anchor_date"]).reset_index(drop=True)
        frame["sample_weight"] = 1.0
        if split == "train" and cfg["patient_balanced"]:
            frame["sample_weight"] = 1 / frame.groupby("patient_id").sample_id.transform("size")
            assert np.allclose(frame.groupby("patient_id").sample_weight.sum(), 1)
        assert set(frame.label) == {0, 1}, f"Fold {fold}/{split} lacks both classes"
        parts[split] = frame
    prep = Preprocessor(cfg["clinical_variables"]).fit(visits, parts["train"])
    assert set(visits.loc[prep.fit_rows, "patient_id"]) <= set(parts["train"].patient_id)
    dump(directory / "preprocessing.json", prep.state())
    tabular = prep.transform(visits)
    graphs = {s: make_graphs(f, visits, images, tabular, original.TAU) for s, f in parts.items()}
    loaders = {s: DataLoader(g, batch_size=cfg["batch_size"] if s == "train" else 64,
                              shuffle=s == "train", num_workers=0) for s, g in graphs.items()}
    model = original.get_model(device, in_dim=images.shape[1] + tabular.shape[1],
                               tab_dim=tabular.shape[1], fusion_type="reliability_gate", fusion_dim=cfg["fusion_dim"])
    weights = parts["train"].sample_weight.to_numpy()
    ytrain = parts["train"].label.to_numpy(int)
    class_weight = np.ones(2)
    if cfg["class_weights"]:
        counts = np.bincount(ytrain, weights=weights, minlength=2)
        class_weight = counts.sum() / (2 * counts)
    effective = weights * class_weight[ytrain]
    weight_normalizer = float(effective.mean())
    cw = torch.tensor(class_weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    ema = original.EMA(model, decay=cfg["ema_decay"])
    best, wait, best_epoch = -np.inf, 0, 0
    log = []
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        losses = []
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = (F.cross_entropy(logits, batch.y, reduction="none", weight=cw)
                    * batch.sample_weight).mean() / weight_normalizer
            assert torch.isfinite(loss), "Nonfinite loss"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ema.update(model)
            losses.append(loss.item())
        backup = ema.swap_in(model)
        vp = predict(model, loaders["val"], device)
        score = float(average_precision_score(parts["val"].label, vp))
        if score > best:
            best, wait, best_epoch = score, 0, epoch
            torch.save({"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "epoch": epoch, "validation_auprc": best, "config": cfg,
                        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}, directory / "best.pt")
        else:
            wait += 1
        ema.restore(model, backup)
        scheduler.step(score)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_AUPRC": score,
               "validation_AUC": float(roc_auc_score(parts["val"].label, vp)),
               "seconds": time.perf_counter() - start, "best_epoch": best_epoch}
        log.append(row)
        pd.DataFrame(log).to_csv(directory / "training_log.csv", index=False)
        if epoch == 1 or epoch % 5 == 0 or wait >= cfg["patience"]:
            print(f"Fold {fold}, epoch {epoch}: val AP={score:.4f}, AUC={row['validation_AUC']:.4f}, best={best_epoch}, {row['seconds']:.1f}s", flush=True)
        if wait >= cfg["patience"]:
            break
    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    vp = predict(model, loaders["val"], device)
    vy = parts["val"].label.to_numpy(int)
    threshold, _ = original.find_best_threshold(vy, vp, "f1")
    calibration = Platt().fit(vp, vy)
    # Transform the fixed raw threshold so calibration cannot change classes.
    calibrated_threshold = float(calibration.predict(np.array([threshold]))[0])
    dump(directory / "calibration.json", {"slope": calibration.a, "intercept": calibration.b,
         "raw_threshold": threshold, "calibrated_threshold": calibrated_threshold, "fit_split": "val"})
    rows = []
    for split in ["val", "test"]:
        p = vp if split == "val" else predict(model, loaders[split], device)
        prediction = parts[split].drop(columns=["history_rows", "sample_weight"]).copy()
        prediction["fold"] = fold
        prediction["split"] = split
        prediction["raw_score"] = p
        prediction["calibrated_probability"] = calibration.predict(p)
        prediction["predicted_label"] = (p >= threshold).astype(int)
        prediction["selected_threshold"] = threshold
        prediction["calibrated_threshold"] = calibrated_threshold
        prediction.to_csv(directory / f"{split}_predictions.csv", index=False)
        if split == "test":
            for evaluation in ["rolling_multivisit", "earliest_multivisit"]:
                selected = prediction if evaluation == "rolling_multivisit" else prediction[prediction.first_multivisit_anchor == 1]
                for scale in ["raw_score", "calibrated_probability"]:
                    rows.append({"fold": fold, "evaluation": evaluation, "score_type": scale,
                                 "samples": len(selected), "patients": selected.patient_id.nunique(),
                                 **metrics(selected.label, selected[scale], selected.predicted_label)})
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    dump(directory / "complete.json", {"best_epoch": best_epoch, "epochs_run": epoch,
         "device": str(device), "trainable_params": checkpoint["trainable_params"],
         "preprocessing_fit_train_only": True, "test_used_for_selection": False})
    print(f"Fold {fold} completed: {rows[0]}", flush=True)


def postprocess(cfg, output):
    dirs = sorted(p.parent for p in output.glob("fold_*/complete.json"))
    if not dirs:
        return
    folds = pd.concat([pd.read_csv(d / "metrics.csv") for d in dirs], ignore_index=True)
    predictions = pd.concat([pd.read_csv(d / "test_predictions.csv", dtype={"patient_id": str}) for d in dirs], ignore_index=True)
    assert not predictions.sample_id.duplicated().any()
    assert predictions.groupby("patient_id").fold.nunique().eq(1).all()
    folds.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    keys = ["ACC", "AUC", "AUPRC", "Precision", "Recall", "F1", "Specificity", "Brier", "ECE"]
    summaries = []
    for (evaluation, scale), group in folds.groupby(["evaluation", "score_type"]):
        summaries.extend({"evaluation": evaluation, "score_type": scale, "metric": k,
                          "mean": group[k].mean(), "std": group[k].std(ddof=1), "folds": len(group)} for k in keys)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    if len(dirs) != 5:
        return
    rng = np.random.default_rng(cfg["seed"])
    cis = []
    for evaluation in ["rolling_multivisit", "earliest_multivisit"]:
        frame = predictions if evaluation == "rolling_multivisit" else predictions[predictions.first_multivisit_anchor == 1]
        groups = [g for _, g in frame.groupby("patient_id")]
        draws = []
        for _ in range(cfg["bootstrap_iterations"]):
            sampled = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            if sampled.label.nunique() == 2:
                draws.append(metrics(sampled.label, sampled.calibrated_probability, sampled.predicted_label))
        boot = pd.DataFrame(draws)
        point = metrics(frame.label, frame.calibrated_probability, frame.predicted_label)
        for key in keys:
            cis.append({"evaluation": evaluation, "score_type": "calibrated_probability", "metric": key, "pooled_estimate": point[key],
                        "lower_95": boot[key].quantile(.025), "upper_95": boot[key].quantile(.975),
                        "bootstrap_unit": "patient", "replicates": len(boot)})
    pd.DataFrame(cis).to_csv(output / "patient_bootstrap_ci.csv", index=False)
    plot(predictions, output)


def plot(predictions, output):
    # Quantitative grid: held-out discrimination and calibration, no efficacy claim.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3), layout="constrained")
    y = predictions.label.to_numpy(int)
    p = predictions.calibrated_probability.to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    axes[0].plot(fpr, tpr, color="#31688e", label=f"Pooled AUC = {roc_auc_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=".7", lw=.8)
    axes[0].set(xlabel="False-positive rate", ylabel="Sensitivity", title="a  ROC")
    axes[1].plot(recall, precision, color="#31688e", label=f"Pooled AP = {average_precision_score(y, p):.3f}")
    axes[1].axhline(y.mean(), linestyle="--", color=".7", lw=.8, label=f"Event fraction = {y.mean():.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="b  Precision-recall")
    curve_rows = []
    for field, color, label in [("raw_score", ".5", "Raw score"), ("calibrated_probability", "#31688e", "Platt calibrated")]:
        observed, predicted = calibration_curve(y, predictions[field], n_bins=8, strategy="quantile")
        axes[2].plot(predicted, observed, "o-", color=color, markersize=3, label=label)
        curve_rows.extend({"score_type": field, "mean_prediction": a, "observed_fraction": b} for a, b in zip(predicted, observed))
    axes[2].plot([0, 1], [0, 1], "--", color=".7", lw=.8)
    axes[2].set(xlabel="Mean score / probability", ylabel="Observed event fraction", title="c  Calibration")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "two_year_prediction.pdf", bbox_inches="tight")
    fig.savefig(figures / "two_year_prediction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(figures / "calibration_source.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_2year.json")
    parser.add_argument("--derive-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cfg["device"] = args.device
    if args.smoke:
        cfg.update(epochs=2, patience=2, bootstrap_iterations=10)
        args.folds = [1]
    output = HERE / (("smoke_results" if args.smoke else "results") + f"_{cfg['horizon_days']}d")
    output.mkdir(exist_ok=True)
    cfg["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                           [Path(cfg["label_path"]), ROOT / cfg["feature_path"], ROOT / cfg["existing_assignments"]]}
    if (output / "config.json").exists():
        assert json.loads((output / "config.json").read_text()) == cfg, "Config/input changed: use a fresh result directory"
    dump(output / "config.json", cfg)
    torch.set_num_threads(cfg["threads"])
    if args.postprocess_only:
        postprocess(cfg, output)
        return
    visits, images, assignments, eligible = prepare(cfg, output)
    if args.derive_only:
        return
    for fold in args.folds:
        run_fold(fold, cfg, output, visits, images, assignments, eligible, torch.device(args.device))
    postprocess(cfg, output)


if __name__ == "__main__":
    main()
