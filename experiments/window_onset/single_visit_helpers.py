#!/usr/bin/env python3
"""
Single-visit two-year first-onset SIC prediction.

This script is intentionally self-contained so the original DTEK-GT training
code and results remain untouched. It reuses the visit-level ResNet-50 image
descriptors and the existing patient-level five-fold split, then builds a
single-visit landmark dataset with strict right-censoring rules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(all="ignore")


SETTING_A = "First-visit training / First-visit evaluation"
SETTING_B = "Rolling-landmark training / First-visit evaluation"
SETTING_C = "Rolling-landmark training / Rolling evaluation"

CORE_MODEL_NAMES = [
    "Age + sex logistic regression",
    "TgAb + TPOAb + TSH logistic regression",
    "All clinical variables logistic regression",
    "All clinical variables + missingness mask logistic regression",
    "Clinical variables HistGradientBoosting",
    "Missingness-only logistic regression",
    "Image-only classifier",
    "Tabular-only MLP",
    "Single-visit image-tabular fusion model",
    "Single-node DTEK-GT",
]

ABLATION_MODEL_NAMES = [
    "No-antibody-values",
    "No-antibody-values-and-masks",
    "Image + clinical without missingness mask",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: dict) -> None:
    for key in ["output_dir", "figure_dir", "checkpoint_dir"]:
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ensure_dirs(cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.json"),
    )
    parser.add_argument("--derive-only", action="store_true", help="Only build samples and leakage audit.")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Use existing prediction CSVs to regenerate aggregate tables, bootstrap CIs, and figures.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny training smoke test.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional exact model names to run. Default: all core and ablation models.",
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        default=None,
        choices=["A", "B", "C"],
        help="Settings to run. Default: A B C.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="Optional fold numbers, e.g. 1 2.")
    return parser.parse_args()


def normalize_patient_id(values: Iterable) -> pd.Series:
    return pd.Series(values, dtype="object").astype(str).str.replace(r"\.0$", "", regex=True)


def load_visit_table(cfg: dict) -> Tuple[pd.DataFrame, np.ndarray]:
    label_path = cfg["label_path"]
    feature_path = cfg["feature_path"]
    patient_col = cfg["patient_id_column"]
    date_col = cfg["visit_date_column"]
    label_col = cfg["label_column"]

    raw = pd.read_excel(label_path)
    required = [patient_col, date_col, label_col] + cfg["clinical_variables"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in label table: {missing}")

    label_df = raw[required].copy()
    label_df["patient_id"] = normalize_patient_id(label_df[patient_col]).values
    label_df["visit_date"] = pd.to_datetime(label_df[date_col], errors="raise").dt.normalize()
    label_df["sic_label"] = pd.to_numeric(label_df[label_col], errors="raise").astype(int)
    label_df["_label_row"] = np.arange(len(label_df))
    label_df = label_df.sort_values(["patient_id", "visit_date", "_label_row"]).reset_index(drop=True)
    label_df["_visit_date_occurrence"] = label_df.groupby(["patient_id", "visit_date"]).cumcount()

    for var in cfg["clinical_variables"]:
        label_df[f"missing_{var}"] = label_df[var].isna().astype(int)

    npz = np.load(feature_path, allow_pickle=True)
    features = npz["features"] if "features" in npz.files else npz["feats"]
    patients = npz["patients"]
    dates = npz["dates"]
    labels = npz["labels"]
    feat_df = pd.DataFrame(
        {
            "patient_id": normalize_patient_id(patients).values,
            "visit_date": pd.to_datetime(pd.Series(dates).astype(str), errors="raise").dt.normalize(),
            "feature_row": np.arange(len(patients), dtype=int),
            "feature_label": labels.astype(int),
        }
    )
    feat_df = feat_df.sort_values(["patient_id", "visit_date", "feature_row"]).reset_index(drop=True)
    feat_df["_visit_date_occurrence"] = feat_df.groupby(["patient_id", "visit_date"]).cumcount()

    merged = label_df.merge(
        feat_df,
        on=["patient_id", "visit_date", "_visit_date_occurrence"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(label_df) or len(merged) != len(feat_df):
        raise ValueError(
            f"Could not one-to-one align labels and image features: labels={len(label_df)}, "
            f"features={len(feat_df)}, merged={len(merged)}"
        )
    mismatched = int((merged["sic_label"].astype(int) != merged["feature_label"].astype(int)).sum())
    if mismatched:
        raise ValueError(f"SIC labels disagree between Excel and feature cache for {mismatched} visits.")

    keep_cols = [
        "patient_id",
        "visit_date",
        "sic_label",
        "feature_row",
        "_label_row",
    ] + cfg["clinical_variables"] + [f"missing_{v}" for v in cfg["clinical_variables"]]
    visit_df = merged[keep_cols].copy()
    visit_df = visit_df.sort_values(["patient_id", "visit_date", "_label_row"]).reset_index(drop=True)
    visit_df["visit_id"] = visit_df.groupby("patient_id").cumcount().map(lambda x: f"V{x + 1}")
    return visit_df, features.astype(np.float32, copy=False)


def derive_single_visit_samples(visit_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    horizon = int(cfg["horizon_days"])
    rows: List[dict] = []
    clinical_vars = cfg["clinical_variables"]

    for patient_id, g in visit_df.groupby("patient_id", sort=True):
        g = g.sort_values(["visit_date", "_label_row"]).reset_index(drop=True)
        positive_dates = g.loc[g["sic_label"].astype(int) == 1, "visit_date"]
        onset_date = positive_dates.iloc[0] if len(positive_dates) else pd.NaT
        has_onset = pd.notna(onset_date)
        last_observed = g["visit_date"].max()

        for anchor_index, row in g.iterrows():
            anchor_date = row["visit_date"]
            current_sic = int(row["sic_label"])
            if current_sic != 0:
                continue
            if has_onset and anchor_date >= onset_date:
                continue

            days_to_onset = np.nan
            label = np.nan
            censored = 1
            if has_onset:
                days_to_onset = int((onset_date - anchor_date).days)
            followup_days = int((last_observed - anchor_date).days)
            horizon_end = anchor_date + pd.Timedelta(days=horizon)

            if has_onset and 0 < days_to_onset <= horizon:
                label = 1
                censored = 0
            elif last_observed >= horizon_end:
                label = 0
                censored = 0

            out = {
                "sample_id": f"{patient_id}__{anchor_date.date()}__{anchor_index}",
                "patient_id": patient_id,
                "anchor_visit_id": row["visit_id"],
                "anchor_date": anchor_date.date().isoformat(),
                "anchor_index": int(anchor_index),
                "label": label,
                "censored": int(censored),
                "onset_date": onset_date.date().isoformat() if has_onset else "",
                "days_to_onset": days_to_onset,
                "last_observed_date": last_observed.date().isoformat(),
                "followup_days_after_anchor": followup_days,
                "is_first_visit": int(anchor_index == 0),
                "current_sic": current_sic,
                "feature_row": int(row["feature_row"]),
            }
            for var in clinical_vars:
                out[var] = row[var]
                out[f"missing_{var}"] = int(row[f"missing_{var}"])
            rows.append(out)

    samples = pd.DataFrame(rows)
    if samples.empty:
        raise RuntimeError("No candidate SIC-negative anchor visits were generated.")
    return samples


def load_existing_folds(cfg: dict) -> List[dict]:
    with open(cfg["split_manifest_path"], "r", encoding="utf-8") as f:
        manifest = json.load(f)

    folds = []
    if isinstance(manifest, dict):
        for key in sorted(manifest.keys(), key=lambda k: int(str(k).split("_")[-1]) if "_" in str(k) else str(k)):
            item = manifest[key]
            fold_id = int(str(key).split("_")[-1]) if "fold_" in str(key) else int(item.get("fold", len(folds) + 1))
            test_ids = item.get("heldout_patient_ids") or item.get("test_patient_ids")
            folds.append(
                {
                    "fold": fold_id,
                    "train_patient_ids": set(map(str, item["train_patient_ids"])),
                    "test_patient_ids": set(map(str, test_ids)),
                }
            )
    elif isinstance(manifest, list):
        for item in manifest:
            fold_id = int(item.get("fold", len(folds)))
            if fold_id == 0:
                fold_id = len(folds) + 1
            test_ids = item.get("heldout_patient_ids") or item.get("test_patient_ids")
            folds.append(
                {
                    "fold": fold_id,
                    "train_patient_ids": set(map(str, item["train_patient_ids"])),
                    "test_patient_ids": set(map(str, test_ids)),
                }
            )
    else:
        raise ValueError("Unsupported split manifest format.")
    return folds


def patient_strata(samples: pd.DataFrame, patient_ids: Sequence[str]) -> np.ndarray:
    eligible = samples[samples["censored"] == 0].copy()
    first = eligible[eligible["is_first_visit"] == 1]
    first_label = first.groupby("patient_id")["label"].max().to_dict()
    roll_label = eligible.groupby("patient_id")["label"].max().to_dict()
    labels = []
    for pid in patient_ids:
        labels.append(int(first_label.get(pid, roll_label.get(pid, 0))))
    return np.asarray(labels, dtype=int)


def add_validation_split(samples: pd.DataFrame, folds: List[dict], cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    expanded_rows: List[pd.DataFrame] = []
    derivation_rows: List[dict] = []
    seed = int(cfg["seed"])
    val_ratio = float(cfg["validation_ratio"])

    for fold in folds:
        fold_id = int(fold["fold"])
        train_pool = np.array(sorted(fold["train_patient_ids"]), dtype=object)
        test_patients = set(fold["test_patient_ids"])

        y = patient_strata(samples, train_pool)
        if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + fold_id)
            train_idx, val_idx = next(splitter.split(train_pool, y))
        else:
            rng = np.random.default_rng(seed + fold_id)
            idx = np.arange(len(train_pool))
            rng.shuffle(idx)
            n_val = max(1, int(round(len(idx) * val_ratio)))
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]

        train_patients = set(map(str, train_pool[train_idx]))
        val_patients = set(map(str, train_pool[val_idx]))
        assert train_patients.isdisjoint(val_patients)
        assert train_patients.isdisjoint(test_patients)
        assert val_patients.isdisjoint(test_patients)

        split_map = {pid: "train" for pid in train_patients}
        split_map.update({pid: "val" for pid in val_patients})
        split_map.update({pid: "test" for pid in test_patients})
        cur = samples[samples["patient_id"].isin(split_map)].copy()
        cur["fold"] = fold_id
        cur["split"] = cur["patient_id"].map(split_map)
        expanded_rows.append(cur)

        for split_name, pids in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
            s = cur[cur["split"] == split_name]
            eligible = s[s["censored"] == 0]
            first = eligible[eligible["is_first_visit"] == 1]
            derivation_rows.append(
                {
                    "fold": fold_id,
                    "split": split_name,
                    "total_patients": int(len(pids)),
                    "candidate_anchor_visits": int(len(s)),
                    "eligible_samples": int(len(eligible)),
                    "positive_samples": int((eligible["label"] == 1).sum()),
                    "negative_samples": int((eligible["label"] == 0).sum()),
                    "censored_samples": int((s["censored"] == 1).sum()),
                    "first_visit_samples": int(len(first)),
                    "rolling_samples": int(len(eligible)),
                    "positive_rate": float(eligible["label"].mean()) if len(eligible) else np.nan,
                }
            )

    expanded = pd.concat(expanded_rows, ignore_index=True)
    derivation = pd.DataFrame(derivation_rows)
    return expanded, derivation


def leakage_audit(samples_by_fold: pd.DataFrame, derivation: pd.DataFrame, cfg: dict) -> dict:
    eligible = samples_by_fold[samples_by_fold["censored"] == 0].copy()
    positives = eligible[eligible["label"] == 1]
    negatives = eligible[eligible["label"] == 0]
    censored = samples_by_fold[samples_by_fold["censored"] == 1]

    checks = {}
    checks["all_anchor_visits_are_sic_negative"] = {
        "passed": bool((samples_by_fold["current_sic"] == 0).all()),
        "violations": int((samples_by_fold["current_sic"] != 0).sum()),
    }
    anchor_dates = pd.to_datetime(samples_by_fold["anchor_date"])
    onset_dates = pd.to_datetime(samples_by_fold["onset_date"], errors="coerce")
    invalid_after_onset = onset_dates.notna() & (anchor_dates >= onset_dates)
    checks["no_anchor_on_or_after_first_sic_positive_visit"] = {
        "passed": bool(~invalid_after_onset.any()),
        "violations": int(invalid_after_onset.sum()),
    }
    pos_days = positives["days_to_onset"].astype(float)
    pos_ok = pos_days.between(1, int(cfg["horizon_days"]), inclusive="both")
    checks["positive_labels_have_onset_within_1_to_730_days"] = {
        "passed": bool(pos_ok.all()),
        "violations": int((~pos_ok).sum()),
    }
    neg_ok = negatives["followup_days_after_anchor"].astype(float) >= int(cfg["horizon_days"])
    checks["negative_labels_have_complete_two_year_followup"] = {
        "passed": bool(neg_ok.all()),
        "violations": int((~neg_ok).sum()),
    }
    checks["censored_samples_not_labeled_negative"] = {
        "passed": bool(censored["label"].isna().all()),
        "violations": int(censored["label"].notna().sum()),
    }

    primary_test = eligible[(eligible["split"] == "test") & (eligible["is_first_visit"] == 1)]
    dup = primary_test.groupby(["fold", "patient_id"]).size()
    checks["primary_first_visit_test_has_at_most_one_sample_per_patient"] = {
        "passed": bool((dup <= 1).all()),
        "violations": int((dup > 1).sum()),
    }
    checks["primary_first_visit_test_uses_true_earliest_visit"] = {
        "passed": bool((primary_test["anchor_index"] == 0).all()),
        "violations": int((primary_test["anchor_index"] != 0).sum()),
    }

    split_overlap_violations = 0
    for fold_id, g in samples_by_fold.groupby("fold"):
        train = set(g.loc[g["split"] == "train", "patient_id"])
        val = set(g.loc[g["split"] == "val", "patient_id"])
        test = set(g.loc[g["split"] == "test", "patient_id"])
        split_overlap_violations += len(train & val) + len(train & test) + len(val & test)
    checks["same_patient_never_crosses_split_within_fold"] = {
        "passed": split_overlap_violations == 0,
        "violations": int(split_overlap_violations),
    }
    checks["input_feature_columns_exclude_future_information"] = {
        "passed": True,
        "input_columns": cfg["clinical_variables"] + [f"missing_{v}" for v in cfg["clinical_variables"]] + ["feature_row"],
        "metadata_not_used_as_input": [
            "anchor_index",
            "onset_date",
            "days_to_onset",
            "last_observed_date",
            "followup_days_after_anchor",
            "is_first_visit",
            "split",
            "fold",
        ],
    }
    checks["preprocessing_statistics_fit_on_train_split_only"] = {
        "passed": True,
        "details": "Median imputation, tabular standardization, and image standardization are fitted inside each fold using train samples only.",
    }
    checks["first_sic_positive_date_used_for_labeling"] = {
        "passed": True,
        "details": "onset_date is the earliest visit_date with SIC=1 per patient.",
    }
    checks["boundary_730_days_included_for_positive_window"] = {
        "passed": True,
        "details": "Positive window is (anchor_date, anchor_date + 730 days].",
    }

    all_passed = all(v.get("passed", False) for v in checks.values())
    return {
        "experiment": cfg["experiment_name"],
        "horizon_days": int(cfg["horizon_days"]),
        "all_passed": bool(all_passed),
        "checks": checks,
        "sample_derivation_summary": derivation.to_dict(orient="records"),
    }


@dataclass
class FeatureSpec:
    name: str
    estimator: str
    include_image: bool = False
    value_vars: Tuple[str, ...] = ()
    mask_vars: Tuple[str, ...] = ()
    use_gate: bool = False
    single_node_transformer: bool = False
    comparison_group: str = "core"


def build_model_specs(cfg: dict) -> Dict[str, FeatureSpec]:
    clinical = tuple(cfg["clinical_variables"])
    masks = tuple(cfg["clinical_variables"])
    no_ab_values = tuple(v for v in cfg["clinical_variables"] if v not in cfg["antibody_variables"])
    no_ab_masks = tuple(v for v in cfg["clinical_variables"] if v not in cfg["antibody_variables"])
    specs = [
        FeatureSpec("Age + sex logistic regression", "logreg", value_vars=("age", "sex")),
        FeatureSpec("TgAb + TPOAb + TSH logistic regression", "logreg", value_vars=("TgAb", "TPOAb", "TSH")),
        FeatureSpec("All clinical variables logistic regression", "logreg", value_vars=clinical),
        FeatureSpec(
            "All clinical variables + missingness mask logistic regression",
            "logreg",
            value_vars=clinical,
            mask_vars=masks,
        ),
        FeatureSpec("Clinical variables HistGradientBoosting", "histgb", value_vars=clinical),
        FeatureSpec("Missingness-only logistic regression", "logreg", mask_vars=masks),
        FeatureSpec("Image-only classifier", "image_mlp", include_image=True),
        FeatureSpec("Tabular-only MLP", "tabular_mlp", value_vars=clinical, mask_vars=masks),
        FeatureSpec(
            "Single-visit image-tabular fusion model",
            "fusion_mlp",
            include_image=True,
            value_vars=clinical,
            mask_vars=masks,
            use_gate=True,
        ),
        FeatureSpec(
            "Single-node DTEK-GT",
            "single_node_dtek_gt",
            include_image=True,
            value_vars=clinical,
            mask_vars=masks,
            use_gate=True,
            single_node_transformer=True,
        ),
        FeatureSpec(
            "No-antibody-values",
            "fusion_mlp",
            include_image=True,
            value_vars=no_ab_values,
            mask_vars=masks,
            use_gate=True,
            comparison_group="ablation",
        ),
        FeatureSpec(
            "No-antibody-values-and-masks",
            "fusion_mlp",
            include_image=True,
            value_vars=no_ab_values,
            mask_vars=no_ab_masks,
            use_gate=True,
            comparison_group="ablation",
        ),
        FeatureSpec(
            "Image + clinical without missingness mask",
            "fusion_mlp",
            include_image=True,
            value_vars=clinical,
            mask_vars=(),
            use_gate=True,
            comparison_group="ablation",
        ),
    ]
    return {s.name: s for s in specs}


class FoldPreprocessor:
    def __init__(self, cfg: dict, spec: FeatureSpec):
        self.cfg = cfg
        self.spec = spec
        self.value_medians: Optional[pd.Series] = None
        self.value_means: Optional[pd.Series] = None
        self.value_stds: Optional[pd.Series] = None
        self.image_mean: Optional[np.ndarray] = None
        self.image_std: Optional[np.ndarray] = None

    def fit(self, train_df: pd.DataFrame, image_features: np.ndarray) -> None:
        if self.spec.value_vars:
            vals = train_df[list(self.spec.value_vars)].apply(pd.to_numeric, errors="coerce")
            self.value_medians = vals.median(axis=0, skipna=True).fillna(0.0)
            filled = vals.fillna(self.value_medians)
            self.value_means = filled.mean(axis=0)
            stds = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
            self.value_stds = stds
        if self.spec.include_image:
            rows = train_df["feature_row"].astype(int).to_numpy()
            imgs = image_features[rows].astype(np.float32)
            self.image_mean = imgs.mean(axis=0)
            self.image_std = imgs.std(axis=0)
            self.image_std[self.image_std < 1e-6] = 1.0

    def transform(self, df: pd.DataFrame, image_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        blocks: List[np.ndarray] = []
        dims = {"image_dim": 0, "value_dim": 0, "mask_dim": 0}
        if self.spec.include_image:
            rows = df["feature_row"].astype(int).to_numpy()
            img = image_features[rows].astype(np.float32)
            img = (img - self.image_mean) / self.image_std
            blocks.append(img.astype(np.float32))
            dims["image_dim"] = int(img.shape[1])
        if self.spec.value_vars:
            vals = df[list(self.spec.value_vars)].apply(pd.to_numeric, errors="coerce")
            vals = vals.fillna(self.value_medians)
            vals = (vals - self.value_means) / self.value_stds
            arr = vals.to_numpy(dtype=np.float32)
            blocks.append(arr)
            dims["value_dim"] = int(arr.shape[1])
        if self.spec.mask_vars:
            mask_cols = [f"missing_{v}" for v in self.spec.mask_vars]
            masks = df[mask_cols].astype(float).to_numpy(dtype=np.float32)
            blocks.append(masks)
            dims["mask_dim"] = int(masks.shape[1])
        if not blocks:
            raise ValueError(f"No input features selected for {self.spec.name}")
        x = np.concatenate(blocks, axis=1).astype(np.float32)
        y = df["label"].astype(int).to_numpy(dtype=np.int64)
        return x, y, dims


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = np.ones_like(y, dtype=np.float32)
    for cls in [0, 1]:
        if counts[cls] > 0:
            weights[y == cls] = len(y) / (2.0 * counts[cls])
    return weights


def compute_sample_weights(train_df: pd.DataFrame, patient_balanced: bool) -> np.ndarray:
    weights = class_balanced_weights(train_df["label"].astype(int).to_numpy())
    if patient_balanced:
        counts = train_df.groupby("patient_id")["sample_id"].transform("count").astype(float).to_numpy()
        weights = weights / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    best_t, best_score = 0.5, -np.inf
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            score = accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-5, 1 - 1e-5)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    logits = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
    try:
        lr = LogisticRegression(max_iter=1000, solver="liblinear")
        lr.fit(logits, y_true)
        return float(lr.intercept_[0]), float(lr.coef_[0][0])
    except Exception:
        return float("nan"), float("nan")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "AUROC": safe_auc(y_true, y_prob),
        "AUPRC": safe_auprc(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp + 1e-12)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))),
        "ECE": compute_ece(y_true, y_prob),
        "Calibration intercept": intercept,
        "Calibration slope": slope,
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def compute_metrics_from_pred(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "AUROC": safe_auc(y_true, y_prob),
        "AUPRC": safe_auprc(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp + 1e-12)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))),
        "ECE": compute_ece(y_true, y_prob),
        "Calibration intercept": intercept,
        "Calibration slope": slope,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


class PlattCalibrator:
    def __init__(self):
        self.model: Optional[LogisticRegression] = None
        self.identity = False

    @staticmethod
    def _logit(prob: np.ndarray) -> np.ndarray:
        prob = np.clip(prob, 1e-5, 1 - 1e-5)
        return np.log(prob / (1 - prob)).reshape(-1, 1)

    def fit(self, val_prob: np.ndarray, y_val: np.ndarray) -> "PlattCalibrator":
        if len(np.unique(y_val)) < 2:
            self.identity = True
            return self
        self.model = LogisticRegression(max_iter=1000, solver="liblinear")
        self.model.fit(self._logit(val_prob), y_val.astype(int))
        return self

    def predict(self, prob: np.ndarray) -> np.ndarray:
        if self.identity or self.model is None:
            return np.asarray(prob, dtype=float)
        return self.model.predict_proba(self._logit(prob))[:, 1]


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.weights = torch.tensor(weights if weights is not None else np.ones(len(y)), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.weights[idx]


class BasicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ReliabilityFusionBlock(nn.Module):
    def __init__(self, image_dim: int, tab_dim: int, mask_dim: int, fusion_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if image_dim <= 0 or tab_dim <= 0:
            raise ValueError("Reliability fusion requires image and tabular inputs.")
        self.image_dim = image_dim
        self.tab_dim = tab_dim
        self.mask_dim = mask_dim
        self.img_mlp = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Linear(2 * fusion_dim + mask_dim, 2)
        self.out = nn.Sequential(
            nn.Linear(3 * fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = x[:, : self.image_dim]
        tab = x[:, self.image_dim : self.image_dim + self.tab_dim]
        img_z = self.img_mlp(img)
        tab_z = self.tab_mlp(tab)
        if self.mask_dim > 0:
            mask = tab[:, -self.mask_dim :]
            gate_input = torch.cat([img_z, tab_z, mask], dim=-1)
        else:
            gate_input = torch.cat([img_z, tab_z], dim=-1)
        gate = torch.softmax(self.gate(gate_input), dim=-1)
        fusion = torch.cat([gate[:, 0:1] * img_z, gate[:, 1:2] * tab_z, img_z * tab_z], dim=-1)
        return self.out(fusion)


class FusionMLP(nn.Module):
    def __init__(self, image_dim: int, tab_dim: int, mask_dim: int, fusion_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fusion = ReliabilityFusionBlock(image_dim, tab_dim, mask_dim, fusion_dim, hidden_dim, dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.fusion(x)).squeeze(-1)


class SingleNodeDTEKGT(nn.Module):
    def __init__(self, image_dim: int, tab_dim: int, mask_dim: int, fusion_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fusion = ReliabilityFusionBlock(image_dim, tab_dim, mask_dim, fusion_dim, hidden_dim, dropout)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token = self.fusion(x).unsqueeze(1)
        encoded = self.transformer(token).squeeze(1)
        return self.head(encoded).squeeze(-1)


def build_torch_model(spec: FeatureSpec, dims: Dict[str, int], cfg: dict) -> nn.Module:
    input_dim = dims["image_dim"] + dims["value_dim"] + dims["mask_dim"]
    if spec.estimator in {"image_mlp", "tabular_mlp"}:
        return BasicMLP(input_dim=input_dim, hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"]))
    tab_dim = dims["value_dim"] + dims["mask_dim"]
    if spec.estimator == "fusion_mlp":
        return FusionMLP(
            image_dim=dims["image_dim"],
            tab_dim=tab_dim,
            mask_dim=dims["mask_dim"],
            fusion_dim=int(cfg["fusion_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
        )
    if spec.estimator == "single_node_dtek_gt":
        return SingleNodeDTEKGT(
            image_dim=dims["image_dim"],
            tab_dim=tab_dim,
            mask_dim=dims["mask_dim"],
            fusion_dim=int(cfg["fusion_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
        )
    raise ValueError(f"Unsupported torch estimator: {spec.estimator}")


def predict_torch(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    loader = torch.utils.data.DataLoader(TensorDataset(x, np.zeros(len(x))), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, _, _ in loader:
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def fit_torch(
    spec: FeatureSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    dims: Dict[str, int],
    cfg: dict,
    fold_id: int,
    setting_code: str,
    patient_balanced: bool,
) -> Tuple[object, np.ndarray, np.ndarray, dict]:
    set_seed(int(cfg["seed"]) + int(fold_id))
    model = build_torch_model(spec, dims, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    train_ds = TensorDataset(x_train, y_train, w_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["patience"])
    best_state = None
    best_score = -np.inf
    wait = 0
    start = time.perf_counter()

    for epoch in range(max_epochs):
        model.train()
        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss_each = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (loss_each * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_prob = predict_torch(model, x_val, batch_size=int(cfg["batch_size"]) * 4)
        score = safe_auprc(y_val, val_prob)
        if math.isnan(score):
            score = -np.inf
        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    elapsed = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    val_prob = predict_torch(model, x_val, batch_size=int(cfg["batch_size"]) * 4)

    ckpt_dir = Path(cfg["checkpoint_dir"]) / f"fold_{fold_id}" / setting_code
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safe_name = spec.name.replace(" ", "_").replace("/", "_").replace("+", "plus")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "spec": spec.__dict__,
            "dims": dims,
            "cfg": cfg,
        },
        ckpt_dir / f"{safe_name}__patient_balanced_{int(patient_balanced)}.pt",
    )
    info = {
        "epochs_ran": int(epoch + 1),
        "training_time_sec": float(elapsed),
        "best_validation_auprc": float(best_score),
    }
    return model, val_prob, w_train, info


def fit_sklearn(spec: FeatureSpec, x_train, y_train, w_train, x_val, cfg) -> Tuple[object, np.ndarray, dict]:
    if spec.estimator == "logreg":
        model = LogisticRegression(max_iter=2000, solver="liblinear")
        model.fit(x_train, y_train, sample_weight=w_train)
    elif spec.estimator == "histgb":
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=int(cfg["seed"]))
        model.fit(x_train, y_train, sample_weight=w_train)
    else:
        raise ValueError(f"Unsupported sklearn estimator: {spec.estimator}")
    val_prob = model.predict_proba(x_val)[:, 1]
    return model, val_prob, {"epochs_ran": np.nan, "training_time_sec": np.nan, "best_validation_auprc": safe_auprc(y_train, model.predict_proba(x_train)[:, 1])}


def predict_model(model: object, spec: FeatureSpec, x: np.ndarray, cfg: dict) -> np.ndarray:
    if spec.estimator in {"logreg", "histgb"}:
        return model.predict_proba(x)[:, 1]
    return predict_torch(model, x, batch_size=int(cfg["batch_size"]) * 4)


def get_setting_data(samples_by_fold: pd.DataFrame, fold_id: int, setting: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold = samples_by_fold[(samples_by_fold["fold"] == fold_id) & (samples_by_fold["censored"] == 0)].copy()
    if setting == SETTING_A:
        train = fold[(fold["split"] == "train") & (fold["is_first_visit"] == 1)]
        val = fold[(fold["split"] == "val") & (fold["is_first_visit"] == 1)]
        test = fold[(fold["split"] == "test") & (fold["is_first_visit"] == 1)]
    elif setting == SETTING_B:
        train = fold[fold["split"] == "train"]
        val = fold[(fold["split"] == "val") & (fold["is_first_visit"] == 1)]
        test = fold[(fold["split"] == "test") & (fold["is_first_visit"] == 1)]
    elif setting == SETTING_C:
        train = fold[fold["split"] == "train"]
        val = fold[fold["split"] == "val"]
        test = fold[fold["split"] == "test"]
    else:
        raise ValueError(setting)
    return train.copy(), val.copy(), test.copy()


def run_one_model_fold(
    spec: FeatureSpec,
    samples_by_fold: pd.DataFrame,
    image_features: np.ndarray,
    fold_id: int,
    setting_code: str,
    setting: str,
    patient_balanced: bool,
    cfg: dict,
) -> Tuple[List[dict], pd.DataFrame, pd.DataFrame, dict]:
    train_df, val_df, test_df = get_setting_data(samples_by_fold, fold_id, setting)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"No data for fold={fold_id}, setting={setting}; train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    if len(np.unique(train_df["label"].astype(int))) < 2:
        raise RuntimeError(f"Training set has one class only for fold={fold_id}, setting={setting}, model={spec.name}")

    pre = FoldPreprocessor(cfg, spec)
    pre.fit(train_df, image_features)
    x_train, y_train, dims = pre.transform(train_df, image_features)
    x_val, y_val, _ = pre.transform(val_df, image_features)
    x_test, y_test, _ = pre.transform(test_df, image_features)
    w_train = compute_sample_weights(train_df, patient_balanced=patient_balanced)

    if spec.estimator in {"logreg", "histgb"}:
        model, val_raw, train_info = fit_sklearn(spec, x_train, y_train, w_train, x_val, cfg)
    else:
        model, val_raw, _, train_info = fit_torch(
            spec,
            x_train,
            y_train,
            w_train,
            x_val,
            y_val,
            dims,
            cfg,
            fold_id=fold_id,
            setting_code=setting_code,
            patient_balanced=patient_balanced,
        )
    test_raw = predict_model(model, spec, x_test, cfg)

    raw_threshold = find_best_threshold(y_val, val_raw, metric=cfg["threshold_metric"])
    calibrator = PlattCalibrator().fit(val_raw, y_val)
    val_cal = calibrator.predict(val_raw)
    test_cal = calibrator.predict(test_raw)
    cal_threshold = find_best_threshold(y_val, val_cal, metric=cfg["threshold_metric"])

    rows = []
    for score_type, y_prob, threshold in [
        ("raw_score", test_raw, raw_threshold),
        ("calibrated_probability", test_cal, cal_threshold),
    ]:
        m = compute_metrics(y_test, y_prob, threshold)
        row = {
            "fold": fold_id,
            "setting": setting,
            "setting_code": setting_code,
            "model_name": spec.name,
            "score_type": score_type,
            "patient_balanced_weighting": bool(patient_balanced),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "train_positive": int(train_df["label"].sum()),
            "val_positive": int(val_df["label"].sum()),
            "test_positive": int(test_df["label"].sum()),
        }
        row.update(m)
        row.update(train_info)
        rows.append(row)

    pred_df = test_df[
        [
            "patient_id",
            "anchor_visit_id",
            "anchor_date",
            "fold",
            "label",
            "sample_id",
            "is_first_visit",
        ]
    ].copy()
    pred_df = pred_df.rename(columns={"label": "true_label"})
    pred_df["raw_score"] = test_raw
    pred_df["calibrated_probability"] = test_cal
    pred_df["threshold"] = cal_threshold
    pred_df["selected_threshold"] = cal_threshold
    pred_df["predicted_label"] = (test_cal >= cal_threshold).astype(int)
    pred_df["model_name"] = spec.name
    pred_df["training_setting"] = setting
    pred_df["setting_code"] = setting_code
    pred_df["patient_balanced_weighting"] = bool(patient_balanced)

    cal_rows = []
    for score_type, prob, threshold in [
        ("raw_score", test_raw, raw_threshold),
        ("calibrated_probability", test_cal, cal_threshold),
    ]:
        m = compute_metrics(y_test, prob, threshold)
        cal_rows.append(
            {
                "fold": fold_id,
                "setting": setting,
                "setting_code": setting_code,
                "model_name": spec.name,
                "score_type": score_type,
                "patient_balanced_weighting": bool(patient_balanced),
                "AUROC": m["AUROC"],
                "AUPRC": m["AUPRC"],
                "Brier": m["Brier"],
                "ECE": m["ECE"],
                "Calibration intercept": m["Calibration intercept"],
                "Calibration slope": m["Calibration slope"],
            }
        )
    return rows, pred_df, pd.DataFrame(cal_rows), train_info


def aggregate_metrics(fold_metrics: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metric_cols = [
        "AUROC",
        "AUPRC",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "Brier",
        "ECE",
        "Calibration intercept",
        "Calibration slope",
    ]
    rows = []
    for keys, g in fold_metrics.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1))
        row["n_folds"] = int(g["fold"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_patient_ci(preds: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    metric_names = ["AUROC", "AUPRC", "Accuracy", "F1", "Specificity", "Brier"]
    for (model, setting, balanced), g in preds.groupby(["model_name", "training_setting", "patient_balanced_weighting"]):
        g = g.reset_index(drop=True)
        patients = np.array(sorted(g["patient_id"].unique()), dtype=object)
        if len(patients) < 2:
            continue
        by_patient = {
            pid: idx.to_numpy(dtype=int)
            for pid, idx in g.groupby("patient_id").groups.items()
        }
        y_all = g["true_label"].to_numpy(dtype=int)
        p_all = g["calibrated_probability"].to_numpy(dtype=float)
        pred_all = g["predicted_label"].to_numpy(dtype=int)
        observed = compute_metrics_from_pred(y_all, p_all, pred_all)
        samples = {m: [] for m in metric_names}
        for _ in range(n_boot):
            draw = rng.choice(patients, size=len(patients), replace=True)
            idx = np.concatenate([by_patient[pid] for pid in draw])
            if len(np.unique(y_all[idx])) < 2:
                continue
            m = compute_metrics_from_pred(y_all[idx], p_all[idx], pred_all[idx])
            for name in metric_names:
                samples[name].append(m[name])
        for name in metric_names:
            vals = np.asarray(samples[name], dtype=float)
            rows.append(
                {
                    "model_name": model,
                    "training_setting": setting,
                    "patient_balanced_weighting": balanced,
                    "metric": name,
                    "estimate": observed[name],
                    "ci_lower": float(np.nanpercentile(vals, 2.5)) if len(vals) else np.nan,
                    "ci_upper": float(np.nanpercentile(vals, 97.5)) if len(vals) else np.nan,
                    "bootstrap_iterations": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)


def run_postprocess(cfg: dict) -> None:
    out_dir = Path(cfg["output_dir"])
    fold_metrics = pd.read_csv(out_dir / "fold_metrics.csv")
    first_path = out_dir / "oof_first_visit_predictions.csv"
    rolling_path = out_dir / "oof_rolling_predictions.csv"
    first_preds = pd.read_csv(first_path) if first_path.exists() and first_path.stat().st_size > 0 else pd.DataFrame()
    rolling_preds = pd.read_csv(rolling_path) if rolling_path.exists() and rolling_path.stat().st_size > 0 else pd.DataFrame()
    preds = pd.concat([first_preds, rolling_preds], ignore_index=True) if not first_preds.empty or not rolling_preds.empty else pd.DataFrame()
    ok_metrics = fold_metrics[fold_metrics["score_type"].isin(["raw_score", "calibrated_probability"])].copy()

    model_comparison = aggregate_metrics(
        ok_metrics[ok_metrics["score_type"] == "calibrated_probability"],
        ["model_name", "setting", "patient_balanced_weighting"],
    )
    model_comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    strategy = aggregate_metrics(
        ok_metrics[
            (ok_metrics["score_type"] == "calibrated_probability")
            & (ok_metrics["model_name"] == "Single-node DTEK-GT")
        ],
        ["setting", "patient_balanced_weighting"],
    )
    if strategy.empty:
        strategy = pd.DataFrame(columns=["setting", "patient_balanced_weighting", "n_folds"])
    strategy.to_csv(out_dir / "training_strategy_comparison.csv", index=False)

    ablation = aggregate_metrics(
        ok_metrics[
            (ok_metrics["score_type"] == "calibrated_probability")
            & (
                ok_metrics["model_name"].isin(
                    ABLATION_MODEL_NAMES
                    + [
                        "Missingness-only logistic regression",
                        "Image-only classifier",
                        "All clinical variables logistic regression",
                    ]
                )
            )
        ],
        ["model_name", "setting", "patient_balanced_weighting"],
    )
    if ablation.empty:
        ablation = pd.DataFrame(columns=["model_name", "setting", "patient_balanced_weighting", "n_folds"])
    ablation.to_csv(out_dir / "ablation_results.csv", index=False)

    weighting = aggregate_metrics(
        ok_metrics[
            (ok_metrics["score_type"] == "calibrated_probability")
            & (ok_metrics["setting"] == SETTING_B)
        ],
        ["model_name", "patient_balanced_weighting"],
    )
    if weighting.empty:
        weighting = pd.DataFrame(columns=["model_name", "patient_balanced_weighting", "n_folds"])
    weighting.to_csv(out_dir / "patient_weighting_comparison.csv", index=False)

    if not first_preds.empty:
        ci = bootstrap_patient_ci(first_preds, n_boot=int(cfg["bootstrap_iterations"]), seed=int(cfg["seed"]))
        ci.to_csv(out_dir / "patient_bootstrap_ci_first_visit.csv", index=False)

    derivation = pd.read_csv(out_dir / "sample_derivation.csv")
    samples_all = pd.read_csv(out_dir / "rolling_samples_metadata.csv")
    plot_curves(preds, ok_metrics, derivation, samples_all, cfg)


def save_derive_outputs(samples_by_fold: pd.DataFrame, derivation: pd.DataFrame, audit: dict, cfg: dict) -> None:
    out_dir = Path(cfg["output_dir"])
    eligible = samples_by_fold[samples_by_fold["censored"] == 0].copy()
    first_all = samples_by_fold[samples_by_fold["is_first_visit"] == 1].copy()
    first_eligible = eligible[eligible["is_first_visit"] == 1].copy()

    derivation.to_csv(out_dir / "sample_derivation.csv", index=False)
    samples_by_fold.to_csv(out_dir / "rolling_samples_metadata.csv", index=False)
    first_all.to_csv(out_dir / "first_visit_samples_metadata.csv", index=False)
    first_eligible.to_csv(out_dir / "first_visit_samples.csv", index=False)
    with open(out_dir / "leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)


def plot_curves(preds: pd.DataFrame, fold_metrics: pd.DataFrame, derivation: pd.DataFrame, samples_by_fold: pd.DataFrame, cfg: dict) -> None:
    fig_dir = Path(cfg["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    if preds.empty:
        return

    primary = preds[
        (preds["training_setting"] == SETTING_B)
        & (preds["patient_balanced_weighting"] == True)
        & (preds["model_name"] == "Single-node DTEK-GT")
        & (preds["is_first_visit"] == 1)
    ].copy()
    if primary.empty:
        primary = preds[preds["is_first_visit"] == 1].copy()
        if not primary.empty:
            best_model = (
                fold_metrics[
                    (fold_metrics["score_type"] == "calibrated_probability")
                    & (fold_metrics["setting"] == SETTING_B)
                ]
                .groupby("model_name")["AUROC"]
                .mean()
                .sort_values(ascending=False)
            )
            if len(best_model):
                primary = primary[primary["model_name"] == best_model.index[0]]
    if primary.empty or len(np.unique(primary["true_label"])) < 2:
        return

    y = primary["true_label"].astype(int).to_numpy()
    p = primary["calibrated_probability"].astype(float).to_numpy()

    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(4.2, 3.4))
    plt.plot(fpr, tpr, lw=1.8, label=f"AUROC={roc_auc_score(y, p):.3f}")
    plt.plot([0, 1], [0, 1], color="0.7", lw=1.0, ls="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Primary first-visit ROC curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "primary_first_visit_roc_curve.pdf")
    plt.savefig(fig_dir / "primary_first_visit_roc_curve.png", dpi=300)
    plt.close()

    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(4.2, 3.4))
    plt.plot(rec, prec, lw=1.8, label=f"AUPRC={average_precision_score(y, p):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Primary first-visit precision-recall curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "primary_first_visit_precision_recall_curve.pdf")
    plt.savefig(fig_dir / "primary_first_visit_precision_recall_curve.png", dpi=300)
    plt.close()

    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(4.2, 3.4))
    plt.plot(prob_pred, prob_true, marker="o", lw=1.5)
    plt.plot([0, 1], [0, 1], color="0.7", lw=1.0, ls="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve")
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_curve.pdf")
    plt.savefig(fig_dir / "calibration_curve.png", dpi=300)
    plt.close()

    threshold = float(primary["threshold"].iloc[0])
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(3.5, 3.2))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("First-visit confusion matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / "first_visit_confusion_matrix.pdf")
    plt.savefig(fig_dir / "first_visit_confusion_matrix.png", dpi=300)
    plt.close()

    comp = aggregate_metrics(
        fold_metrics[
            (fold_metrics["score_type"] == "calibrated_probability")
            & (fold_metrics["setting"] == SETTING_B)
            & (fold_metrics["patient_balanced_weighting"] == True)
        ],
        ["model_name"],
    )
    if not comp.empty:
        comp = comp.sort_values("AUROC_mean", ascending=True)
        ypos = np.arange(len(comp))
        plt.figure(figsize=(7.2, max(3.6, len(comp) * 0.28)))
        plt.barh(ypos - 0.18, comp["AUROC_mean"], height=0.34, label="AUROC")
        plt.barh(ypos + 0.18, comp["AUPRC_mean"], height=0.34, label="AUPRC")
        plt.yticks(ypos, comp["model_name"])
        plt.xlabel("Mean score")
        plt.title("Model comparison on primary first-visit evaluation")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "model_auroc_auprc_comparison.pdf")
        plt.savefig(fig_dir / "model_auroc_auprc_comparison.png", dpi=300)
        plt.close()

    strategy = aggregate_metrics(
        fold_metrics[
            (fold_metrics["score_type"] == "calibrated_probability")
            & (fold_metrics["model_name"] == "Single-node DTEK-GT")
        ],
        ["setting"],
    )
    if not strategy.empty:
        plt.figure(figsize=(5.8, 3.4))
        x = np.arange(len(strategy))
        plt.bar(x - 0.18, strategy["AUROC_mean"], width=0.36, label="AUROC")
        plt.bar(x + 0.18, strategy["AUPRC_mean"], width=0.36, label="AUPRC")
        plt.xticks(x, strategy["setting"], rotation=20, ha="right")
        plt.ylabel("Mean score")
        plt.title("Training and evaluation strategy comparison")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "training_strategy_comparison.pdf")
        plt.savefig(fig_dir / "training_strategy_comparison.png", dpi=300)
        plt.close()

    brier = aggregate_metrics(fold_metrics, ["model_name", "score_type"])
    brier = brier[brier["model_name"] == "Single-node DTEK-GT"]
    if not brier.empty:
        plt.figure(figsize=(4.4, 3.2))
        plt.bar(brier["score_type"], brier["Brier_mean"], color=["0.5", "0.2"])
        plt.ylabel("Brier score")
        plt.title("Brier score before and after calibration")
        plt.tight_layout()
        plt.savefig(fig_dir / "brier_calibration_comparison.pdf")
        plt.savefig(fig_dir / "brier_calibration_comparison.png", dpi=300)
        plt.close()

    onset = samples_by_fold[(samples_by_fold["censored"] == 0) & (samples_by_fold["label"] == 1)].copy()
    if not onset.empty:
        plt.figure(figsize=(4.6, 3.2))
        plt.hist(onset["days_to_onset"].astype(float), bins=20, color="0.35")
        plt.xlabel("Days from anchor to first SIC-positive visit")
        plt.ylabel("Sample count")
        plt.title("Time from anchor to onset")
        plt.tight_layout()
        plt.savefig(fig_dir / "anchor_to_onset_time_distribution.pdf")
        plt.savefig(fig_dir / "anchor_to_onset_time_distribution.png", dpi=300)
        plt.close()

    per_patient = samples_by_fold[samples_by_fold["censored"] == 0].drop_duplicates(["patient_id", "sample_id"])
    if not per_patient.empty:
        counts = per_patient.groupby("patient_id").size()
        plt.figure(figsize=(4.6, 3.2))
        plt.hist(counts, bins=range(1, int(counts.max()) + 2), color="0.35", align="left")
        plt.xlabel("Eligible rolling samples per patient")
        plt.ylabel("Patient count")
        plt.title("Rolling sample count per patient")
        plt.tight_layout()
        plt.savefig(fig_dir / "rolling_samples_per_patient_distribution.pdf")
        plt.savefig(fig_dir / "rolling_samples_per_patient_distribution.png", dpi=300)
        plt.close()

    comp = derivation.copy()
    comp["fold_split"] = comp["fold"].astype(str) + "-" + comp["split"]
    plt.figure(figsize=(8.5, 3.8))
    x = np.arange(len(comp))
    plt.bar(x, comp["positive_samples"], label="Positive")
    plt.bar(x, comp["negative_samples"], bottom=comp["positive_samples"], label="Negative")
    plt.bar(
        x,
        comp["censored_samples"],
        bottom=comp["positive_samples"] + comp["negative_samples"],
        label="Censored",
        color="0.7",
    )
    plt.xticks(x, comp["fold_split"], rotation=60, ha="right")
    plt.ylabel("Sample count")
    plt.title("Fold sample composition")
    plt.legend(frameon=False, ncol=3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fold_sample_composition.pdf")
    plt.savefig(fig_dir / "fold_sample_composition.png", dpi=300)
    plt.close()


def run_training(samples_by_fold: pd.DataFrame, image_features: np.ndarray, cfg: dict, args: argparse.Namespace) -> None:
    specs = build_model_specs(cfg)
    model_names = args.models if args.models else list(specs.keys())
    unknown = [m for m in model_names if m not in specs]
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}\nAvailable: {list(specs)}")

    setting_map = {
        "A": SETTING_A,
        "B": SETTING_B,
        "C": SETTING_C,
    }
    settings = args.settings if args.settings else ["A", "B", "C"]
    fold_ids = [int(x) for x in sorted(samples_by_fold["fold"].unique().astype(int))]
    if args.folds:
        fold_ids = [f for f in fold_ids if f in set(args.folds)]
    if args.smoke:
        fold_ids = fold_ids[:1]
        model_names = [m for m in model_names if m in {"Age + sex logistic regression", "Single-visit image-tabular fusion model"}]
        settings = ["B"]
        cfg["max_epochs"] = min(int(cfg["max_epochs"]), 3)
        cfg["patience"] = min(int(cfg["patience"]), 2)
        cfg["bootstrap_iterations"] = min(int(cfg["bootstrap_iterations"]), 20)
    if args.max_epochs is not None:
        cfg["max_epochs"] = int(args.max_epochs)
    if args.patience is not None:
        cfg["patience"] = int(args.patience)

    fold_metric_rows: List[dict] = []
    prediction_rows: List[pd.DataFrame] = []
    calibration_rows: List[pd.DataFrame] = []

    for setting_code in settings:
        setting = setting_map[setting_code]
        for patient_balanced in ([False, True] if setting_code == "B" else [False]):
            for fold_id in fold_ids:
                for model_name in model_names:
                    spec = specs[model_name]
                    try:
                        rows, pred_df, cal_df, _ = run_one_model_fold(
                            spec,
                            samples_by_fold,
                            image_features,
                            fold_id=fold_id,
                            setting_code=setting_code,
                            setting=setting,
                            patient_balanced=patient_balanced,
                            cfg=cfg,
                        )
                    except Exception as exc:
                        fold_metric_rows.append(
                            {
                                "fold": fold_id,
                                "setting": setting,
                                "setting_code": setting_code,
                                "model_name": model_name,
                                "score_type": "failed",
                                "patient_balanced_weighting": bool(patient_balanced),
                                "error": repr(exc),
                            }
                        )
                        print(f"[WARN] failed: fold={fold_id} setting={setting_code} model={model_name}: {exc}")
                        continue
                    fold_metric_rows.extend(rows)
                    prediction_rows.append(pred_df)
                    calibration_rows.append(cal_df)
                    print(
                        f"[DONE] fold={fold_id} setting={setting_code} balanced={patient_balanced} "
                        f"model={model_name}"
                    )

    out_dir = Path(cfg["output_dir"])
    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)

    if prediction_rows:
        preds = pd.concat(prediction_rows, ignore_index=True)
    else:
        preds = pd.DataFrame()
    first_preds = preds[preds["is_first_visit"] == 1].copy() if not preds.empty else pd.DataFrame()
    rolling_preds = preds[preds["is_first_visit"] != 1].copy() if not preds.empty else pd.DataFrame()
    first_preds.to_csv(out_dir / "oof_first_visit_predictions.csv", index=False)
    rolling_preds.to_csv(out_dir / "oof_rolling_predictions.csv", index=False)

    if calibration_rows:
        calibration = pd.concat(calibration_rows, ignore_index=True)
    else:
        calibration = pd.DataFrame()
    calibration.to_csv(out_dir / "calibration_metrics.csv", index=False)

    ok_metrics = fold_metrics[fold_metrics["score_type"].isin(["raw_score", "calibrated_probability"])].copy()
    model_comparison = aggregate_metrics(
        ok_metrics[ok_metrics["score_type"] == "calibrated_probability"],
        ["model_name", "setting", "patient_balanced_weighting"],
    )
    model_comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    strategy = aggregate_metrics(
        ok_metrics[
            (ok_metrics["score_type"] == "calibrated_probability")
            & (ok_metrics["model_name"] == "Single-node DTEK-GT")
        ],
        ["setting", "patient_balanced_weighting"],
    )
    if strategy.empty:
        strategy = pd.DataFrame(columns=["setting", "patient_balanced_weighting", "n_folds"])
    strategy.to_csv(out_dir / "training_strategy_comparison.csv", index=False)

    ablation = aggregate_metrics(
        ok_metrics[
            (ok_metrics["score_type"] == "calibrated_probability")
            & (ok_metrics["model_name"].isin(ABLATION_MODEL_NAMES + ["Missingness-only logistic regression", "Image-only classifier", "All clinical variables logistic regression"]))
        ],
        ["model_name", "setting", "patient_balanced_weighting"],
    )
    if ablation.empty:
        ablation = pd.DataFrame(columns=["model_name", "setting", "patient_balanced_weighting", "n_folds"])
    ablation.to_csv(out_dir / "ablation_results.csv", index=False)

    weighting = aggregate_metrics(
        ok_metrics[
            (ok_metrics["score_type"] == "calibrated_probability")
            & (ok_metrics["setting"] == SETTING_B)
        ],
        ["model_name", "patient_balanced_weighting"],
    )
    if weighting.empty:
        weighting = pd.DataFrame(columns=["model_name", "patient_balanced_weighting", "n_folds"])
    weighting.to_csv(out_dir / "patient_weighting_comparison.csv", index=False)

    if not first_preds.empty:
        ci = bootstrap_patient_ci(first_preds, n_boot=int(cfg["bootstrap_iterations"]), seed=int(cfg["seed"]))
        ci.to_csv(out_dir / "patient_bootstrap_ci_first_visit.csv", index=False)

    derivation = pd.read_csv(out_dir / "sample_derivation.csv")
    samples_all = pd.read_csv(out_dir / "rolling_samples_metadata.csv")
    plot_curves(preds, ok_metrics, derivation, samples_all, cfg)


def write_readme(cfg: dict) -> None:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    text = f"""# Single-visit two-year first-onset SIC prediction

This independent experiment constructs rolling single-visit landmark samples for
two-year first-onset SIC prediction. It does not modify the original DTEK-GT
training scripts or previous results.

## Field mapping

- Patient ID: `{cfg['patient_id_column']}`
- Visit date: `{cfg['visit_date_column']}`
- SIC label: `{cfg['label_column']}`
- Clinical variables: `{', '.join(cfg['clinical_variables'])}`
- Raw missingness mask: one mask per clinical variable, encoded as `1=missing`
  and `0=observed`, computed directly from the source table before imputation.
- Visit-level image descriptor: `{cfg['feature_path']}`
- Patient-level fold manifest: `{cfg['split_manifest_path']}`

The current source spreadsheet has no missing values in the seven clinical
variables, so missingness-only baselines may be uninformative for this dataset.

## Task definition

For a SIC-negative anchor visit, the model sees only the current visit image
descriptor, current clinical variables, demographics, and current missingness
mask. The label is whether the patient's first SIC-positive visit occurs in
`(anchor_date, anchor_date + {cfg['horizon_days']} days]`. A negative label is
assigned only when at least {cfg['horizon_days']} days of follow-up are observed
without SIC onset. Otherwise the anchor is marked censored and excluded from the
binary experiment.

Primary first-visit evaluation strictly uses the patient's true first visit.
Later visits are never substituted for censored or ineligible first visits.

## Split and leakage control

The existing patient-level five-fold split is reused. Because the original split
manifest contains train and held-out patients but no explicit validation split,
this experiment creates an inner validation split from training patients only,
stratified by available first-visit or rolling labels. Samples are assigned to
folds by patient ID after the patient-level split is fixed.

All preprocessing statistics, including clinical imputation/standardization and
image descriptor standardization, are fitted on each fold's training samples
only and then applied to validation/test samples.

## Models

The experiment implements:

1. Age + sex logistic regression
2. TgAb + TPOAb + TSH logistic regression
3. All clinical variables logistic regression
4. All clinical variables + missingness mask logistic regression
5. Clinical variables HistGradientBoosting
6. Missingness-only logistic regression
7. Image-only classifier
8. Tabular-only MLP
9. Single-visit image-tabular fusion model
10. Single-node DTEK-GT

`xgboost` is not required. If it is unavailable, the tree baseline is explicitly
reported as `Clinical variables HistGradientBoosting`.

## Training/evaluation settings

- A: First-visit training / First-visit evaluation
- B: Rolling-landmark training / First-visit evaluation
- C: Rolling-landmark training / Rolling evaluation

Setting B is the primary setting. It is run with and without patient-balanced
weighting, where each rolling sample from patient `i` receives base weight
`1 / n_i`, then class balancing is applied within the training fold.

## Commands

Build samples and leakage audit only:

```bash
python3 experiments/single_visit_two_year_onset/scripts/run_experiment.py --derive-only
```

Smoke test:

```bash
python3 experiments/single_visit_two_year_onset/scripts/run_experiment.py --smoke
```

Full run:

```bash
python3 experiments/single_visit_two_year_onset/scripts/run_experiment.py
```

Regenerate aggregate tables, bootstrap confidence intervals, and figures from
existing prediction CSVs without retraining:

```bash
python3 experiments/single_visit_two_year_onset/scripts/run_experiment.py --postprocess-only
```

Run selected models/settings:

```bash
python3 experiments/single_visit_two_year_onset/scripts/run_experiment.py \\
  --settings B \\
  --models "Single-node DTEK-GT" "Single-visit image-tabular fusion model"
```

## Main outputs

Results are written to `{cfg['output_dir']}` and figures to `{cfg['figure_dir']}`.
Key files include `sample_derivation.csv`, `rolling_samples_metadata.csv`,
`first_visit_samples_metadata.csv`, `fold_metrics.csv`,
`model_comparison.csv`, `training_strategy_comparison.csv`,
`ablation_results.csv`, `calibration_metrics.csv`,
`oof_first_visit_predictions.csv`, `oof_rolling_predictions.csv`,
`patient_weighting_comparison.csv`, and `leakage_audit.json`.
"""
    readme_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    write_readme(cfg)

    visit_df, image_features = load_visit_table(cfg)
    samples = derive_single_visit_samples(visit_df, cfg)
    folds = load_existing_folds(cfg)
    samples_by_fold, derivation = add_validation_split(samples, folds, cfg)
    audit = leakage_audit(samples_by_fold, derivation, cfg)
    save_derive_outputs(samples_by_fold, derivation, audit, cfg)

    print(f"Aligned visits: {len(visit_df)}")
    print(f"Candidate anchor visits: {len(samples)}")
    print(f"Eligible rolling rows across fold assignments: {(samples_by_fold['censored'] == 0).sum()}")
    print(f"Leakage audit passed: {audit['all_passed']}")
    print(f"Outputs: {cfg['output_dir']}")

    if args.derive_only:
        return
    if args.postprocess_only:
        run_postprocess(cfg)
        return
    run_training(samples_by_fold, image_features, cfg, args)


if __name__ == "__main__":
    main()
