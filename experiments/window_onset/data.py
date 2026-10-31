"""Historical-prefix landmarks for recorded first onset, not future visit status."""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]


def load_visits(cfg):
    # Reuse the checked Excel/cache alignment, never the cache's tabular values.
    path = Path(__file__).resolve().parent / "single_visit_helpers.py"
    spec = importlib.util.spec_from_file_location("single_visit_data_helpers", path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    config = dict(cfg, feature_path=str(ROOT / cfg["feature_path"]))
    visits, images = helper.load_visit_table(config)
    assert visits.sic_label.isin([0, 1]).all()
    assert not visits.duplicated(["patient_id", "visit_date"]).any(), "Ambiguous same-day visits"
    assert np.isfinite(images).all()
    return visits.reset_index(drop=True), images


def derive(visits, horizon=730, min_history=2):
    records = []
    for pid, group in visits.groupby("patient_id", sort=True):
        group = group.sort_values("visit_date", kind="stable")
        onset = group.loc[group.sic_label == 1, "visit_date"].min()
        last = group.loc[group.sic_label.isin([0, 1]), "visit_date"].max()
        for j, (_, visit) in enumerate(group.iterrows()):
            if j + 1 < min_history or visit.sic_label != 0:
                continue
            anchor = visit.visit_date
            if pd.notna(onset) and anchor >= onset:
                continue
            delay = (onset - anchor).days if pd.notna(onset) else np.nan
            followup = (last - anchor).days
            positive = pd.notna(delay) and 0 < delay <= horizon
            censored = not positive and followup < horizon
            history = group.iloc[:j + 1]
            assert (history.sic_label == 0).all()
            records.append({
                "sample_id": f"{pid}__{visit.visit_id}", "patient_id": str(pid),
                "anchor_visit_id": visit.visit_id, "anchor_date": anchor,
                "history_visits": j + 1, "history_span_days": (anchor - group.iloc[0].visit_date).days,
                "history_rows": ";".join(map(str, history.index)),
                "label": np.nan if censored else int(positive), "censored": int(censored),
                "onset_date": onset, "days_to_onset": delay,
                "last_observed_date": last, "followup_days": followup,
                "first_multivisit_anchor": int(j + 1 == min_history),
            })
    return pd.DataFrame(records)


def load_splits(cfg, visits):
    import json
    old = pd.read_csv(ROOT / cfg["existing_assignments"], dtype={"patient_id": str})
    assignments = old[["fold", "patient_id", "split"]].drop_duplicates()
    assert not assignments.duplicated(["fold", "patient_id"]).any()
    outer = json.loads((ROOT / cfg["outer_manifest"]).read_text())
    population = set(visits.patient_id)
    for fold, g in assignments.groupby("fold"):
        assert set(g.patient_id) == population
        pools = {s: set(g.loc[g.split == s, "patient_id"]) for s in ["train", "val", "test"]}
        assert not (pools["train"] & pools["val"] or pools["train"] & pools["test"] or pools["val"] & pools["test"])
        source = outer[f"fold_{fold}"]
        assert pools["test"] == set(map(str, source["heldout_patient_ids"]))
        assert pools["train"] | pools["val"] == set(map(str, source["train_patient_ids"]))
        assert abs(len(pools["val"]) / (len(pools["train"]) + len(pools["val"])) - .2) < .001
    heldout = assignments[assignments.split == "test"]
    assert not heldout.patient_id.duplicated().any()
    assert set(heldout.patient_id) == population
    return assignments.sort_values(["fold", "split", "patient_id"])


def history_rows(metadata):
    return sorted({int(i) for value in metadata.history_rows for i in value.split(";")})


class Preprocessor:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, visits, train):
        self.fit_rows = history_rows(train)
        raw = visits.loc[self.fit_rows, self.columns].apply(pd.to_numeric, errors="raise")
        # TgAb and TPOAb: clip at the training 99th percentile, then log(1 + x)
        self.clip = {c: raw[c].quantile(0.99) for c in ('TgAb', 'TPOAb') if c in self.columns}
        raw = self._antibody(raw)
        self.median = raw.median().fillna(0)
        filled = raw.fillna(self.median)
        self.mean = filled.mean()
        self.std = filled.std(ddof=0).replace(0, 1).fillna(1)
        return self

    def transform(self, visits):
        raw = visits[self.columns].apply(pd.to_numeric, errors="raise")
        raw = self._antibody(raw)
        masks = raw.isna().to_numpy(np.float32)
        values = ((raw.fillna(self.median) - self.mean) / self.std).to_numpy(np.float32)
        assert np.isfinite(values).all()
        return np.concatenate([values, masks], axis=1)

    def _antibody(self, raw):
        raw = raw.copy()
        for c, upper in getattr(self, 'clip', {}).items():
            raw[c] = np.log1p(raw[c].clip(upper=upper))
        return raw

    def state(self):
        return {"columns": self.columns, "median": self.median.to_dict(),
                "mean": self.mean.to_dict(), "std": self.std.to_dict(), "fit_rows": self.fit_rows}


def make_graphs(metadata, visits, images, tabular, tau=180):
    graphs = []
    for row in metadata.itertuples():
        ids = [int(i) for i in row.history_rows.split(";")]
        history = visits.loc[ids]
        assert len(ids) >= 2 and history.visit_date.max() == row.anchor_date
        assert history.sic_label.eq(0).all()
        image_rows = history.feature_row.to_numpy(int)
        x = torch.tensor(np.concatenate([images[image_rows], tabular[ids]], axis=1))
        delta = np.diff(history.visit_date.to_numpy().astype("datetime64[D]")).astype(int)
        assert (delta > 0).all()
        edges, attributes = [], []
        for i, days in enumerate(delta):
            edges.extend([[i, i + 1], [i + 1, i]])
            attributes.extend([[days, np.exp(-days / tau)]] * 2)
        g = Data(x=x, edge_index=torch.tensor(edges).T.contiguous(),
                 edge_attr=torch.tensor(attributes, dtype=torch.float32),
                 y=torch.tensor(int(row.label), dtype=torch.long))
        g.sample_weight = torch.tensor(float(row.sample_weight), dtype=torch.float32)
        g.missing_mask = x[:, -len(tabular[0]) // 2:]
        graphs.append(g)
    return graphs


def audit(samples, assignments, visits, cfg):
    valid = samples[samples.censored == 0]
    pos, neg = valid[valid.label == 1], valid[valid.label == 0]
    checks = {
        "minimum_two_historical_visits": bool(samples.history_visits.ge(2).all()),
        "all_positive_first_onsets_in_window": bool(pos.days_to_onset.between(1, cfg["horizon_days"]).all()),
        "all_negative_followup_covers_window": bool(neg.followup_days.ge(cfg["horizon_days"]).all()),
        "negative_onsets_absent_or_after_window": bool((neg.days_to_onset.isna() | neg.days_to_onset.gt(cfg["horizon_days"])).all()),
        "censored_labels_unknown": bool(samples.loc[samples.censored == 1, "label"].isna().all()),
        "anchors_before_first_recorded_onset": bool((samples.days_to_onset.isna() | samples.days_to_onset.gt(0)).all()),
        "unique_sample_ids": bool(not samples.sample_id.duplicated().any()),
        "one_earliest_multivisit_anchor_per_patient": bool(not samples.loc[samples.first_multivisit_anchor == 1, "patient_id"].duplicated().any()),
        "patient_splits_match_saved_assignments": True,
        "outer_test_patients_held_out_once": bool(not assignments[assignments.split == "test"].patient_id.duplicated().any()),
        "repeat_derivation_identical": samples.equals(derive(visits, cfg["horizon_days"], cfg["min_history_visits"])),
        "feature_whitelist": cfg["clinical_variables"] == ["sex", "age", "FT3", "FT4", "TSH", "TgAb", "TPOAb"],
    }
    assert all(checks.values()), checks
    return {"checks": checks, "all_passed": all(checks.values()),
            "endpoint": f"first recorded HT-positive examination within (anchor, anchor+{cfg['horizon_days']} days]",
            "limitation": "Observed examination dates do not locate biological onset between visits; censoring exclusions can select for longer follow-up."}
