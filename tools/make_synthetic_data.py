"""Create a small synthetic dataset in the same format as the real visit-level feature file.

The file contains random numbers only. It lets users check that the pipeline runs end to end
without access to the private HT cohort.

Output (.npz) keys, one row per visit:
  features      float32 (n_visits, 2048)  visit-level ultrasound descriptor (mean of ResNet-50 features)
  tabular_feats float32 (n_visits, 14)    7 standardised indicators (sex, age, FT3, FT4, TSH, TgAb, TPOAb)
                                          followed by 7 missingness masks (1 = missing, 0 = measured)
  labels        int64   (n_visits,)       visit-level HT label (0/1)
  patients      str     (n_visits,)       patient identifier
  dates         str     (n_visits,)       visit date, YYYY-MM-DD
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_features.npz")
    parser.add_argument("--patients", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    feats, tabs, labels, pids, dates = [], [], [], [], []
    for p in range(args.patients):
        n_visits = int(rng.integers(3, 7))
        future_onset = rng.random() < 0.45
        day = date(2018, 1, 1) + timedelta(days=int(rng.integers(0, 700)))
        for j in range(n_visits):
            positive = future_onset and j == n_visits - 1
            shift = 0.4 * (j + 1) / n_visits if future_onset else 0.0
            feats.append(rng.normal(shift, 1.0, 2048).astype(np.float32))
            values = rng.normal(shift, 1.0, 7)
            missing = np.zeros(7)
            missing[2:5] = rng.random(3) < 0.01      # FT3, FT4, TSH almost always measured
            missing[5:7] = rng.random(2) < 0.75      # TgAb, TPOAb missing at most visits
            values[missing == 1] = 0.0               # median-imputed and standardised value
            tabs.append(np.concatenate([values, missing]).astype(np.float32))
            labels.append(int(positive))
            pids.append(f"P{p:04d}")
            dates.append(day.isoformat())
            day += timedelta(days=int(rng.integers(150, 500)))  # irregular visit intervals

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=np.stack(feats), tabular_feats=np.stack(tabs), labels=np.array(labels),
             patients=np.array(pids), dates=np.array(dates))
    print(f"Saved {len(labels)} visits from {args.patients} patients to {out}")


if __name__ == "__main__":
    main()
