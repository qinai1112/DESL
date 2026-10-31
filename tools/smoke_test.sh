#!/usr/bin/env bash
# Quick end-to-end check on synthetic data (2 folds, 2 epochs). Takes about one minute on a CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/make_synthetic_data.py --out data/synthetic_features.npz
python run_5fold_cv.py --feature_path data/synthetic_features.npz --output_dir results/smoke_test \
    --folds 2 --epochs 2 --patience 1
echo "Smoke test finished. See results/smoke_test/"
