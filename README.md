# DESL: dual-temporal edge-kernel sequence-graph learning for early Hashimoto's thyroiditis prediction

This repository contains the code for DESL, which predicts future Hashimoto's thyroiditis (HT) from
longitudinal follow-up visits with irregular intervals and incomplete measurements. Each visit
combines thyroid ultrasound images with clinical indicators. Visits are linked by temporal edges and
the sequence is summarised by a Transformer into a patient-level risk score.

The private HT cohort is not included. Data can be requested from the corresponding authors
(see the Data availability statement of the paper). A synthetic dataset generator is provided so
that the pipeline can be run without the private data.

## Installation

```bash
conda create -n desl python=3.9
conda activate desl
pip install -r requirements.txt
```

`torch_geometric` should match your PyTorch and CUDA versions; see the PyG installation guide.

## Quick check with synthetic data

```bash
bash tools/smoke_test.sh
```

This creates `data/synthetic_features.npz` and runs a short two-fold training run.

## Data format

`extract_ultrasound_features.py` reads the visit table and the ultrasound images and writes one
`.npz` file with one row per visit:

| key | shape | content |
|---|---|---|
| `features` | (visits, 2048) | mean ResNet-50 (ImageNet, frozen) descriptor of up to five randomly selected images per visit (fixed seed) |
| `tabular_feats` | (visits, 14) | sex, age, FT3, FT4, TSH, TgAb and TPOAb after preprocessing, followed by 7 missingness masks (1 = missing) |
| `labels` | (visits,) | visit-level HT label (0/1) |
| `patients` | (visits,) | patient identifier |
| `dates` | (visits,) | visit date (YYYY-MM-DD) |

Expected inputs for feature extraction:

* `--label_path`: a table with columns `patient_Id`, `time`, `sex`, `age`, `FT3`, `FT4`, `TSH`, `TgAb`, `TPOAb` and `SIC` (HT label).
* `--root_dir`: one folder per patient, with one subfolder per visit date (`YYYY-MM-DD`) that holds the ultrasound images.

TgAb and TPOAb are clipped at the 99th percentile and log-transformed, missing values are filled with
medians, and all indicators are standardised (`tabular_extractor.py`).

## Usage

```bash
# 1. Visit-level features
python extract_ultrasound_features.py --root_dir /path/to/images --label_path /path/to/visits.xlsx \
    --feature_save_path data/features.npz

# 2. Next-visit prediction, patient-level five-fold cross-validation.
#    In each fold the held-out fold is used only for testing; the other four folds are split 8:2 by
#    patient into training and validation sets, and validation selects the checkpoint and threshold.
python run_5fold_cv.py --feature_path data/features.npz --output_dir results/next_visit_5fold
```

Paths default to `data/` and `results/` inside the repository and can be changed with the
`HT_DATA_DIR` and `HT_RESULTS_DIR` environment variables or the command-line options.

## Window-based onset tasks

`experiments/window_onset/` predicts the first recorded HT onset within a window after each negative
anchor visit, using all visits up to the anchor (at least two visits). Negative outcomes require
follow-up through the whole window; anchors with shorter follow-up and no onset are censored.

```bash
python experiments/window_onset/run.py --config experiments/window_onset/config_2year.json   # 2-year window
python experiments/window_onset/run.py --config experiments/window_onset/config_3year.json   # 3-year window
```

These tasks use the same seven indicators and masks as above and reuse the patient splits of the main
experiment; the split files and input paths are set in each config file.

## Repository layout

| file | role |
|---|---|
| `graphtransformer.py` | DESL model: visit encoding with the modality gate, temporal graph attention, Transformer aggregation |
| `graph_builder.py` | builds the sequence-graph of a patient with dual-temporal edge attributes |
| `tabular_extractor.py` | clinical indicator preprocessing and missingness masks |
| `losses.py` | weighted cross-entropy and alternative losses |
| `train.py` | data loading, training loop, evaluation and threshold selection |
| `run_5fold_cv.py` | five-fold cross-validation for next-visit prediction |
| `extract_ultrasound_features.py` | visit-level ultrasound and clinical features |
| `experiments/window_onset/` | 2-year and 3-year onset prediction |
| `tools/` | synthetic data and smoke test |

## License

MIT License. See `LICENSE`.
