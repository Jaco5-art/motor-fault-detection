# Reproducibility and Test integrity

## Split contract

One timestamped run directory is the unit of independence. The split is fixed in `configs/split_group_future_holdout_v1.yaml`; a run ID may appear in exactly one of Train, Validation, or Test. Windows are generated only after the group split and never cross run boundaries.

The selected input contract is `[batch, 18, 200] → [batch, 6]` with stride 10 and the `any fault point` window-label rule. Window length 200 was selected against 400 on Validation; 800 was not evaluated.

## Fit and selection ownership

| Operation | Allowed split |
|---|---|
| Raw-channel scaler fit | Train only |
| Handcrafted-feature scaler fit | Train only |
| Class weights | Train only |
| GMM/IF/XGBoost/CNN/Transformer/SSL fitting | Train only |
| Architecture and hyperparameter selection | Validation |
| Early stopping | Validation |
| Threshold selection | Validation |
| Final metrics | Test once |

M6 has Validation positives and uses an F1-optimal Validation threshold. M1–M5 lack Validation positives and use their Validation normal-score 95th percentile. The limitation is reported rather than repaired with Test labels.

## Frozen final evaluation

`configs/phase6_final_evaluation.yaml` records SHA-256 values for:

- the dataset archive;
- Train, Validation, and Test manifests;
- raw and handcrafted-feature scalers;
- all six selected checkpoints and the XGBoost calibrator.

`scripts/run_phase6_final_evaluation.py --preflight-only` verifies these inputs without parsing Test labels or generating Test scores. The actual evaluator disables training and threshold selection, persists a status file before scoring, and refuses to overwrite a completed evaluation.

The first execution attempt stopped before writing metrics because XGBoost was absent from the runtime. The dependency-only recovery retained identical hashes, checkpoints, thresholds, and evaluation code. This technical attempt history is preserved in the private experiment artifact.

## Metrics

Primary metrics flatten the six motor decisions over all Test windows:

- PR-AUC;
- Precision;
- Recall;
- F1;
- ROC-AUC as a secondary diagnostic;
- TP, TN, FP, and FN.

The project also reports per-motor and per-run metrics, a greedy non-overlapping-window diagnostic, and 2,000-run bootstrap intervals.

Event metrics use contiguous raw fault-point episodes within the evaluated coverage. Positive alarm windows are merged into intervals and matched to true events with maximum-cardinality one-to-one interval overlap. Window predictions are never extended, relabeled, or point-adjusted.

## Reproduction commands

```bash
python -m pip install -e ".[deep,viz]"
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

Full training requires the Kaggle archive and runs the scripts in numeric phase order. Training is CPU-compatible but SSL is substantially faster on a supported GPU.

To regenerate the public figures from the already exported aggregate tables:

```bash
python scripts/generate_readme_assets.py
```

Do not delete a completed final-evaluation status merely to rerun Test. Any future model iteration requires a newly declared development protocol and genuinely new held-out labeled runs.
