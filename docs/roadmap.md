# Upgrade roadmap

Each phase has an exit gate. A later phase must not compensate for a failed earlier data contract.

| Phase | Deliverable | Exit gate |
|---|---|---|
| 0–1 | Archive audit, frozen group split, train-only scaler, window manifests | **Complete:** leakage tests pass |
| 2A | Handcrafted features + GMM / Isolation Forest | **Complete on Validation:** Test untouched |
| 2B | XGBoost supervised baseline | **Complete on Validation:** Train-only weights and early stopping |
| 3 | 1D-CNN raw-sequence baseline | **Complete on Validation:** compact CNN frozen; cross-activity overfit diagnosed |
| 4 | 1D-CNN + 1–2 layer Transformer encoder | **Complete on Validation:** two layers frozen; parameter/latency ablation recorded |
| 5 | Contrastive pretraining + fine-tuning | **Complete on Validation:** Train-only SSL, matched ablation, three-seed and non-overlap audits |
| 6 | Final evaluation and error analysis | **Complete:** single frozen Test evaluation, per-motor/run/event metrics, run bootstrap and errors |
| 7 | Resume bullet | **Complete:** uses only real Test metrics and explicitly blocks unsupported SSL claims |

## Phase 2 baseline contract

- GMM and Isolation Forest train on Train only and output continuous anomaly scores.
- XGBoost receives handcrafted Train features and `scale_pos_weight` computed from Train labels only.
- Hyperparameters and support-aware per-motor thresholds are selected on Validation.
- The historical GMM F1 values (M2 ≈ 0.532, M4 ≈ 0.470) remain contextual references only because they were produced on reused data.

## Three required ablations

1. Handcrafted features vs raw sequence.
2. CNN vs CNN + Transformer.
3. Without SSL vs with SSL.

Raw vs Z-score-cleaned is optional but strongly motivated. The main pipeline never replaces Z-score outliers.

## Final reporting contract

Produce one model-comparison table and one per-motor table from the same frozen Test set. Primary metrics are PR-AUC, Precision, Recall, and F1. Include confusion matrices and event-level Recall/F1 as diagnostics. Do not use point adjustment.
