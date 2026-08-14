# Frozen independent Test results

## Unified comparison

| Model | PR-AUC | Precision | Recall | F1 | Event F1 |
|---|---:|---:|---:|---:|---:|
| GMM | **0.294** | **0.342** | 0.488 | **0.402** | **0.429** |
| Isolation Forest | 0.163 | 0.285 | 0.460 | 0.352 | 0.214 |
| XGBoost | 0.169 | 0.170 | 0.712 | 0.275 | 0.353 |
| 1D-CNN | 0.096 | 0.118 | **0.818** | 0.206 | 0.321 |
| CNN + Transformer | 0.087 | 0.119 | 0.404 | 0.184 | 0.364 |
| CNN + Transformer + SSL | 0.088 | 0.113 | 0.439 | 0.180 | 0.279 |

All models use the same group-held-out Test windows and Validation-frozen thresholds. No training, Test threshold selection, or point adjustment was performed.

## Support

| Motor | Positive windows | Fault-window ratio | Positive runs |
|---|---:|---:|---:|
| M1 | 26 | 4.90% | 1 |
| M2 | 28 | 5.27% | 1 |
| M3 | 28 | 5.27% | 1 |
| M4 | 28 | 5.27% | 1 |
| M5 | 27 | 5.08% | 1 |
| M6 | 148 | 27.87% | 4 |

The complete Test contains 531 windows, 3,186 motor-window decisions, and 285 positive decisions. Its positive prevalence is 8.95%; GMM PR-AUC is 3.29 times that prevalence.

## Event definition

True events are contiguous raw fault-point intervals inside window-covered portions of a run. Predicted positive windows are merged into alarm intervals. Maximum-cardinality one-to-one interval-overlap matching determines matched events, false alarms, and misses. This is aggregation only: individual window predictions are not modified.

## Ablations

- **Handcrafted vs raw sequence:** the pragmatic GMM-to-CNN comparison changes both representation and model. Relative to GMM, 1D-CNN increases Recall by 0.330 but reduces F1 by 0.196.
- **CNN vs CNN-Transformer:** adding the Transformer reduces PR-AUC by 0.009 and F1 by 0.022.
- **Without vs with SSL:** SSL changes PR-AUC by less than 0.001, increases Recall by 0.035, and reduces F1 by 0.004.

## Error analysis

- GMM has the best overall balance but misses all M1 and M3 positive windows at the frozen thresholds.
- Isolation Forest detects only M6 faults under the frozen threshold policy.
- XGBoost reaches Recall 1.0 for M3, M5, and M6 but creates 988 false positives overall.
- 1D-CNN reaches the highest overall Recall, 0.818, but creates 1,744 false positives and has Precision 0.118.
- SSL improves M3 ranking and M2/M4 Recall, but M6 Recall falls from 0.264 without SSL to 0.182 with SSL.
- The greedy non-overlapping-window diagnostic preserves the overall ranking: GMM remains best with PR-AUC 0.356 and F1 0.400.

## Uncertainty and conclusion

Run-level bootstrap intervals are wide: the GMM 95% intervals are approximately 0.120–0.546 for PR-AUC and 0.168–0.639 for F1. The defensible statement is therefore limited to this split.

The final result does not support a Transformer or SSL improvement claim. The engineering contribution is the leakage-free evaluation system and the evidence that model complexity did not overcome cross-run distribution shift in this small labeled dataset.
