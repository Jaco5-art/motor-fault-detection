# Model card: frozen GMM inference bundle

## Intended use

The bundled model demonstrates window-level anomaly scoring for six motors in the Kaggle Robot Predictive Maintenance data schema. It accepts one archive run, calculates 144 handcrafted features per 200-point window, and returns six calibrated anomaly scores and Validation-frozen alarm decisions.

It is suitable for portfolio demonstrations, reproducibility checks, and offline experiments. It is not a safety-certified predictive-maintenance system and must not control machinery or replace engineering inspection.

## Inputs and outputs

- Input channels: position, temperature, and voltage for Motors 1–6.
- Window length: 200 points.
- Default stride: 10 points.
- Features: mean, standard deviation, range, endpoint difference, mean absolute difference, energy, crest factor, and smoothing index for each channel.
- Outputs: six empirical-percentile anomaly scores, six thresholds, and six binary window alarms.

## Training and thresholds

The GMMs and empirical score calibrators were fitted on Train only. Model components and all six thresholds were chosen or calculated from Validation. The bundled artifact is not refitted during inference.

## Frozen Test evidence

- PR-AUC: 0.294
- Precision: 0.342
- Recall: 0.488
- F1: 0.402
- Event-level F1: 0.429

These results apply only to the declared Test split. M1–M5 each have positive Test windows from one independent run, and run-bootstrap uncertainty is wide.

## Known failure modes

- GMM completely misses M1 and M3 at their frozen thresholds on the Test split.
- Distribution shift between activities can create false alarms.
- Overlapping windows create correlated alarms.
- The model does not estimate remaining useful life or fault severity.
- Scores should not be interpreted as calibrated fault probabilities.

## Security

The model is serialized with `joblib`, which uses pickle-compatible loading. Only load the repository-owned bundle. Never pass an untrusted checkpoint to `joblib.load`.
