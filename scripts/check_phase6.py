#!/usr/bin/env python3
"""Verify persisted Phase 6 results without evaluating the dataset again."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from motor_fault.evaluation.metrics import evaluate_multimotor


EXPECTED_MODELS = [
    "GMM",
    "Isolation Forest",
    "XGBoost",
    "1D-CNN",
    "CNN + Transformer",
    "CNN + Transformer + SSL",
]
SCORE_KEYS = {
    "GMM": "gmm",
    "Isolation Forest": "isolation_forest",
    "XGBoost": "xgboost",
    "1D-CNN": "1d_cnn",
    "CNN + Transformer": "cnn_plus_transformer",
    "CNN + Transformer + SSL": "cnn_plus_transformer_plus_ssl",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-dir", type=Path, default=Path("artifacts/final_evaluation")
    )
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("artifacts/experiments")
    )
    args = parser.parse_args()
    root = args.evaluation_dir
    status = json.loads((root / "final_evaluation_status.json").read_text())
    metrics_payload = json.loads((root / "test_metrics.json").read_text())
    frozen_checks = json.loads((root / "frozen_input_checks.json").read_text())
    comparison = pd.read_csv(root / "model_comparison.csv")
    per_motor = pd.read_csv(root / "per_motor_metrics.csv")
    events = pd.read_csv(root / "event_metrics.csv")
    resume = json.loads((root / "resume_evidence.json").read_text())
    scores = np.load(root / "test_scores_and_labels.npz")
    labels = scores["labels"].astype(np.int8)

    traditional = {
        "GMM": joblib.load(args.experiment_dir / "gmm_seed42/model.joblib"),
        "Isolation Forest": joblib.load(
            args.experiment_dir / "isolation_forest_seed42/model.joblib"
        ),
        "XGBoost": joblib.load(
            args.experiment_dir / "xgboost_seed42/score_calibrator.joblib"
        ),
    }
    deep_paths = {
        "1D-CNN": args.experiment_dir / "cnn_seed42/model.pt",
        "CNN + Transformer": args.experiment_dir / "cnn_transformer_seed42/model.pt",
        "CNN + Transformer + SSL": args.experiment_dir
        / "cnn_transformer_ssl_seed42/model.pt",
    }
    checkpoint_thresholds = {
        name: np.asarray(payload["thresholds"], dtype=np.float64)
        for name, payload in traditional.items()
    }
    for name, path in deep_paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_thresholds[name] = np.asarray(
            checkpoint["thresholds"], dtype=np.float64
        )

    recomputed_passed = True
    threshold_passed = True
    for name in EXPECTED_MODELS:
        persisted_thresholds = np.asarray(
            metrics_payload["model_metrics"][name]["thresholds"], dtype=np.float64
        )
        threshold_passed &= np.array_equal(
            persisted_thresholds, checkpoint_thresholds[name]
        )
        recomputed = evaluate_multimotor(
            labels, scores[SCORE_KEYS[name]], persisted_thresholds
        )["overall_micro"]
        observed = comparison.set_index("model").loc[name]
        recomputed_passed &= all(
            np.isclose(float(recomputed[key]), float(observed[key]))
            for key in ("pr_auc", "precision", "recall", "f1", "roc_auc")
        )

    all_hash_checks = [frozen_checks["archive"]["passed"]]
    all_hash_checks.extend(
        item["passed"] for item in frozen_checks["inputs"].values()
    )
    for model_checks in frozen_checks["models"].values():
        all_hash_checks.extend(item["passed"] for item in model_checks.values())
    checks = {
        "status_completed": status.get("status") == "completed",
        "single_successful_evaluation_after_recorded_technical_failure": (
            status.get("test_evaluation_attempt") == 2
            and status.get("previous_technical_failure", {}).get("metrics_written")
            is False
        ),
        "no_training": status.get("training_performed") is False,
        "no_test_threshold_selection": status.get("threshold_selection_performed")
        is False,
        "point_adjustment_disabled": metrics_payload["protocol"]["point_adjustment"]
        is False,
        "all_frozen_hashes_passed": all(all_hash_checks),
        "six_expected_models": comparison["model"].tolist() == EXPECTED_MODELS,
        "all_motors_reported": len(per_motor) == 36
        and set(per_motor["motor"]) == {f"M{i}" for i in range(1, 7)},
        "event_rows_complete": len(events) == 42
        and set(events["motor"]) == {"overall", *[f"M{i}" for i in range(1, 7)]},
        "score_label_contract": labels.shape == (531, 6)
        and all(scores[key].shape == labels.shape for key in SCORE_KEYS.values()),
        "thresholds_identical_to_validation_checkpoints": bool(threshold_passed),
        "persisted_metrics_recompute_exactly": bool(recomputed_passed),
        "resume_evidence_uses_observed_best_model": resume["best_model"] == "GMM"
        and np.isclose(resume["best_model_f1"], comparison.iloc[0]["f1"]),
        "unsupported_ssl_claim_blocked": resume["ssl_improves_transformer_test_f1"]
        is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    (root / "phase6_check.json").write_text(
        json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("All Phase 6 frozen-result checks passed")


if __name__ == "__main__":
    main()
