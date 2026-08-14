#!/usr/bin/env python3
"""Run the single frozen held-out Test evaluation for all six selected models."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import platform
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from motor_fault.config import read_yaml  # noqa: E402
from motor_fault.data.archive import KaggleRobotArchive  # noqa: E402
from motor_fault.data.handcrafted import (  # noqa: E402
    FeatureStandardizer,
    extract_handcrafted_features,
    labels_from_manifest,
)
from motor_fault.data.scaler import ChannelStandardizer  # noqa: E402
from motor_fault.evaluation.events import (  # noqa: E402
    event_counts_for_motor,
    greedy_nonoverlap_indices,
)
from motor_fault.evaluation.metrics import evaluate_multimotor, json_safe  # noqa: E402
from motor_fault.models.cnn import build_cnn  # noqa: E402
from motor_fault.models.traditional import _gmm_scores, _iforest_scores  # noqa: E402
from motor_fault.models.transformer import build_cnn_transformer  # noqa: E402
from motor_fault.training.cnn_trainer import materialize_windows, predict_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/phase6_final_evaluation.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts/final_evaluation"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify frozen hashes/checkpoints without parsing Test labels or scoring Test.",
    )
    parser.add_argument(
        "--resume-technical-failure",
        action="store_true",
        help="Resume only from an explicitly recorded technical failure; never retune inputs.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_hash(spec: dict[str, str], label: str) -> dict[str, str | bool]:
    path = project_path(spec["path"])
    observed = file_sha256(path)
    expected = str(spec["sha256"])
    if observed != expected:
        raise ValueError(f"Frozen hash mismatch for {label}: {observed} != {expected}")
    return {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": observed, "passed": True}


def preflight(config: dict, archive_path: Path) -> dict[str, Any]:
    if config["experiment"].get("test_access") != "single_frozen_evaluation":
        raise ValueError("Phase 6 must use the single frozen Test protocol")
    if config["experiment"].get("allow_training") is not False:
        raise ValueError("Training must be disabled during Test evaluation")
    if config["experiment"].get("allow_threshold_selection") is not False:
        raise ValueError("Threshold selection must be disabled during Test evaluation")
    if config["protocol"].get("point_adjustment") is not False:
        raise ValueError("Point adjustment must remain disabled")

    archive = KaggleRobotArchive(archive_path)
    expected_archive = str(config["data"]["archive_sha256"])
    if archive.archive_sha256 != expected_archive:
        raise ValueError("Dataset archive checksum differs from the frozen protocol")
    checks: dict[str, Any] = {
        "archive": {"sha256": archive.archive_sha256, "passed": True},
        "dependencies": {
            "torch": {"version": torch.__version__, "passed": True},
        },
        "inputs": {},
        "models": {},
    }
    try:
        import xgboost

        checks["dependencies"]["xgboost"] = {
            "version": xgboost.__version__,
            "passed": True,
        }
    except ImportError as exc:
        raise ImportError("Frozen Phase 6 environment is missing XGBoost") from exc
    for key in (
        "test_manifest",
        "train_manifest",
        "validation_manifest",
        "raw_scaler",
        "feature_scaler",
    ):
        checks["inputs"][key] = verify_hash(config["data"][key], key)
    for model_spec in config["models"]:
        model_checks = {
            "checkpoint": verify_hash(model_spec["checkpoint"], model_spec["name"])
        }
        if "calibrator" in model_spec:
            model_checks["calibrator"] = verify_hash(
                model_spec["calibrator"], f"{model_spec['name']} calibrator"
            )
        checks["models"][model_spec["name"]] = model_checks
    return checks


def build_stacked_motor_inputs(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    motor_ids = np.tile(np.arange(6, dtype=np.int8), len(x))
    repeated = np.repeat(x, 6, axis=0)
    one_hot = np.eye(6, dtype=np.float32)[motor_ids]
    return np.concatenate([repeated, one_hot], axis=1)


def score_traditional(
    model_spec: dict,
    features: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    family = str(model_spec["family"])
    checkpoint_path = project_path(model_spec["checkpoint"]["path"])
    if family in {"gmm", "isolation_forest"}:
        payload = joblib.load(checkpoint_path)
        if tuple(payload["feature_names"]) != tuple(feature_names):
            raise ValueError(f"Feature-name mismatch for {model_spec['name']}")
        raw_scores = (
            _gmm_scores(payload["models"], features, tuple(feature_names))
            if family == "gmm"
            else _iforest_scores(payload["models"], features, tuple(feature_names))
        )
        return (
            payload["calibrator"].transform(raw_scores),
            np.asarray(payload["thresholds"], dtype=np.float64),
            dict(payload["threshold_policies"]),
        )
    if family == "xgboost":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(checkpoint_path)
        stacked = build_stacked_motor_inputs(features)
        metrics = json.loads(
            project_path(model_spec["validation_metrics"]).read_text(encoding="utf-8")
        )
        end_iteration = int(metrics["best_hyperparameters"]["best_iteration"]) + 1
        raw_scores = booster.predict(
            xgb.DMatrix(stacked), iteration_range=(0, end_iteration)
        ).reshape(len(features), 6)
        payload = joblib.load(project_path(model_spec["calibrator"]["path"]))
        return (
            payload["calibrator"].transform(raw_scores),
            np.asarray(payload["thresholds"], dtype=np.float64),
            dict(payload["threshold_policies"]),
        )
    raise ValueError(f"Unsupported traditional family: {family}")


def score_deep(
    model_spec: dict,
    raw_arrays,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    checkpoint = torch.load(
        project_path(model_spec["checkpoint"]["path"]),
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("test_evaluated") is not False:
        raise ValueError(f"Checkpoint is not development-frozen: {model_spec['name']}")
    if int(checkpoint["window_length"]) != 200:
        raise ValueError("Deep checkpoint window length changed")
    family = str(model_spec["family"])
    model = (
        build_cnn(checkpoint["model_config"])
        if family == "cnn"
        else build_cnn_transformer(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(raw_arrays.x), torch.from_numpy(raw_arrays.y)),
        batch_size=batch_size,
        shuffle=False,
    )
    scores, observed_labels = predict_scores(model, loader, torch.device("cpu"))
    if not np.array_equal(observed_labels, raw_arrays.y.astype(np.int8)):
        raise AssertionError("Deep DataLoader changed Test label order")
    return (
        scores,
        np.asarray(checkpoint["thresholds"], dtype=np.float64),
        dict(checkpoint["threshold_policies"]),
    )


def aggregate_event_rows(rows: list[dict], model_name: str) -> dict:
    selected = [row for row in rows if row["model"] == model_name and row["motor"] != "overall"]
    true_events = sum(int(row["true_events"]) for row in selected)
    predicted_events = sum(int(row["predicted_events"]) for row in selected)
    matched = sum(int(row["matched_events"]) for row in selected)
    precision = matched / predicted_events if predicted_events else 0.0
    recall = matched / true_events if true_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "model": model_name,
        "motor": "overall",
        "true_events": true_events,
        "predicted_events": predicted_events,
        "matched_events": matched,
        "false_alarm_events": predicted_events - matched,
        "missed_events": true_events - matched,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
    }


def metric_row(model: str, split: str, block: dict) -> dict:
    return {
        "model": model,
        "split": split,
        "pr_auc": block["pr_auc"],
        "precision": block["precision"],
        "recall": block["recall"],
        "f1": block["f1"],
        "roc_auc": block["roc_auc"],
        "n": block["n"],
        "positive_support": block["positive_support"],
        "negative_support": block["negative_support"],
        "tn": block["tn"],
        "fp": block["fp"],
        "fn": block["fn"],
        "tp": block["tp"],
    }


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    frozen_checks = preflight(config, args.archive)
    if args.preflight_only:
        print("Phase 6 preflight passed; Test labels and scores were not accessed")
        return

    status_path = args.output / "final_evaluation_status.json"
    attempt_number = 1
    previous_technical_failure = None
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        resumable = (
            args.resume_technical_failure
            and existing.get("status") == "failed_technical"
            and existing.get("metrics_written") is False
            and existing.get("protocol") == config["protocol"]["version"]
        )
        if not resumable:
            raise RuntimeError(
                "Final evaluation status already exists; refusing to evaluate Test again: "
                f"{existing.get('status')}"
            )
        attempt_number = int(existing["test_evaluation_attempt"]) + 1
        previous_technical_failure = existing
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        status_path,
        {
            "status": "started",
            "protocol": config["protocol"]["version"],
            "test_evaluation_attempt": attempt_number,
            "resumed_after_technical_failure": attempt_number > 1,
        },
    )

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    archive = KaggleRobotArchive(args.archive)
    test_manifest = pd.read_csv(
        project_path(config["data"]["test_manifest"]["path"]), dtype={"run_id": str}
    ).reset_index(drop=True)
    train_manifest = pd.read_csv(
        project_path(config["data"]["train_manifest"]["path"]), dtype={"run_id": str}
    )
    validation_manifest = pd.read_csv(
        project_path(config["data"]["validation_manifest"]["path"]),
        dtype={"run_id": str},
    )
    if set(test_manifest["split"]) != {"test"}:
        raise ValueError("Final evaluator received a non-Test manifest")
    test_runs = set(test_manifest["run_id"])
    if test_runs & set(train_manifest["run_id"]) or test_runs & set(validation_manifest["run_id"]):
        raise ValueError("Run leakage detected before final scoring")
    if set(test_manifest["window_length"]) != {int(config["data"]["window_length"])}:
        raise ValueError("Frozen Test window length changed")

    labels = labels_from_manifest(test_manifest)
    raw_scaler = ChannelStandardizer.load(
        project_path(config["data"]["raw_scaler"]["path"])
    )
    if test_runs & set(raw_scaler.fitted_run_ids):
        raise ValueError("Test run leaked into raw-channel scaler")
    raw_arrays = materialize_windows(archive, test_manifest, raw_scaler)
    if not np.array_equal(raw_arrays.y.astype(np.int8), labels):
        raise AssertionError("Manifest/materialized Test labels differ")

    feature_scaler = FeatureStandardizer.load(
        project_path(config["data"]["feature_scaler"]["path"])
    )
    if feature_scaler.fitted_split != "train":
        raise ValueError("Handcrafted feature scaler was not fitted on Train")
    raw_features, feature_names = extract_handcrafted_features(archive, test_manifest)
    if tuple(feature_names) != tuple(feature_scaler.feature_names):
        raise ValueError("Frozen handcrafted feature schema changed")
    features = feature_scaler.transform(raw_features)

    model_outputs: dict[str, dict[str, Any]] = {}
    for model_spec in config["models"]:
        name = str(model_spec["name"])
        if model_spec["family"] in {"gmm", "isolation_forest", "xgboost"}:
            scores, thresholds, threshold_policies = score_traditional(
                model_spec, features, feature_names
            )
        else:
            scores, thresholds, threshold_policies = score_deep(model_spec, raw_arrays)
        if scores.shape != labels.shape or thresholds.shape != (6,):
            raise ValueError(f"Unexpected score/threshold shape for {name}")
        if not np.isfinite(scores).all() or not np.isfinite(thresholds).all():
            raise ValueError(f"Non-finite scores/thresholds for {name}")
        model_outputs[name] = {
            "scores": scores,
            "thresholds": thresholds,
            "threshold_policies": threshold_policies,
            "predictions": (scores >= thresholds.reshape(1, 6)).astype(np.int8),
            "metrics": evaluate_multimotor(labels, scores, thresholds),
        }

    comparison_rows: list[dict] = []
    per_motor_rows: list[dict] = []
    nonoverlap_rows: list[dict] = []
    per_run_rows: list[dict] = []
    error_rows: list[dict] = []
    nonoverlap_index = greedy_nonoverlap_indices(test_manifest)
    for name, output in model_outputs.items():
        metrics = output["metrics"]
        comparison_rows.append(metric_row(name, "test", metrics["overall_micro"]))
        for motor in range(1, 7):
            block = metrics["per_motor"][f"m{motor}"]
            per_motor_rows.append(
                {
                    "model": name,
                    "motor": f"M{motor}",
                    "threshold": float(output["thresholds"][motor - 1]),
                    "threshold_policy": output["threshold_policies"][f"m{motor}"],
                    **{key: value for key, value in block.items() if key != "thresholds"},
                }
            )
        nonoverlap = evaluate_multimotor(
            labels[nonoverlap_index],
            output["scores"][nonoverlap_index],
            output["thresholds"],
        )
        nonoverlap_rows.append(
            metric_row(name, "test_greedy_nonoverlap", nonoverlap["overall_micro"])
        )
        for run_id, group in test_manifest.groupby("run_id", sort=True):
            index = group.index.to_numpy(dtype=np.int64)
            run_metrics = evaluate_multimotor(
                labels[index], output["scores"][index], output["thresholds"]
            )
            per_run_rows.append(
                {
                    "model": name,
                    "run_id": str(run_id),
                    "activity": str(group["activity"].iloc[0]),
                    "motor": "overall",
                    **run_metrics["overall_micro"],
                }
            )
            for motor in range(1, 7):
                per_run_rows.append(
                    {
                        "model": name,
                        "run_id": str(run_id),
                        "activity": str(group["activity"].iloc[0]),
                        "motor": f"M{motor}",
                        **run_metrics["per_motor"][f"m{motor}"],
                    }
                )
        for row_index, row in test_manifest.iterrows():
            for motor in range(1, 7):
                truth = int(labels[row_index, motor - 1])
                prediction = int(output["predictions"][row_index, motor - 1])
                if truth == prediction:
                    continue
                error_rows.append(
                    {
                        "model": name,
                        "error_type": "false_positive" if prediction else "false_negative",
                        "window_id": str(row["window_id"]),
                        "run_id": str(row["run_id"]),
                        "activity": str(row["activity"]),
                        "motor": f"M{motor}",
                        "start_idx": int(row["start_idx"]),
                        "end_idx": int(row["end_idx"]),
                        "fault_points": int(row[f"m{motor}_fault_points"]),
                        "label": truth,
                        "prediction": prediction,
                        "score": float(output["scores"][row_index, motor - 1]),
                        "threshold": float(output["thresholds"][motor - 1]),
                    }
                )

    point_labels_by_run = {
        str(run_id): archive.load_run(str(run_id)).labels for run_id in sorted(test_runs)
    }
    if any(value is None for value in point_labels_by_run.values()):
        raise ValueError("Raw Test point labels are missing")
    event_rows: list[dict] = []
    for name, output in model_outputs.items():
        for motor_index in range(6):
            event_rows.append(
                {
                    "model": name,
                    "motor": f"M{motor_index + 1}",
                    **event_counts_for_motor(
                        test_manifest,
                        point_labels_by_run,
                        output["predictions"][:, motor_index],
                        motor_index,
                    ),
                }
            )
        event_rows.append(aggregate_event_rows(event_rows, name))

    support_rows = []
    for motor in range(1, 7):
        positive = int(labels[:, motor - 1].sum())
        run_positive = int(
            test_manifest.groupby("run_id")[f"m{motor}_label"].max().sum()
        )
        support_rows.append(
            {
                "motor": f"M{motor}",
                "windows": len(test_manifest),
                "positive_windows": positive,
                "fault_window_ratio": positive / len(test_manifest),
                "positive_runs": run_positive,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    per_motor = pd.DataFrame(per_motor_rows)
    nonoverlap_frame = pd.DataFrame(nonoverlap_rows)
    event_frame = pd.DataFrame(event_rows)
    per_run = pd.DataFrame(per_run_rows)
    errors = pd.DataFrame(error_rows)
    support = pd.DataFrame(support_rows)
    comparison.to_csv(args.output / "model_comparison.csv", index=False)
    per_motor.to_csv(args.output / "per_motor_metrics.csv", index=False)
    nonoverlap_frame.to_csv(args.output / "nonoverlap_model_comparison.csv", index=False)
    event_frame.to_csv(args.output / "event_metrics.csv", index=False)
    per_run.to_csv(args.output / "per_run_metrics.csv", index=False)
    errors.to_csv(args.output / "window_errors.csv", index=False)
    support.to_csv(args.output / "test_support.csv", index=False)
    np.savez_compressed(
        args.output / "test_scores_and_labels.npz",
        labels=labels,
        **{
            name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_"): output[
                "scores"
            ]
            for name, output in model_outputs.items()
        },
    )
    shutil.copy2(args.config, args.output / "frozen_protocol.yaml")
    write_json(args.output / "frozen_input_checks.json", frozen_checks)
    write_json(
        args.output / "test_metrics.json",
        {
            "protocol": config["protocol"],
            "model_metrics": {
                name: {
                    "thresholds": output["thresholds"].tolist(),
                    "threshold_policies": output["threshold_policies"],
                    "test": output["metrics"],
                }
                for name, output in model_outputs.items()
            },
            "test_windows": len(test_manifest),
            "test_runs_with_windows": len(test_runs),
            "test_evaluated": True,
        },
    )
    write_json(
        args.output / "run_context.json",
        {
            "seed": seed,
            "protocol": config["protocol"]["version"],
            "test_evaluation_attempt": attempt_number,
            "training_performed": False,
            "threshold_selection_performed": False,
            "point_adjustment": False,
            "archive_sha256": archive.archive_sha256,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "test_evaluated": True,
        },
    )
    write_json(
        status_path,
        {
            "status": "completed",
            "protocol": config["protocol"]["version"],
            "test_evaluation_attempt": attempt_number,
            "resumed_after_technical_failure": attempt_number > 1,
            "test_evaluated": True,
            "training_performed": False,
            "threshold_selection_performed": False,
            "threshold_source": "frozen_validation_checkpoint",
            "previous_technical_failure": previous_technical_failure,
        },
    )
    print("Phase 6 single frozen Test evaluation completed")
    print(comparison[["model", "pr_auc", "precision", "recall", "f1"]].to_string(index=False))
    print("\nEvent-level overall")
    print(
        event_frame[event_frame["motor"] == "overall"][[
            "model", "event_precision", "event_recall", "event_f1"
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
