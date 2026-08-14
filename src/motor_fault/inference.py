from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from motor_fault.data.archive import KaggleRobotArchive, RunData
from motor_fault.data.handcrafted import (
    DEFAULT_STATISTICS,
    FeatureStandardizer,
    _window_statistics,
)
from motor_fault.models.traditional import _gmm_scores


@dataclass(frozen=True)
class GMMInferenceBundle:
    checkpoint: dict[str, Any]
    feature_scaler: FeatureStandardizer


def build_window_spans(
    n_points: int, window_length: int = 200, stride: int = 10
) -> list[tuple[int, int]]:
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    if n_points < window_length:
        return []
    return [
        (start, start + window_length - 1)
        for start in range(0, n_points - window_length + 1, stride)
    ]


def extract_run_features(
    run: RunData,
    spans: list[tuple[int, int]],
    statistics: tuple[str, ...] = DEFAULT_STATISTICS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    feature_names = tuple(
        f"{channel}_{statistic}"
        for channel in run.channel_names
        for statistic in statistics
    )
    output = np.empty((len(spans), len(feature_names)), dtype=np.float32)
    for row_number, (start, end) in enumerate(spans):
        calculated = _window_statistics(run.features[start : end + 1])
        output[row_number] = np.stack(
            [calculated[name] for name in statistics], axis=1
        ).reshape(-1)
    if not np.isfinite(output).all():
        raise ValueError("Inference feature extraction produced NaN/Inf")
    return output, feature_names


def load_gmm_bundle(bundle_dir: str | Path | None = None) -> GMMInferenceBundle:
    if bundle_dir is not None:
        root = Path(bundle_dir)
        checkpoint = joblib.load(root / "model.joblib")
        scaler = FeatureStandardizer.load(root / "feature_scaler_train_only.json")
        return GMMInferenceBundle(checkpoint=checkpoint, feature_scaler=scaler)

    resource_root = files("motor_fault.resources.gmm")
    with as_file(resource_root / "model.joblib") as checkpoint_path:
        checkpoint = joblib.load(checkpoint_path)
    with as_file(resource_root / "feature_scaler_train_only.json") as scaler_path:
        scaler = FeatureStandardizer.load(scaler_path)
    return GMMInferenceBundle(checkpoint=checkpoint, feature_scaler=scaler)


def predict_archive_run(
    archive_path: str | Path,
    run_id: str,
    archive_split: str = "testing",
    window_length: int = 200,
    stride: int = 10,
    bundle_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    archive = KaggleRobotArchive(archive_path)
    run_id = str(run_id)
    if run_id not in archive.run_ids(archive_split):
        raise ValueError(f"Run {run_id} does not exist in archive split {archive_split}")
    run = archive.load_run(run_id, split=archive_split)
    spans = build_window_spans(run.n_rows, window_length, stride)
    if not spans:
        raise ValueError(
            f"Run {run_id} has {run.n_rows} points, shorter than window {window_length}"
        )
    bundle = load_gmm_bundle(bundle_dir)
    raw_features, feature_names = extract_run_features(run, spans)
    if tuple(feature_names) != tuple(bundle.feature_scaler.feature_names):
        raise ValueError("Input feature schema differs from the frozen GMM bundle")
    features = bundle.feature_scaler.transform(raw_features)
    checkpoint = bundle.checkpoint
    if tuple(checkpoint["feature_names"]) != tuple(feature_names):
        raise ValueError("GMM checkpoint feature schema mismatch")
    raw_scores = _gmm_scores(checkpoint["models"], features, feature_names)
    scores = checkpoint["calibrator"].transform(raw_scores)
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float64)
    predictions = (scores >= thresholds.reshape(1, 6)).astype(np.int8)

    rows: dict[str, Any] = {
        "window_id": [f"{run_id}:{start:06d}:{end:06d}" for start, end in spans],
        "run_id": run_id,
        "activity": run.activity,
        "start_idx": [start for start, _ in spans],
        "end_idx": [end for _, end in spans],
        "start_time": [float(run.time[start]) for start, _ in spans],
        "end_time": [float(run.time[end]) for _, end in spans],
    }
    for motor in range(1, 7):
        rows[f"m{motor}_score"] = scores[:, motor - 1]
        rows[f"m{motor}_threshold"] = float(thresholds[motor - 1])
        rows[f"m{motor}_predicted_fault"] = predictions[:, motor - 1]
    frame = pd.DataFrame(rows)
    motor_summary = {}
    for motor in range(1, 7):
        active = predictions[:, motor - 1].astype(bool)
        motor_summary[f"M{motor}"] = {
            "threshold": float(thresholds[motor - 1]),
            "predicted_fault_windows": int(active.sum()),
            "predicted_fault_ratio": float(active.mean()),
            "maximum_anomaly_score": float(scores[:, motor - 1].max()),
            "first_alarm_start_idx": int(frame.loc[active, "start_idx"].iloc[0])
            if active.any()
            else None,
        }
    summary = {
        "model": "GMM",
        "run_id": run_id,
        "archive_split": archive_split,
        "activity": run.activity,
        "n_points": run.n_rows,
        "window_length": window_length,
        "stride": stride,
        "n_windows": len(frame),
        "threshold_source": "frozen_validation",
        "motor_summary": motor_summary,
    }
    return frame, summary
