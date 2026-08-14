#!/usr/bin/env python3
"""Audit the selected SSL checkpoint without evaluating the held-out Test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from motor_fault.config import read_yaml  # noqa: E402
from motor_fault.data.archive import KaggleRobotArchive  # noqa: E402
from motor_fault.data.scaler import ChannelStandardizer  # noqa: E402
from motor_fault.models.transformer import build_cnn_transformer  # noqa: E402
from motor_fault.training.cnn_trainer import materialize_windows, predict_scores  # noqa: E402


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "n_windows": int(len(y)),
        "n_positive": int(y.sum()),
        "pr_auc": float(average_precision_score(y, score))
        if len(np.unique(y)) == 2
        else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive": int(pred.sum()),
    }


def greedy_nonoverlap_indices(frame: pd.DataFrame) -> np.ndarray:
    selected: list[int] = []
    for _, group in frame.groupby("run_id", sort=True):
        last_end = -1
        for index, row in group.sort_values("start_idx").iterrows():
            if int(row["start_idx"]) > last_end:
                selected.append(int(index))
                last_end = int(row["end_idx"])
    return np.asarray(selected, dtype=np.int64)


def contiguous_positive_events(
    frame: pd.DataFrame, labels: np.ndarray, predicted: np.ndarray
) -> dict:
    working = frame.assign(label=labels, predicted=predicted)
    total = 0
    detected = 0
    for _, group in working.groupby("run_id", sort=True):
        active = False
        event_detected = False
        for row in group.sort_values("start_idx").itertuples(index=False):
            if int(row.label) == 1:
                if not active:
                    total += 1
                    active = True
                    event_detected = False
                event_detected = event_detected or bool(row.predicted)
            elif active:
                detected += int(event_detected)
                active = False
        if active:
            detected += int(event_detected)
    return {
        "n_fault_events": total,
        "n_detected_events": detected,
        "event_recall": float(detected / total) if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_ssl.yaml"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("artifacts/experiments/cnn_transformer_ssl_seed42"),
    )
    args = parser.parse_args()

    config = read_yaml(args.config)
    if config["experiment"].get("test_access") != "forbidden":
        raise ValueError("Audit requires Test access to remain forbidden")
    manifest = pd.read_csv(
        args.manifest_dir / "windows_validation.csv", dtype={"run_id": str}
    ).reset_index(drop=True)
    if set(manifest["split"]) != {"validation"}:
        raise ValueError("Audit input must contain Validation windows only")
    archive = KaggleRobotArchive(args.archive)
    scaler = ChannelStandardizer.load(args.manifest_dir / "scaler_train_only.json")
    arrays = materialize_windows(archive, manifest, scaler)

    checkpoint = torch.load(
        args.experiment_dir / "model.pt", map_location="cpu", weights_only=False
    )
    model = build_cnn_transformer(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(arrays.x), torch.from_numpy(arrays.y)),
        batch_size=int(config["data"]["finetune_batch_size"]),
        shuffle=False,
    )
    scores, labels = predict_scores(model, loader, torch.device("cpu"))
    motor_index = 5
    y = labels[:, motor_index].astype(np.int64)
    score = scores[:, motor_index]
    threshold = float(checkpoint["thresholds"][motor_index])
    pred = (score >= threshold).astype(np.int64)

    per_run: list[dict] = []
    for run_id, group in manifest.groupby("run_id", sort=True):
        index = group.index.to_numpy(dtype=np.int64)
        per_run.append(
            {
                "run_id": str(run_id),
                "activity": str(group["activity"].iloc[0]),
                **binary_metrics(y[index], score[index], threshold),
                "mean_normal_score": float(score[index][y[index] == 0].mean())
                if np.any(y[index] == 0)
                else None,
                "mean_fault_score": float(score[index][y[index] == 1].mean())
                if np.any(y[index] == 1)
                else None,
            }
        )

    nonoverlap = greedy_nonoverlap_indices(manifest)
    audit = {
        "scope": "Validation only; held-out Test was not evaluated",
        "motor": "M6",
        "frozen_threshold": threshold,
        "full_overlapping_windows": binary_metrics(y, score, threshold),
        "greedy_nonoverlapping_windows": binary_metrics(
            y[nonoverlap], score[nonoverlap], threshold
        ),
        "per_run": per_run,
        "window_event_recall": contiguous_positive_events(manifest, y, pred),
        "limitations": [
            "Validation contains only two runs with M6-positive windows.",
            "Fault type and activity are partly confounded; Test is needed for generalization claims.",
            "Event recall is based on contiguous positive-window regions, not raw point episodes.",
        ],
        "test_evaluated": False,
    }
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "robustness_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    pd.DataFrame(per_run).to_csv(
        args.experiment_dir / "validation_per_run.csv", index=False
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
