#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from motor_fault.data.archive import KaggleRobotArchive
from motor_fault.data.scaler import ChannelStandardizer
from motor_fault.models.cnn import build_cnn
from motor_fault.training.cnn_trainer import materialize_windows

try:
    import torch
except ImportError as exc:
    raise ImportError("Install requirements-deep.txt to diagnose Phase 3") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose CNN Train-to-Validation transfer")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/experiments/cnn_seed42/model.pt")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/cnn_seed42/generalization_diagnosis.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("test_evaluated") is not False:
        raise ValueError("Checkpoint does not attest that Test remained locked")
    archive = KaggleRobotArchive(args.archive)
    scaler = ChannelStandardizer.load(args.manifest_dir / "scaler_train_only.json")
    model = build_cnn(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    result = {"test_evaluated": False, "per_motor_pr_auc": {}, "m6_by_run": {}}
    for split in ("train", "validation"):
        manifest = pd.read_csv(
            args.manifest_dir / f"windows_{split}.csv", dtype={"run_id": str}
        )
        arrays = materialize_windows(archive, manifest, scaler)
        with torch.inference_mode():
            scores = torch.sigmoid(model(torch.from_numpy(arrays.x))).numpy()
        result["per_motor_pr_auc"][split] = {
            f"m{motor + 1}": (
                float(average_precision_score(arrays.y[:, motor], scores[:, motor]))
                if arrays.y[:, motor].sum()
                else None
            )
            for motor in range(6)
        }
        if split == "validation":
            for run_id, indices in manifest.groupby("run_id").groups.items():
                index = np.asarray(list(indices), dtype=np.int64)
                labels = arrays.y[index, 5]
                run_scores = scores[index, 5]
                result["m6_by_run"][str(run_id)] = {
                    "activity": str(manifest.loc[index[0], "activity"]),
                    "windows": int(len(index)),
                    "positive_windows": int(labels.sum()),
                    "mean_normal_score": (
                        float(run_scores[labels == 0].mean()) if np.any(labels == 0) else None
                    ),
                    "mean_fault_score": (
                        float(run_scores[labels == 1].mean()) if np.any(labels == 1) else None
                    ),
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
