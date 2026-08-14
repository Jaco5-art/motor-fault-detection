#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from motor_fault.config import read_yaml
from motor_fault.data.archive import KaggleRobotArchive
from motor_fault.data.dataset import NumpyWindowDataset
from motor_fault.data.scaler import ChannelStandardizer
from motor_fault.data.split import SplitDefinition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Stage 0–1 leakage invariants")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--split-config", type=Path, default=Path("configs/split_group_future_holdout_v1.yaml")
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    data_config = read_yaml(args.data_config)
    expected_length = int(data_config["window"]["length"])
    archive = KaggleRobotArchive(args.archive)
    split = SplitDefinition.from_yaml(args.split_config)
    split.validate(archive.run_ids("training"))
    scaler = ChannelStandardizer.load(args.manifest_dir / "scaler_train_only.json")

    checks = {
        "group_sets_disjoint": True,
        "all_labeled_runs_covered": True,
        "scaler_train_only": set(scaler.fitted_run_ids) == set(split.train),
        "scaler_archive_checksum_matches": scaler.archive_sha256 == archive.archive_sha256,
        "window_groups_match_split": True,
        "window_ids_unique": True,
        "window_lengths_fixed": True,
        "sample_shapes": {},
    }
    require(checks["scaler_train_only"], "Scaler contains non-training or missing training runs")
    require(checks["scaler_archive_checksum_matches"], "Scaler was fit on another archive")

    for split_name, run_ids in split.by_name.items():
        manifest = pd.read_csv(
            args.manifest_dir / f"windows_{split_name}.csv", dtype={"run_id": str}
        )
        groups_ok = set(manifest["run_id"]).issubset(set(run_ids))
        ids_unique = not manifest["window_id"].duplicated().any()
        lengths_ok = (
            ((manifest["end_idx"] - manifest["start_idx"] + 1) == expected_length).all()
            if len(manifest)
            else True
        )
        checks["window_groups_match_split"] &= bool(groups_ok)
        checks["window_ids_unique"] &= bool(ids_unique)
        checks["window_lengths_fixed"] &= bool(lengths_ok)
        require(groups_ok, f"{split_name} manifest contains a foreign run")
        require(ids_unique, f"{split_name} has duplicate window IDs")
        require(lengths_ok, f"{split_name} has a non-{expected_length} window")
        if len(manifest):
            dataset = NumpyWindowDataset(archive, manifest, scaler=scaler)
            sample = dataset[0]
            shape = list(sample["x"].shape)
            label_shape = list(sample["y"].shape)
            checks["sample_shapes"][split_name] = {"x": shape, "y": label_shape}
            require(shape == [18, expected_length], f"Bad {split_name} input shape: {shape}")
            require(label_shape == [6], f"Bad {split_name} label shape: {label_shape}")

    # Recompute aggregate statistics from exactly the scaler's training runs.
    scaled_train = np.concatenate(
        [scaler.transform(archive.load_run(run_id).features) for run_id in split.train], axis=0
    ).astype(np.float64)
    mean_error = float(np.max(np.abs(scaled_train.mean(axis=0))))
    std_error = float(np.max(np.abs(scaled_train.std(axis=0) - 1.0)))
    checks["train_scaled_max_abs_mean"] = mean_error
    checks["train_scaled_max_abs_std_error"] = std_error
    require(mean_error < 1e-4, f"Train scaling mean check failed: {mean_error}")
    require(std_error < 1e-4, f"Train scaling std check failed: {std_error}")

    output = args.manifest_dir / "pipeline_check.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"status": "passed", "checks": checks}, handle, indent=2)
        handle.write("\n")
    print("All leakage/data-contract checks passed")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
