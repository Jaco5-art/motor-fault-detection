#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motor_fault.config import read_yaml
from motor_fault.data.archive import KaggleRobotArchive, MOTORS
from motor_fault.data.scaler import ChannelStandardizer
from motor_fault.data.split import SplitDefinition
from motor_fault.data.windowing import WindowSpec, build_window_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build immutable split/window manifests")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--split-config", type=Path, default=Path("configs/split_group_future_holdout_v1.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifests"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = read_yaml(args.data_config)
    split = SplitDefinition.from_yaml(args.split_config)
    spec = WindowSpec(**data_config["window"])
    archive = KaggleRobotArchive(args.archive)
    split.validate(archive.run_ids("training"))

    args.output.mkdir(parents=True, exist_ok=True)
    run_manifest = split.build_run_manifest(archive)
    run_manifest.to_csv(args.output / "runs.csv", index=False)

    window_manifests = build_window_manifests(archive, split, spec)
    for split_name, manifest in window_manifests.items():
        manifest.to_csv(args.output / f"windows_{split_name}.csv", index=False)

    scaler = ChannelStandardizer.fit(archive, split.train)
    scaler.save(args.output / "scaler_train_only.json")

    summary = {
        "split_name": split.name,
        "archive_sha256": archive.archive_sha256,
        "window": {
            "length": spec.length,
            "stride": spec.stride,
            "label_rule": spec.label_rule,
            "min_fault_points": spec.min_fault_points,
        },
        "scaler_fitted_run_ids": list(scaler.fitted_run_ids),
        "splits": {},
    }
    for split_name, run_ids in split.by_name.items():
        manifest = window_manifests[split_name]
        split_summary = {
            "runs": len(run_ids),
            "runs_with_full_windows": int(manifest["run_id"].nunique()) if len(manifest) else 0,
            "windows": len(manifest),
            "positive_windows": {},
            "positive_window_ratio": {},
        }
        for motor in MOTORS:
            positives = int(manifest[f"m{motor}_label"].sum()) if len(manifest) else 0
            split_summary["positive_windows"][f"m{motor}"] = positives
            split_summary["positive_window_ratio"][f"m{motor}"] = (
                float(positives / len(manifest)) if len(manifest) else 0.0
            )
        summary["splits"][split_name] = split_summary
    with (args.output / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Built manifests for split={split.name}")
    for split_name, manifest in window_manifests.items():
        print(f"{split_name}: {len(manifest)} windows, {manifest['run_id'].nunique()} runs")
    print(f"Scaler fit on {scaler.n_samples} raw training timesteps only")


if __name__ == "__main__":
    main()
