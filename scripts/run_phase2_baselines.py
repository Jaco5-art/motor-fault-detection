#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import platform
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pandas as pd
import sklearn
import yaml

from motor_fault.config import read_yaml
from motor_fault.data.archive import KaggleRobotArchive
from motor_fault.data.handcrafted import (
    FeatureStandardizer,
    extract_handcrafted_features,
    labels_from_manifest,
)
from motor_fault.evaluation.metrics import json_safe
from motor_fault.models.traditional import run_gmm_trials, run_isolation_forest_trials
from motor_fault.models.xgboost_baseline import run_xgboost_trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 Train/Validation baselines")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_baselines.yaml")
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments"))
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    if config["experiment"].get("test_access") != "forbidden":
        raise ValueError("Phase 2 development runner must forbid Test access")
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)

    archive = KaggleRobotArchive(args.archive)
    train_manifest = pd.read_csv(
        args.manifest_dir / "windows_train.csv", dtype={"run_id": str}
    )
    validation_manifest = pd.read_csv(
        args.manifest_dir / "windows_validation.csv", dtype={"run_id": str}
    )
    if set(train_manifest["split"]) != {"train"}:
        raise ValueError("Train manifest contains another split")
    if set(validation_manifest["split"]) != {"validation"}:
        raise ValueError("Validation manifest contains another split")

    statistics = tuple(config["features"]["statistics"])
    x_train_raw, feature_names = extract_handcrafted_features(
        archive, train_manifest, statistics
    )
    x_validation_raw, validation_names = extract_handcrafted_features(
        archive, validation_manifest, statistics
    )
    if validation_names != feature_names:
        raise AssertionError("Train/Validation feature names differ")
    feature_scaler = FeatureStandardizer.fit(x_train_raw, feature_names)
    x_train = feature_scaler.transform(x_train_raw)
    x_validation = feature_scaler.transform(x_validation_raw)
    y_train = labels_from_manifest(train_manifest)
    y_validation = labels_from_manifest(validation_manifest)

    feature_dir = args.output_root / "_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_scaler.save(feature_dir / "feature_scaler_train_only.json")
    np.savez_compressed(
        feature_dir / "train_validation_features.npz",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        feature_names=np.asarray(feature_names),
    )
    write_json(
        feature_dir / "metadata.json",
        {
            "test_features_extracted": False,
            "statistics": statistics,
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_validation.shape),
            "feature_scaler_fitted_split": feature_scaler.fitted_split,
            "archive_sha256": archive.archive_sha256,
            "manifest_sha256": {
                "train": file_sha256(args.manifest_dir / "windows_train.csv"),
                "validation": file_sha256(args.manifest_dir / "windows_validation.csv"),
            },
        },
    )

    context = {
        "seed": seed,
        "test_evaluated": False,
        "selection_split": "validation",
        "archive_sha256": archive.archive_sha256,
        "config_sha256": file_sha256(args.config),
        "train_windows": len(train_manifest),
        "validation_windows": len(validation_manifest),
        "feature_count": len(feature_names),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import xgboost

        context["xgboost"] = xgboost.__version__
    except ImportError:
        context["xgboost"] = "not-installed"

    experiment_specs = [
        (
            f"gmm_seed{seed}",
            lambda destination: run_gmm_trials(
                x_train,
                x_validation,
                y_validation,
                feature_names,
                config["gmm"],
                seed,
                destination,
            ),
        ),
        (
            f"isolation_forest_seed{seed}",
            lambda destination: run_isolation_forest_trials(
                x_train,
                x_validation,
                y_validation,
                feature_names,
                config["isolation_forest"],
                seed,
                destination,
            ),
        ),
        (
            f"xgboost_seed{seed}",
            lambda destination: run_xgboost_trials(
                x_train,
                x_validation,
                y_train,
                y_validation,
                config["xgboost"],
                seed,
                destination,
            ),
        ),
    ]

    rows = []
    for experiment_name, runner in experiment_specs:
        destination = args.output_root / experiment_name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, destination / "resolved_config.yaml")
        write_json(destination / "run_context.json", context)
        result = runner(destination)
        result["run_context"] = context
        write_json(destination / "validation_metrics.json", result)
        overall = result["validation"]["overall_micro"]
        rows.append(
            {
                "model": result["model"],
                "split": "validation",
                "pr_auc": overall["pr_auc"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "threshold_policy": "hybrid_support_aware",
                "test_evaluated": False,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_root / "phase2_validation_summary.csv", index=False)
    write_json(
        args.output_root / "phase2_status.json",
        {
            "status": "validation_complete",
            "test_evaluated": False,
            "experiments": [name for name, _ in experiment_specs],
        },
    )
    print("Phase 2 Train/Validation experiments completed; Test was not accessed")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
