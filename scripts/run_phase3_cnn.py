#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import platform
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from motor_fault.config import read_yaml
from motor_fault.data.archive import KaggleRobotArchive
from motor_fault.data.scaler import ChannelStandardizer
from motor_fault.evaluation.metrics import json_safe
from motor_fault.training.cnn_trainer import (
    materialize_windows,
    train_candidate,
)

try:
    import torch
except ImportError as exc:
    raise ImportError("Install requirements-deep.txt to run Phase 3") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 3 Validation-only CNN")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_cnn.yaml"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/experiments/cnn_seed42")
    )
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


def serializable_result(result: dict) -> dict:
    excluded = {"model", "state_dict"}
    return {key: value for key, value in result.items() if key not in excluded}


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    experiment_config = config["experiment"]
    if experiment_config.get("test_access") != "forbidden":
        raise ValueError("Phase 3 development runner must forbid Test access")
    if int(config["data"]["window_length"]) != 200:
        raise ValueError("Phase 3 CNN is frozen to the Validation-selected 200-point window")

    train_path = args.manifest_dir / "windows_train.csv"
    validation_path = args.manifest_dir / "windows_validation.csv"
    train_manifest = pd.read_csv(train_path, dtype={"run_id": str})
    validation_manifest = pd.read_csv(validation_path, dtype={"run_id": str})
    if set(train_manifest["split"]) != {"train"}:
        raise ValueError("Train manifest contains another split")
    if set(validation_manifest["split"]) != {"validation"}:
        raise ValueError("Validation manifest contains another split")
    observed_lengths = set(
        pd.concat([train_manifest["window_length"], validation_manifest["window_length"]])
    )
    if observed_lengths != {int(config["data"]["window_length"])}:
        raise ValueError(f"Manifest/config window mismatch: {observed_lengths}")

    archive = KaggleRobotArchive(args.archive)
    scaler_path = args.manifest_dir / "scaler_train_only.json"
    scaler = ChannelStandardizer.load(scaler_path)
    if scaler.archive_sha256 != archive.archive_sha256:
        raise ValueError("Scaler/archive checksum mismatch")
    validation_runs = set(validation_manifest["run_id"].astype(str))
    if validation_runs & set(scaler.fitted_run_ids):
        raise ValueError("Validation run leaked into the raw-channel scaler")

    train_arrays = materialize_windows(archive, train_manifest, scaler)
    validation_arrays = materialize_windows(archive, validation_manifest, scaler)
    expected_train_shape = (
        len(train_manifest),
        int(config["data"]["input_channels"]),
        int(config["data"]["window_length"]),
    )
    if train_arrays.x.shape != expected_train_shape:
        raise ValueError(f"Unexpected Train tensor shape: {train_arrays.x.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(experiment_config["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output / "resolved_config.yaml")

    results = []
    for candidate_name, candidate_config in config["candidates"].items():
        result = train_candidate(
            candidate_name=candidate_name,
            model_config=candidate_config,
            train_arrays=train_arrays,
            validation_arrays=validation_arrays,
            training_config=config["training"],
            data_config=config["data"],
            seed=seed,
            deterministic=bool(experiment_config["deterministic"]),
            device=device,
        )
        checkpoint = {
            "model_type": "CNN1DClassifier",
            "candidate": candidate_name,
            "model_config": result["model_config"],
            "model_state_dict": result["state_dict"],
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "thresholds": result["thresholds"],
            "threshold_policies": result["threshold_policies"],
            "train_only_pos_weight": result["train_only_pos_weight"],
            "channel_names": archive.channel_names,
            "window_length": int(config["data"]["window_length"]),
            "scaler_path": str(scaler_path),
            "archive_sha256": archive.archive_sha256,
            "train_manifest_sha256": file_sha256(train_path),
            "validation_manifest_sha256": file_sha256(validation_path),
            "test_evaluated": False,
        }
        torch.save(checkpoint, candidate_dir / f"{candidate_name}.pt")
        pd.DataFrame(result["history"]).to_csv(
            candidate_dir / f"{candidate_name}_history.csv", index=False
        )
        write_json(
            candidate_dir / f"{candidate_name}_validation_metrics.json",
            serializable_result(result),
        )
        results.append(result)

    best_pr_auc = max(result["best_m6_validation_pr_auc"] for result in results)
    tolerance = float(config["training"]["near_best_pr_auc_tolerance"])
    near_best = [
        result
        for result in results
        if result["best_m6_validation_pr_auc"] >= best_pr_auc - tolerance
    ]
    best = min(
        near_best,
        key=lambda result: (
            result["parameter_count"],
            -result["validation"]["per_motor"]["m6"]["f1"],
        ),
    )
    selected_source = candidate_dir / f"{best['candidate']}.pt"
    shutil.copy2(selected_source, args.output / "model.pt")
    selected = serializable_result(best)
    selected["model"] = "1D-CNN"
    selected["selection_split"] = "validation"
    selected["model_selection_metric"] = "m6_validation_pr_auc"
    selected["architecture_selection_rule"] = config["training"][
        "architecture_selection_rule"
    ]
    selected["test_evaluated"] = False
    selected["all_candidates"] = [
        {
            "candidate": result["candidate"],
            "parameter_count": result["parameter_count"],
            "best_epoch": result["best_epoch"],
            "epochs_ran": result["epochs_ran"],
            "m6_validation_pr_auc": result["best_m6_validation_pr_auc"],
            "m6_validation_f1": result["validation"]["per_motor"]["m6"]["f1"],
            "overall_validation_f1": result["validation"]["overall_micro"]["f1"],
        }
        for result in results
    ]
    write_json(args.output / "validation_metrics.json", selected)
    write_json(
        args.output / "run_context.json",
        {
            "seed": seed,
            "test_evaluated": False,
            "selection_split": "validation",
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "train_windows": len(train_manifest),
            "validation_windows": len(validation_manifest),
            "archive_sha256": archive.archive_sha256,
            "config_sha256": file_sha256(args.config),
            "train_manifest_sha256": file_sha256(train_path),
            "validation_manifest_sha256": file_sha256(validation_path),
        },
    )

    phase2_path = args.output.parent / "phase2_validation_summary.csv"
    development = pd.read_csv(phase2_path) if phase2_path.exists() else pd.DataFrame()
    overall = best["validation"]["overall_micro"]
    m6 = best["validation"]["per_motor"]["m6"]
    cnn_row = pd.DataFrame(
        [
            {
                "model": "1D-CNN",
                "split": "validation",
                "pr_auc": overall["pr_auc"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "m6_pr_auc": m6["pr_auc"],
                "m6_f1": m6["f1"],
                "threshold_policy": "hybrid_support_aware",
                "test_evaluated": False,
            }
        ]
    )
    if not development.empty:
        development["m6_pr_auc"] = np.nan
        development["m6_f1"] = np.nan
        baseline_directories = {
            "GMM": "gmm_seed42",
            "Isolation Forest": "isolation_forest_seed42",
            "XGBoost": "xgboost_seed42",
        }
        for model_name, directory in baseline_directories.items():
            metrics_path = args.output.parent / directory / "validation_metrics.json"
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as handle:
                    baseline_metrics = json.load(handle)["validation"]["per_motor"]["m6"]
                development.loc[development["model"] == model_name, "m6_pr_auc"] = (
                    baseline_metrics["pr_auc"]
                )
                development.loc[development["model"] == model_name, "m6_f1"] = (
                    baseline_metrics["f1"]
                )
    pd.concat([development, cnn_row], ignore_index=True).to_csv(
        args.output.parent / "development_validation_summary.csv", index=False
    )
    write_json(
        args.output.parent / "phase3_status.json",
        {
            "status": "validation_complete",
            "selected_candidate": best["candidate"],
            "checkpoint": str(args.output / "model.pt"),
            "test_evaluated": False,
        },
    )
    print("Phase 3 CNN completed; Test was not accessed")
    print(
        pd.DataFrame(selected["all_candidates"]).to_string(index=False)
    )
    print("Selected:", best["candidate"])
    print("Validation overall:", overall)
    print("Validation M6:", m6)


if __name__ == "__main__":
    main()
