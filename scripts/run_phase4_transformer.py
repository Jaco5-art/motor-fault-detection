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

from motor_fault.config import read_yaml
from motor_fault.data.archive import KaggleRobotArchive
from motor_fault.data.scaler import ChannelStandardizer
from motor_fault.evaluation.metrics import json_safe
from motor_fault.training.cnn_trainer import materialize_windows, train_candidate

try:
    import torch
except ImportError as exc:
    raise ImportError("Install requirements-deep.txt to run Phase 4") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Validation-only CNN + Transformer candidates"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase4_transformer.yaml")
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/cnn_transformer_seed42"),
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
    return {
        key: value for key, value in result.items() if key not in {"model", "state_dict"}
    }


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    experiment = config["experiment"]
    if experiment.get("test_access") != "forbidden":
        raise ValueError("Phase 4 development runner must forbid Test access")
    if int(config["data"]["window_length"]) != 200:
        raise ValueError("Phase 4 is frozen to the Validation-selected 200-point window")

    train_path = args.manifest_dir / "windows_train.csv"
    validation_path = args.manifest_dir / "windows_validation.csv"
    train_manifest = pd.read_csv(train_path, dtype={"run_id": str})
    validation_manifest = pd.read_csv(validation_path, dtype={"run_id": str})
    if set(train_manifest["split"]) != {"train"}:
        raise ValueError("Train manifest contains another split")
    if set(validation_manifest["split"]) != {"validation"}:
        raise ValueError("Validation manifest contains another split")
    if set(train_manifest["window_length"]) != {200} or set(
        validation_manifest["window_length"]
    ) != {200}:
        raise ValueError("Phase 4 manifest window length is not 200")

    archive = KaggleRobotArchive(args.archive)
    scaler_path = args.manifest_dir / "scaler_train_only.json"
    scaler = ChannelStandardizer.load(scaler_path)
    if scaler.archive_sha256 != archive.archive_sha256:
        raise ValueError("Scaler/archive checksum mismatch")
    if set(validation_manifest["run_id"].astype(str)) & set(scaler.fitted_run_ids):
        raise ValueError("Validation run leaked into the raw scaler")
    train_arrays = materialize_windows(archive, train_manifest, scaler)
    validation_arrays = materialize_windows(archive, validation_manifest, scaler)

    args.output.mkdir(parents=True, exist_ok=True)
    candidates_dir = args.output / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output / "resolved_config.yaml")
    seed = int(experiment["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for candidate_name, candidate_values in config["candidates"].items():
        model_config = {**config["shared_model"], **candidate_values}
        result = train_candidate(
            candidate_name=candidate_name,
            model_config=model_config,
            train_arrays=train_arrays,
            validation_arrays=validation_arrays,
            training_config=config["training"],
            data_config=config["data"],
            seed=seed,
            deterministic=bool(experiment["deterministic"]),
            device=device,
            model_family="cnn_transformer",
        )
        checkpoint = {
            "model_type": "CNNTransformerClassifier",
            "candidate": candidate_name,
            "model_config": result["model_config"],
            "model_state_dict": result["state_dict"],
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "thresholds": result["thresholds"],
            "threshold_policies": result["threshold_policies"],
            "train_only_pos_weight": result["train_only_pos_weight"],
            "initialization": config["training"]["initialization"],
            "channel_names": archive.channel_names,
            "window_length": 200,
            "scaler_path": str(scaler_path),
            "archive_sha256": archive.archive_sha256,
            "train_manifest_sha256": file_sha256(train_path),
            "validation_manifest_sha256": file_sha256(validation_path),
            "test_evaluated": False,
        }
        torch.save(checkpoint, candidates_dir / f"{candidate_name}.pt")
        pd.DataFrame(result["history"]).to_csv(
            candidates_dir / f"{candidate_name}_history.csv", index=False
        )
        write_json(
            candidates_dir / f"{candidate_name}_validation_metrics.json",
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
    shutil.copy2(candidates_dir / f"{best['candidate']}.pt", args.output / "model.pt")
    selected = serializable_result(best)
    selected.update(
        {
            "model": "CNN + Transformer",
            "selection_split": "validation",
            "model_selection_metric": "m6_validation_pr_auc",
            "architecture_selection_rule": config["training"][
                "architecture_selection_rule"
            ],
            "test_evaluated": False,
            "all_candidates": [
                {
                    "candidate": result["candidate"],
                    "parameter_count": result["parameter_count"],
                    "best_epoch": result["best_epoch"],
                    "epochs_ran": result["epochs_ran"],
                    "m6_validation_pr_auc": result["best_m6_validation_pr_auc"],
                    "m6_validation_f1": result["validation"]["per_motor"]["m6"][
                        "f1"
                    ],
                    "overall_validation_f1": result["validation"]["overall_micro"][
                        "f1"
                    ],
                }
                for result in results
            ],
        }
    )
    write_json(args.output / "validation_metrics.json", selected)
    write_json(
        args.output / "run_context.json",
        {
            "seed": seed,
            "test_evaluated": False,
            "selection_split": "validation",
            "initialization": "from_scratch",
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

    summary_path = args.output.parent / "development_validation_summary.csv"
    development = pd.read_csv(summary_path)
    development = development[development["model"] != "CNN + Transformer"]
    overall = best["validation"]["overall_micro"]
    m6 = best["validation"]["per_motor"]["m6"]
    transformer_row = pd.DataFrame(
        [
            {
                "model": "CNN + Transformer",
                "split": "validation",
                "pr_auc": overall["pr_auc"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "threshold_policy": "hybrid_support_aware",
                "test_evaluated": False,
                "m6_pr_auc": m6["pr_auc"],
                "m6_f1": m6["f1"],
            }
        ]
    )
    pd.concat([development, transformer_row], ignore_index=True).to_csv(
        summary_path, index=False
    )
    write_json(
        args.output.parent / "phase4_status.json",
        {
            "status": "validation_complete",
            "selected_candidate": best["candidate"],
            "checkpoint": str(args.output / "model.pt"),
            "test_evaluated": False,
        },
    )
    print("Phase 4 CNN + Transformer completed; Test was not accessed")
    print(pd.DataFrame(selected["all_candidates"]).to_string(index=False))
    print("Selected:", best["candidate"])
    print("Validation overall:", overall)
    print("Validation M6:", m6)


if __name__ == "__main__":
    main()
