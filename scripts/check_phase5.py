#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from motor_fault.models.transformer import build_cnn_transformer

try:
    import torch
except ImportError as exc:
    raise ImportError("Install requirements-deep.txt to check Phase 5") from exc


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SSL and fine-tuned checkpoints")
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("artifacts/experiments/cnn_transformer_ssl_seed42"),
    )
    args = parser.parse_args()
    pretrain = torch.load(
        args.experiment_dir / "pretrain_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    finetuned = torch.load(
        args.experiment_dir / "model.pt", map_location="cpu", weights_only=False
    )
    with (args.experiment_dir / "validation_metrics.json").open(
        "r", encoding="utf-8"
    ) as handle:
        validation_metrics = json.load(handle)
    with (args.experiment_dir / "run_context.json").open("r", encoding="utf-8") as handle:
        run_context = json.load(handle)
    train_checksum = digest(args.manifest_dir / "windows_train.csv")
    validation_checksum = digest(args.manifest_dir / "windows_validation.csv")
    checks = {
        "pretrain_train_only": pretrain.get("pretraining_split") == "train_only",
        "pretrain_labels_unused": pretrain.get("labels_used") is False,
        "pretrain_test_unused": pretrain.get("test_evaluated") is False,
        "pretrain_train_manifest_checksum": pretrain["train_manifest_sha256"]
        == train_checksum,
        "finetune_ssl_initialization": finetuned.get("initialization") == "ssl_encoder",
        "finetune_test_unused": finetuned.get("test_evaluated") is False,
        "finetune_train_manifest_checksum": finetuned["train_manifest_sha256"]
        == train_checksum,
        "finetune_validation_manifest_checksum": finetuned["validation_manifest_sha256"]
        == validation_checksum,
        "six_frozen_thresholds": len(finetuned["thresholds"]) == 6,
        "encoder_report_loaded": validation_metrics["initialization_report"]["loaded"]
        is True,
        "encoder_report_only_head_missing": sorted(
            validation_metrics["initialization_report"]["missing_keys"]
        )
        == ["head.1.bias", "head.1.weight"],
        "encoder_report_no_unexpected_keys": validation_metrics[
            "initialization_report"
        ]["unexpected_keys"]
        == [],
        "run_context_labels_unused": run_context["pretraining_labels_used"] is False,
        "run_context_test_unused": run_context["test_evaluated"] is False,
    }
    model = build_cnn_transformer(finetuned["model_config"])
    model.load_state_dict(finetuned["model_state_dict"])
    with torch.inference_mode():
        output = model(torch.zeros(2, 18, 200))
    checks["input_output_contract"] = list(output.shape) == [2, 6]
    if not all(checks.values()):
        raise AssertionError(checks)
    with (args.experiment_dir / "phase5_check.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "passed", "checks": checks}, handle, indent=2)
        handle.write("\n")
    print("All Phase 5 SSL checks passed")


if __name__ == "__main__":
    main()
