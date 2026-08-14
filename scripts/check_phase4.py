#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from motor_fault.models.cnn import count_trainable_parameters
from motor_fault.models.transformer import build_cnn_transformer

try:
    import torch
except ImportError as exc:
    raise ImportError("Install requirements-deep.txt to check Phase 4") from exc


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen Phase 4 checkpoint")
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/experiments/cnn_transformer_seed42/model.pt"),
    )
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checks = {
        "model_type": checkpoint.get("model_type") == "CNNTransformerClassifier",
        "from_scratch": checkpoint.get("initialization") == "from_scratch",
        "test_evaluated_false": checkpoint.get("test_evaluated") is False,
        "train_manifest_checksum": checkpoint["train_manifest_sha256"]
        == digest(args.manifest_dir / "windows_train.csv"),
        "validation_manifest_checksum": checkpoint["validation_manifest_sha256"]
        == digest(args.manifest_dir / "windows_validation.csv"),
        "window_length_200": checkpoint["window_length"] == 200,
        "six_frozen_thresholds": len(checkpoint["thresholds"]) == 6,
    }
    model = build_cnn_transformer(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    with torch.inference_mode():
        output = model(torch.zeros(2, 18, 200))
    checks["input_output_contract"] = list(output.shape) == [2, 6]
    checks["parameter_count_positive"] = count_trainable_parameters(model) > 0
    if not all(checks.values()):
        raise AssertionError(checks)
    output_path = args.checkpoint.parent / "phase4_check.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"status": "passed", "checks": checks}, handle, indent=2)
        handle.write("\n")
    print("All Phase 4 checkpoint checks passed")


if __name__ == "__main__":
    main()
