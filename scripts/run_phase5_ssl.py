#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import yaml

from motor_fault.config import read_yaml
from motor_fault.data.archive import KaggleRobotArchive
from motor_fault.data.scaler import ChannelStandardizer
from motor_fault.evaluation.metrics import json_safe
from motor_fault.ssl.trainer import pretrain_contrastive_encoder
from motor_fault.training.cnn_trainer import (
    materialize_windows,
    predict_scores,
    train_candidate,
)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError("Install requirements-deep.txt to run Phase 5") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contrastive SSL + fine-tuning")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_ssl.yaml"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/cnn_transformer_ssl_seed42"),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Keep stability-seed results separate from the canonical Phase 5 summary.",
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


def generalization_diagnosis(
    model: "torch.nn.Module",
    train_arrays,
    validation_arrays,
    validation_manifest: pd.DataFrame,
    batch_size: int,
    device: "torch.device",
) -> dict:
    output = {"test_evaluated": False, "per_motor_pr_auc": {}, "m6_by_run": {}}
    split_payloads = {
        "train": (train_arrays, None),
        "validation": (validation_arrays, validation_manifest),
    }
    for split, (arrays, manifest) in split_payloads.items():
        loader = DataLoader(
            TensorDataset(torch.from_numpy(arrays.x), torch.from_numpy(arrays.y)),
            batch_size=batch_size,
            shuffle=False,
        )
        scores, labels = predict_scores(model, loader, device)
        output["per_motor_pr_auc"][split] = {
            f"m{motor + 1}": (
                float(average_precision_score(labels[:, motor], scores[:, motor]))
                if labels[:, motor].sum()
                else None
            )
            for motor in range(6)
        }
        if split == "validation":
            for run_id, indices in manifest.groupby("run_id").groups.items():
                index = np.asarray(list(indices), dtype=np.int64)
                run_labels = labels[index, 5]
                run_scores = scores[index, 5]
                output["m6_by_run"][str(run_id)] = {
                    "activity": str(manifest.loc[index[0], "activity"]),
                    "windows": int(len(index)),
                    "positive_windows": int(run_labels.sum()),
                    "mean_normal_score": (
                        float(run_scores[run_labels == 0].mean())
                        if np.any(run_labels == 0)
                        else None
                    ),
                    "mean_fault_score": (
                        float(run_scores[run_labels == 1].mean())
                        if np.any(run_labels == 1)
                        else None
                    ),
                }
    return output


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    experiment = config["experiment"]
    if experiment.get("test_access") != "forbidden":
        raise ValueError("Phase 5 runner must forbid Test access")
    if config["data"].get("pretraining_split") != "train_only":
        raise ValueError("SSL pretraining must use Train only")
    if config["data"].get("labels_used_for_pretraining") is not False:
        raise ValueError("SSL pretraining must ignore labels")
    if int(config["data"]["window_length"]) != 200:
        raise ValueError("Phase 5 is frozen to 200-point windows")

    train_path = args.manifest_dir / "windows_train.csv"
    validation_path = args.manifest_dir / "windows_validation.csv"
    train_manifest = pd.read_csv(train_path, dtype={"run_id": str})
    validation_manifest = pd.read_csv(validation_path, dtype={"run_id": str})
    if set(train_manifest["split"]) != {"train"}:
        raise ValueError("SSL input contains non-Train windows")
    if set(validation_manifest["split"]) != {"validation"}:
        raise ValueError("Fine-tuning Validation manifest is invalid")

    archive = KaggleRobotArchive(args.archive)
    scaler_path = args.manifest_dir / "scaler_train_only.json"
    scaler = ChannelStandardizer.load(scaler_path)
    if scaler.archive_sha256 != archive.archive_sha256:
        raise ValueError("Scaler/archive checksum mismatch")
    train_arrays = materialize_windows(archive, train_manifest, scaler)
    validation_arrays = materialize_windows(archive, validation_manifest, scaler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(args.seed if args.seed is not None else experiment["seed"])
    model_config = {
        "input_channels": int(config["data"]["input_channels"]),
        "output_motors": int(config["data"]["output_motors"]),
        **config["model"],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    resolved_config = copy.deepcopy(config)
    resolved_config["experiment"]["seed"] = seed
    with (args.output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)
    pretraining = pretrain_contrastive_encoder(
        train_arrays=train_arrays,
        train_manifest=train_manifest,
        model_config=model_config,
        augmentation_config=config["augmentations"],
        contrastive_config=config["contrastive"],
        batch_size=int(config["data"]["pretrain_batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        seed=seed,
        deterministic=bool(experiment["deterministic"]),
        device=device,
    )
    torch.save(
        {
            "model_type": "ContrastiveCNNTransformer",
            "model_config": model_config,
            "full_state_dict": pretraining["full_state_dict"],
            "encoder_state_dict": pretraining["encoder_state_dict"],
            "seed": seed,
            "pretraining_split": "train_only",
            "labels_used": False,
            "train_manifest_sha256": file_sha256(train_path),
            "archive_sha256": archive.archive_sha256,
            "test_evaluated": False,
        },
        args.output / "pretrain_checkpoint.pt",
    )
    pd.DataFrame(pretraining["history"]).to_csv(
        args.output / "pretrain_history.csv", index=False
    )
    write_json(
        args.output / "pretrain_metrics.json",
        {
            "train_windows": pretraining["train_windows"],
            "train_runs": pretraining["train_runs"],
            "labels_used": pretraining["labels_used"],
            "encoder_parameter_count": pretraining["encoder_parameter_count"],
            "contrastive_parameter_count": pretraining["contrastive_parameter_count"],
            "final_epoch": pretraining["history"][-1],
            "test_evaluated": False,
        },
    )

    finetune_data_config = {
        "window_length": int(config["data"]["window_length"]),
        "input_channels": int(config["data"]["input_channels"]),
        "output_motors": int(config["data"]["output_motors"]),
        "batch_size": int(config["data"]["finetune_batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
    }
    finetuned = train_candidate(
        candidate_name="transformer_2layer_ssl",
        model_config=config["model"],
        train_arrays=train_arrays,
        validation_arrays=validation_arrays,
        training_config=config["finetuning"],
        data_config=finetune_data_config,
        seed=seed,
        deterministic=bool(experiment["deterministic"]),
        device=device,
        model_family="cnn_transformer",
        initial_state_dict=pretraining["encoder_state_dict"],
    )
    if finetuned["initialization_report"]["loaded"] is not True:
        raise AssertionError("SSL encoder weights were not loaded before fine-tuning")
    checkpoint = {
        "model_type": "CNNTransformerClassifier",
        "candidate": finetuned["candidate"],
        "model_config": finetuned["model_config"],
        "model_state_dict": finetuned["state_dict"],
        "seed": seed,
        "best_epoch": finetuned["best_epoch"],
        "thresholds": finetuned["thresholds"],
        "threshold_policies": finetuned["threshold_policies"],
        "train_only_pos_weight": finetuned["train_only_pos_weight"],
        "initialization": "ssl_encoder",
        "pretrain_checkpoint": str(args.output / "pretrain_checkpoint.pt"),
        "channel_names": archive.channel_names,
        "window_length": 200,
        "scaler_path": str(scaler_path),
        "archive_sha256": archive.archive_sha256,
        "train_manifest_sha256": file_sha256(train_path),
        "validation_manifest_sha256": file_sha256(validation_path),
        "test_evaluated": False,
    }
    torch.save(checkpoint, args.output / "model.pt")
    pd.DataFrame(finetuned["history"]).to_csv(
        args.output / "finetune_history.csv", index=False
    )
    result = serializable_result(finetuned)
    result.update(
        {
            "model": "CNN + Transformer + SSL",
            "selection_split": "validation",
            "model_selection_metric": "m6_validation_pr_auc",
            "pretraining_split": "train_only",
            "pretraining_labels_used": False,
            "test_evaluated": False,
        }
    )
    write_json(args.output / "validation_metrics.json", result)
    diagnosis = generalization_diagnosis(
        finetuned["model"],
        train_arrays,
        validation_arrays,
        validation_manifest,
        batch_size=int(config["data"]["finetune_batch_size"]),
        device=device,
    )
    write_json(args.output / "generalization_diagnosis.json", diagnosis)
    write_json(
        args.output / "run_context.json",
        {
            "seed": seed,
            "test_evaluated": False,
            "pretraining_split": "train_only",
            "pretraining_labels_used": False,
            "selection_split": "validation",
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "archive_sha256": archive.archive_sha256,
            "config_sha256": file_sha256(args.config),
            "train_manifest_sha256": file_sha256(train_path),
            "validation_manifest_sha256": file_sha256(validation_path),
        },
    )

    experiment_root = args.manifest_dir.parent / "experiments"
    scratch_path = experiment_root / "cnn_transformer_seed42" / "validation_metrics.json"
    with scratch_path.open("r", encoding="utf-8") as handle:
        scratch = json.load(handle)
    ablation_rows = []
    for initialization, payload in (("without_ssl", scratch), ("with_ssl", result)):
        overall = payload["validation"]["overall_micro"]
        m6 = payload["validation"]["per_motor"]["m6"]
        ablation_rows.append(
            {
                "initialization": initialization,
                "pr_auc": overall["pr_auc"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "m6_pr_auc": m6["pr_auc"],
                "m6_f1": m6["f1"],
                "test_evaluated": False,
            }
        )
    pd.DataFrame(ablation_rows).to_csv(args.output / "ssl_ablation.csv", index=False)

    if not args.skip_summary:
        summary_path = experiment_root / "development_validation_summary.csv"
        development = pd.read_csv(summary_path)
        development = development[development["model"] != "CNN + Transformer + SSL"]
        overall = result["validation"]["overall_micro"]
        m6 = result["validation"]["per_motor"]["m6"]
        ssl_row = pd.DataFrame(
            [
                {
                    "model": "CNN + Transformer + SSL",
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
        pd.concat([development, ssl_row], ignore_index=True).to_csv(
            summary_path, index=False
        )
        write_json(
            experiment_root / "phase5_status.json",
            {
                "status": "validation_complete",
                "checkpoint": str(args.output / "model.pt"),
                "pretraining_split": "train_only",
                "pretraining_labels_used": False,
                "test_evaluated": False,
            },
        )
    print("Phase 5 SSL completed; Test was not accessed")
    print(pd.DataFrame(ablation_rows).to_string(index=False))
    print("Pretraining final:", pretraining["history"][-1])
    print("Fine-tuning best epoch:", finetuned["best_epoch"])


if __name__ == "__main__":
    main()
