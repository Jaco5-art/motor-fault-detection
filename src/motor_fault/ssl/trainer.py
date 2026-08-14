from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import pandas as pd

from motor_fault.models.cnn import count_trainable_parameters, require_torch
from motor_fault.models.transformer import build_cnn_transformer
from motor_fault.ssl.contrastive import (
    ContrastiveEncoder,
    augment_batch,
    encoder_only_state_dict,
    overlap_aware_nt_xent,
)
from motor_fault.training.cnn_trainer import PreparedArrays, set_reproducible_seed

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    DataLoader = None
    TensorDataset = None


def build_ssl_metadata(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"run_id", "start_idx", "end_idx", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"SSL manifest lacks {sorted(missing)}")
    if set(manifest["split"]) != {"train"}:
        raise ValueError("SSL pretraining accepts Train windows only")
    categories = {run_id: index for index, run_id in enumerate(sorted(manifest["run_id"].unique()))}
    run_codes = manifest["run_id"].map(categories).to_numpy(dtype=np.int64)
    return (
        run_codes,
        manifest["start_idx"].to_numpy(dtype=np.int64),
        manifest["end_idx"].to_numpy(dtype=np.int64),
    )


def pretrain_contrastive_encoder(
    train_arrays: PreparedArrays,
    train_manifest: pd.DataFrame,
    model_config: dict[str, Any],
    augmentation_config: dict[str, Any],
    contrastive_config: dict[str, Any],
    batch_size: int,
    num_workers: int,
    seed: int,
    deterministic: bool,
    device: "torch.device",
) -> dict[str, Any]:
    require_torch()
    set_reproducible_seed(seed, deterministic)
    run_codes, starts, ends = build_ssl_metadata(train_manifest)
    base_encoder = build_cnn_transformer(model_config).to(device)
    model = ContrastiveEncoder(
        base_encoder,
        representation_dim=base_encoder.representation_dim,
        projection_hidden_dim=int(contrastive_config["projection_hidden_dim"]),
        projection_dim=int(contrastive_config["projection_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contrastive_config["learning_rate"]),
        weight_decay=float(contrastive_config["weight_decay"]),
    )
    loader_generator = torch.Generator().manual_seed(seed)
    augmentation_generator = torch.Generator(device=device).manual_seed(seed + 1)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_arrays.x),
            torch.from_numpy(run_codes),
            torch.from_numpy(starts),
            torch.from_numpy(ends),
        ),
        batch_size=int(batch_size),
        shuffle=True,
        generator=loader_generator,
        num_workers=int(num_workers),
        drop_last=False,
    )
    history = []
    for epoch in range(1, int(contrastive_config["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        overlap_weighted_sum = 0.0
        for inputs, batch_runs, batch_starts, batch_ends in loader:
            inputs = inputs.to(device)
            first_view = augment_batch(
                inputs,
                jitter_sigma=float(augmentation_config["jitter_sigma"]),
                scaling_sigma=float(augmentation_config["scaling_sigma"]),
                time_mask_ratio=float(augmentation_config["time_mask_ratio"]),
                generator=augmentation_generator,
            )
            second_view = augment_batch(
                inputs,
                jitter_sigma=float(augmentation_config["jitter_sigma"]),
                scaling_sigma=float(augmentation_config["scaling_sigma"]),
                time_mask_ratio=float(augmentation_config["time_mask_ratio"]),
                generator=augmentation_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            first_projection = model(first_view)
            second_projection = model(second_view)
            loss, diagnostics = overlap_aware_nt_xent(
                first_projection,
                second_projection,
                batch_runs,
                batch_starts,
                batch_ends,
                temperature=float(contrastive_config["temperature"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(contrastive_config["gradient_clip_norm"])
            )
            optimizer.step()
            count = len(inputs)
            loss_sum += float(loss.item()) * count
            overlap_weighted_sum += diagnostics["excluded_overlap_fraction"] * count
            samples += count
        history.append(
            {
                "epoch": epoch,
                "contrastive_loss": loss_sum / samples,
                "excluded_overlap_fraction": overlap_weighted_sum / samples,
            }
        )
    return {
        "model": model,
        "encoder_state_dict": encoder_only_state_dict(model),
        "full_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "history": history,
        "encoder_parameter_count": count_trainable_parameters(base_encoder),
        "contrastive_parameter_count": count_trainable_parameters(model),
        "train_windows": len(train_arrays.x),
        "train_runs": int(train_manifest["run_id"].nunique()),
        "labels_used": False,
    }
