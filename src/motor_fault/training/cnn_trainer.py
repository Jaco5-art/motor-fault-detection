from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from motor_fault.data.archive import KaggleRobotArchive
from motor_fault.data.dataset import NumpyWindowDataset
from motor_fault.data.scaler import ChannelStandardizer
from motor_fault.evaluation.metrics import evaluate_multimotor, select_support_aware_thresholds
from motor_fault.models.cnn import build_cnn, count_trainable_parameters, require_torch
from motor_fault.models.transformer import build_cnn_transformer

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass(frozen=True)
class PreparedArrays:
    x: np.ndarray
    y: np.ndarray


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def materialize_windows(
    archive: KaggleRobotArchive,
    manifest: pd.DataFrame,
    scaler: ChannelStandardizer,
) -> PreparedArrays:
    dataset = NumpyWindowDataset(archive, manifest, scaler=scaler, cache_runs=3)
    if len(dataset) == 0:
        raise ValueError("Cannot materialize an empty window manifest")
    first = dataset[0]
    x = np.empty((len(dataset), *first["x"].shape), dtype=np.float32)
    y = np.empty((len(dataset), 6), dtype=np.float32)
    x[0], y[0] = first["x"], first["y"]
    for index in range(1, len(dataset)):
        item = dataset[index]
        x[index], y[index] = item["x"], item["y"]
    return PreparedArrays(x=x, y=y)


def train_only_pos_weight(
    labels: np.ndarray, cap: float | None = None
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 6:
        raise ValueError(f"Expected Train labels [windows, 6], got {y.shape}")
    positives = y.sum(axis=0)
    negatives = len(y) - positives
    if np.any(positives == 0):
        raise ValueError(f"Train lacks positives for motors {np.where(positives == 0)[0] + 1}")
    weights = negatives / positives
    if cap is not None:
        if cap <= 0:
            raise ValueError("Positive-weight cap must be positive")
        weights = np.minimum(weights, float(cap))
    return weights.astype(np.float32)


def predict_scores(
    model: "nn.Module", loader: "DataLoader", device: "torch.device"
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(scores), np.concatenate(labels).astype(np.int8)


def benchmark_batch1_ms(
    model: "nn.Module", window_length: int, input_channels: int, device: "torch.device"
) -> float:
    model.eval()
    sample = torch.zeros(1, input_channels, window_length, device=device)
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(50):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))


def train_candidate(
    candidate_name: str,
    model_config: dict[str, Any],
    train_arrays: PreparedArrays,
    validation_arrays: PreparedArrays,
    training_config: dict[str, Any],
    data_config: dict[str, Any],
    seed: int,
    deterministic: bool,
    device: "torch.device",
    model_family: str = "cnn",
    initial_state_dict: dict[str, "torch.Tensor"] | None = None,
) -> dict[str, Any]:
    require_torch()
    set_reproducible_seed(seed, deterministic)
    full_model_config = {
        "input_channels": int(data_config["input_channels"]),
        "output_motors": int(data_config["output_motors"]),
        **model_config,
    }
    if model_family == "cnn":
        model = build_cnn(full_model_config).to(device)
    elif model_family == "cnn_transformer":
        model = build_cnn_transformer(full_model_config).to(device)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    initialization_report = {"loaded": False, "missing_keys": [], "unexpected_keys": []}
    if initial_state_dict is not None:
        incompatible = model.load_state_dict(initial_state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected or any(not key.startswith("head.") for key in missing):
            raise ValueError(
                f"Invalid pretrained encoder state: missing={missing}, unexpected={unexpected}"
            )
        initialization_report = {
            "loaded": True,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
    pos_weight_np = train_only_pos_weight(
        train_arrays.y, cap=training_config.get("positive_weight_cap")
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(pos_weight_np, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_arrays.x), torch.from_numpy(train_arrays.y)),
        batch_size=int(data_config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(data_config["num_workers"]),
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_arrays.x), torch.from_numpy(validation_arrays.y)
        ),
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
    )

    best_metric = -np.inf
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, int(training_config["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config["gradient_clip_norm"])
            )
            optimizer.step()
            loss_sum += float(loss.item()) * len(inputs)
            samples += len(inputs)
        validation_scores, validation_labels = predict_scores(
            model, validation_loader, device
        )
        m6_pr_auc = float(
            average_precision_score(validation_labels[:, 5], validation_scores[:, 5])
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / samples,
            "m6_validation_pr_auc": m6_pr_auc,
        }
        history.append(record)
        if m6_pr_auc > best_metric + float(training_config["early_stopping_min_delta"]):
            best_metric = m6_pr_auc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(training_config["early_stopping_patience"]):
            break

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    validation_scores, validation_labels = predict_scores(model, validation_loader, device)
    thresholds, threshold_policies = select_support_aware_thresholds(
        validation_labels, validation_scores
    )
    metrics = evaluate_multimotor(validation_labels, validation_scores, thresholds)
    return {
        "candidate": candidate_name,
        "model_family": model_family,
        "initialization_report": initialization_report,
        "model": model,
        "state_dict": best_state,
        "model_config": full_model_config,
        "parameter_count": count_trainable_parameters(model),
        "batch1_median_latency_ms": benchmark_batch1_ms(
            model,
            int(data_config["window_length"]),
            int(data_config["input_channels"]),
            device,
        ),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_m6_validation_pr_auc": best_metric,
        "train_only_pos_weight": pos_weight_np.tolist(),
        "thresholds": thresholds.tolist(),
        "threshold_policies": threshold_policies,
        "validation": metrics,
        "history": history,
    }
