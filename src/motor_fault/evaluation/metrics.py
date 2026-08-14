from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> ThresholdResult:
    labels = np.asarray(y_true, dtype=np.int8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != values.shape:
        raise ValueError("Labels and scores must have the same flattened shape")
    if labels.sum() == 0:
        raise ValueError("F1 threshold selection requires validation positives")
    candidates = np.unique(values)
    best = None
    for threshold in candidates:
        predicted = values >= threshold
        result = ThresholdResult(
            threshold=float(threshold),
            f1=float(f1_score(labels, predicted, zero_division=0)),
            precision=float(precision_score(labels, predicted, zero_division=0)),
            recall=float(recall_score(labels, predicted, zero_division=0)),
        )
        # Deterministic tie-break: F1, then Recall, then Precision, then higher threshold.
        key = (result.f1, result.recall, result.precision, result.threshold)
        if best is None or key > best[0]:
            best = (key, result)
    return best[1]


def _metric_block_from_predictions(
    y_true: np.ndarray, scores: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.int8).reshape(-1)
    support = int(labels.sum())
    negatives = int((labels == 0).sum())
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    result: dict[str, Any] = {
        "n": int(len(labels)),
        "positive_support": support,
        "negative_support": negatives,
        "pr_auc": float(average_precision_score(labels, values)) if support else None,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, values)) if support and negatives else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return result


def _metric_block(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    return _metric_block_from_predictions(y_true, values, values >= threshold)


def evaluate_multimotor(
    labels: np.ndarray, scores: np.ndarray, threshold: float | np.ndarray | list[float]
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 2 or y.shape[1] != 6:
        raise ValueError(f"Expected matching [windows, 6] arrays, got {y.shape}, {s.shape}")
    thresholds = np.asarray(threshold, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = np.repeat(thresholds.item(), 6)
    if thresholds.shape != (6,):
        raise ValueError(f"Threshold must be scalar or length 6, got {thresholds.shape}")
    predicted = (s >= thresholds.reshape(1, 6)).astype(np.int8)
    return {
        "thresholds": thresholds.tolist(),
        "overall_micro": _metric_block_from_predictions(y, s, predicted),
        "per_motor": {
            f"m{motor}": _metric_block(
                y[:, motor - 1], s[:, motor - 1], thresholds[motor - 1]
            )
            for motor in range(1, 7)
        },
    }


def select_support_aware_thresholds(
    labels: np.ndarray, scores: np.ndarray, normal_percentile: float = 95.0
) -> tuple[np.ndarray, dict[str, str]]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 2 or y.shape[1] != 6:
        raise ValueError("Expected matching [windows, 6] arrays")
    thresholds = np.empty(6, dtype=np.float64)
    policies = {}
    for motor in range(6):
        if y[:, motor].sum() > 0:
            thresholds[motor] = select_f1_threshold(y[:, motor], s[:, motor]).threshold
            policies[f"m{motor + 1}"] = "validation_f1_optimal"
        else:
            normal_scores = s[y[:, motor] == 0, motor]
            thresholds[motor] = float(np.percentile(normal_scores, normal_percentile))
            policies[f"m{motor + 1}"] = f"validation_normal_percentile_{normal_percentile:g}"
    return thresholds, policies


def compare_thresholds(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    percentile_thresholds = np.asarray(
        [np.percentile(s[:, motor], 95) for motor in range(6)], dtype=np.float64
    )
    hybrid_thresholds, policies = select_support_aware_thresholds(y, s)
    candidates = {
        "fixed_0.5": np.full(6, 0.5),
        "per_motor_validation_percentile_95": percentile_thresholds,
        "hybrid_support_aware": hybrid_thresholds,
    }
    output = {}
    for name, thresholds in candidates.items():
        output[name] = evaluate_multimotor(y, s, thresholds)["overall_micro"] | {
            "thresholds": thresholds.tolist()
        }
    output["hybrid_support_aware"]["per_motor_policy"] = policies
    return output


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
