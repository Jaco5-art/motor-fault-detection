from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


Interval = tuple[int, int]


def contiguous_intervals(binary_values: np.ndarray, offset: int = 0) -> list[Interval]:
    """Convert a binary point sequence to inclusive contiguous intervals."""
    values = np.asarray(binary_values, dtype=np.int8).reshape(-1)
    intervals: list[Interval] = []
    start: int | None = None
    for index, active in enumerate(values):
        if active and start is None:
            start = index + int(offset)
        elif not active and start is not None:
            intervals.append((start, index - 1 + int(offset)))
            start = None
    if start is not None:
        intervals.append((start, len(values) - 1 + int(offset)))
    return intervals


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping or directly adjacent inclusive intervals."""
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def maximum_overlap_matches(
    truth_intervals: list[Interval], predicted_intervals: list[Interval]
) -> int:
    """Maximum-cardinality one-to-one interval-overlap matching."""
    adjacency = [
        [
            truth_index
            for truth_index, (truth_start, truth_end) in enumerate(truth_intervals)
            if predicted_start <= truth_end and truth_start <= predicted_end
        ]
        for predicted_start, predicted_end in predicted_intervals
    ]
    truth_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for truth_index in adjacency[prediction_index]:
            if truth_index in visited:
                continue
            visited.add(truth_index)
            if truth_index not in truth_to_prediction or augment(
                truth_to_prediction[truth_index], visited
            ):
                truth_to_prediction[truth_index] = prediction_index
                return True
        return False

    matches = 0
    for prediction_index in range(len(predicted_intervals)):
        matches += int(augment(prediction_index, set()))
    return matches


def event_counts_for_motor(
    manifest: pd.DataFrame,
    point_labels_by_run: dict[str, np.ndarray],
    predicted_windows: np.ndarray,
    motor_index: int,
) -> dict[str, int | float]:
    """Score raw point events against merged alarm-window intervals.

    Only the raw point range covered by at least one evaluated window is eligible.
    Alarm windows are merged before maximum-cardinality one-to-one overlap matching.
    This is event aggregation, not point adjustment; window predictions are never edited.
    """
    predictions = np.asarray(predicted_windows, dtype=np.int8).reshape(-1)
    if len(predictions) != len(manifest):
        raise ValueError("Prediction/manifest length mismatch")
    true_events = 0
    predicted_events = 0
    matched_events = 0
    for run_id, group in manifest.groupby("run_id", sort=True):
        index = group.index.to_numpy(dtype=np.int64)
        coverage_start = int(group["start_idx"].min())
        coverage_end = int(group["end_idx"].max())
        labels = point_labels_by_run[str(run_id)]
        if labels.ndim != 2 or labels.shape[1] != 6:
            raise ValueError(f"Unexpected raw labels for run {run_id}: {labels.shape}")
        truth = contiguous_intervals(
            labels[coverage_start : coverage_end + 1, motor_index],
            offset=coverage_start,
        )
        alarm_intervals = [
            (int(row.start_idx), int(row.end_idx))
            for row, active in zip(group.itertuples(index=False), predictions[index])
            if active
        ]
        predicted = merge_intervals(alarm_intervals)
        true_events += len(truth)
        predicted_events += len(predicted)
        matched_events += maximum_overlap_matches(truth, predicted)
    false_alarms = predicted_events - matched_events
    missed_events = true_events - matched_events
    precision = matched_events / predicted_events if predicted_events else 0.0
    recall = matched_events / true_events if true_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_events": int(true_events),
        "predicted_events": int(predicted_events),
        "matched_events": int(matched_events),
        "false_alarm_events": int(false_alarms),
        "missed_events": int(missed_events),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
    }


def greedy_nonoverlap_indices(manifest: pd.DataFrame) -> np.ndarray:
    """Select the earliest mutually non-overlapping windows within each run."""
    selected: list[int] = []
    for _, group in manifest.groupby("run_id", sort=True):
        last_end = -1
        for index, row in group.sort_values("start_idx").iterrows():
            if int(row["start_idx"]) > last_end:
                selected.append(int(index))
                last_end = int(row["end_idx"])
    return np.asarray(selected, dtype=np.int64)
