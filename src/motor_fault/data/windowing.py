from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from motor_fault.data.archive import KaggleRobotArchive, MOTORS, RunData
from motor_fault.data.split import SplitDefinition


@dataclass(frozen=True)
class WindowSpec:
    length: int = 400
    stride: int = 10
    label_rule: str = "any"
    min_fault_points: int = 1

    def __post_init__(self) -> None:
        if self.length <= 0 or self.stride <= 0:
            raise ValueError("Window length and stride must be positive")
        if self.label_rule not in {"any", "endpoint", "min_points"}:
            raise ValueError(f"Unsupported label rule: {self.label_rule}")
        if self.min_fault_points <= 0:
            raise ValueError("min_fault_points must be positive")


def _labels_from_counts(
    run: RunData,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    spec: WindowSpec,
) -> np.ndarray:
    if spec.label_rule == "endpoint":
        return run.labels[ends - 1].astype(np.int8)
    if spec.label_rule == "min_points":
        return (counts >= spec.min_fault_points).astype(np.int8)
    return (counts > 0).astype(np.int8)


def build_run_window_manifest(
    run: RunData, split_name: str, spec: WindowSpec
) -> pd.DataFrame:
    if run.labels is None:
        raise ValueError("Window labels require a labeled training-data run")
    if run.n_rows < spec.length:
        return pd.DataFrame()

    starts = np.arange(0, run.n_rows - spec.length + 1, spec.stride, dtype=np.int64)
    ends = starts + spec.length
    prefix = np.vstack(
        [np.zeros((1, len(MOTORS)), dtype=np.int64), np.cumsum(run.labels, axis=0)]
    )
    counts = prefix[ends] - prefix[starts]
    labels = _labels_from_counts(run, starts, ends, counts, spec)

    records = []
    for row_index, (start, end) in enumerate(zip(starts, ends)):
        record = {
            "window_id": f"{split_name}:{run.run_id}:{start:06d}:{end-1:06d}",
            "split": split_name,
            "run_id": run.run_id,
            "activity": run.activity,
            "start_idx": int(start),
            "end_idx": int(end - 1),
            "start_time": float(run.time[start]),
            "end_time": float(run.time[end - 1]),
            "window_length": spec.length,
            "label_rule": spec.label_rule,
        }
        for motor in MOTORS:
            record[f"m{motor}_fault_points"] = int(counts[row_index, motor - 1])
            record[f"m{motor}_label"] = int(labels[row_index, motor - 1])
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_window_manifests(
    archive: KaggleRobotArchive,
    split: SplitDefinition,
    spec: WindowSpec,
) -> dict[str, pd.DataFrame]:
    split.validate(archive.run_ids("training"))
    manifests = {}
    columns = [
        "window_id",
        "split",
        "run_id",
        "activity",
        "start_idx",
        "end_idx",
        "start_time",
        "end_time",
        "window_length",
        "label_rule",
        *[item for motor in MOTORS for item in (f"m{motor}_fault_points", f"m{motor}_label")],
    ]
    for split_name, run_ids in split.by_name.items():
        frames = [
            build_run_window_manifest(archive.load_run(run_id), split_name, spec)
            for run_id in run_ids
        ]
        non_empty = [frame for frame in frames if not frame.empty]
        manifests[split_name] = (
            pd.concat(non_empty, ignore_index=True)
            if non_empty
            else pd.DataFrame(columns=columns)
        )
    return manifests
