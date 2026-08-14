from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from motor_fault.data.archive import KaggleRobotArchive, RunData
from motor_fault.data.scaler import ChannelStandardizer

try:
    import torch
    from torch.utils.data import Dataset as TorchDatasetBase
except ImportError:  # Stage 0–1 can run without the deep-learning extra.
    torch = None
    TorchDatasetBase = object


class NumpyWindowDataset:
    """Lazy window reader returning [channels, sequence_length] arrays."""

    def __init__(
        self,
        archive: KaggleRobotArchive,
        manifest: pd.DataFrame,
        scaler: ChannelStandardizer | None = None,
        cache_runs: int = 3,
    ):
        self.archive = archive
        self.manifest = manifest.reset_index(drop=True).copy()
        self.scaler = scaler
        self.cache_runs = int(cache_runs)
        self._cache: OrderedDict[str, RunData] = OrderedDict()
        required = {"run_id", "start_idx", "end_idx", *[f"m{i}_label" for i in range(1, 7)]}
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_run(self, run_id: str) -> RunData:
        if run_id in self._cache:
            self._cache.move_to_end(run_id)
            return self._cache[run_id]
        run = self.archive.load_run(run_id)
        self._cache[run_id] = run
        while len(self._cache) > self.cache_runs:
            self._cache.popitem(last=False)
        return run

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(index)]
        run_id = str(row["run_id"])
        run = self._load_run(run_id)
        start = int(row["start_idx"])
        end_exclusive = int(row["end_idx"]) + 1
        features = run.features[start:end_exclusive]
        if self.scaler is not None:
            features = self.scaler.transform(features)
        labels = np.asarray([row[f"m{i}_label"] for i in range(1, 7)], dtype=np.float32)
        return {
            "x": np.ascontiguousarray(features.T, dtype=np.float32),
            "y": labels,
            "metadata": {
                "window_id": str(row.get("window_id", index)),
                "run_id": run_id,
                "start_idx": start,
                "end_idx": end_exclusive - 1,
            },
        }


class TorchWindowDataset(TorchDatasetBase):
    """PyTorch adapter; install `requirements-deep.txt` before use."""

    def __init__(self, numpy_dataset: NumpyWindowDataset):
        if torch is None:
            raise ImportError("PyTorch is not installed; install requirements-deep.txt")
        self.dataset = numpy_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        return {
            "x": torch.from_numpy(item["x"]),
            "y": torch.from_numpy(item["y"]),
            "metadata": item["metadata"],
        }
