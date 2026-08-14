from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from motor_fault.data.archive import KaggleRobotArchive


@dataclass(frozen=True)
class ChannelStandardizer:
    channel_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    fitted_run_ids: tuple[str, ...]
    n_samples: int
    archive_sha256: str

    @classmethod
    def fit(
        cls, archive: KaggleRobotArchive, train_run_ids: list[str] | tuple[str, ...]
    ) -> "ChannelStandardizer":
        run_ids = tuple(map(str, train_run_ids))
        if not run_ids:
            raise ValueError("Cannot fit a scaler without training runs")
        available = set(archive.run_ids("training"))
        unknown = set(run_ids) - available
        if unknown:
            raise ValueError(f"Unknown training runs: {sorted(unknown)}")

        total = 0
        sums = np.zeros(len(archive.channel_names), dtype=np.float64)
        sum_squares = np.zeros_like(sums)
        for run_id in run_ids:
            features = archive.load_run(run_id).features.astype(np.float64)
            total += features.shape[0]
            sums += features.sum(axis=0)
            sum_squares += np.square(features).sum(axis=0)
        mean = sums / total
        variance = np.maximum(sum_squares / total - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale == 0.0] = 1.0
        return cls(
            channel_names=archive.channel_names,
            mean=mean,
            scale=scale,
            fitted_run_ids=run_ids,
            n_samples=total,
            archive_sha256=archive.archive_sha256,
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.channel_names):
            raise ValueError(
                f"Expected [time, {len(self.channel_names)}], got {values.shape}"
            )
        return ((values - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "type": "per_channel_standard",
            "channel_names": list(self.channel_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fitted_run_ids": list(self.fitted_run_ids),
            "n_samples": self.n_samples,
            "archive_sha256": self.archive_sha256,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "ChannelStandardizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            channel_names=tuple(payload["channel_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            fitted_run_ids=tuple(payload["fitted_run_ids"]),
            n_samples=int(payload["n_samples"]),
            archive_sha256=str(payload["archive_sha256"]),
        )
