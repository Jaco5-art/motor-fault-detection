from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from motor_fault.data.archive import KaggleRobotArchive


DEFAULT_STATISTICS = (
    "mean",
    "std",
    "range",
    "last_first_diff",
    "mean_abs_diff",
    "energy",
    "crest_factor",
    "smoothing_index",
)


def _window_statistics(window: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(window, dtype=np.float64)
    diff = np.diff(values, axis=0)
    rms = np.sqrt(np.mean(np.square(values), axis=0))
    mean_abs = np.mean(np.abs(values), axis=0)
    return {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0, ddof=0),
        "range": np.ptp(values, axis=0),
        "last_first_diff": values[-1] - values[0],
        "mean_abs_diff": np.mean(np.abs(diff), axis=0),
        "energy": np.sum(np.square(values), axis=0),
        "crest_factor": np.divide(
            np.max(np.abs(values), axis=0),
            rms,
            out=np.zeros_like(rms),
            where=rms > 0,
        ),
        "smoothing_index": np.divide(
            np.sum(np.abs(diff), axis=0),
            mean_abs,
            out=np.zeros_like(mean_abs),
            where=mean_abs > 0,
        ),
    }


def extract_handcrafted_features(
    archive: KaggleRobotArchive,
    manifest: pd.DataFrame,
    statistics: tuple[str, ...] | list[str] = DEFAULT_STATISTICS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    stats = tuple(statistics)
    unknown = set(stats) - set(DEFAULT_STATISTICS)
    if unknown:
        raise ValueError(f"Unknown handcrafted statistics: {sorted(unknown)}")
    feature_names = tuple(
        f"{channel}_{stat}" for channel in archive.channel_names for stat in stats
    )
    output = np.empty((len(manifest), len(feature_names)), dtype=np.float32)
    cached_run_id = None
    cached_features = None
    for row_number, row in enumerate(manifest.itertuples(index=False)):
        run_id = str(row.run_id)
        if run_id != cached_run_id:
            cached_features = archive.load_run(run_id).features
            cached_run_id = run_id
        start = int(row.start_idx)
        end = int(row.end_idx) + 1
        calculated = _window_statistics(cached_features[start:end])
        # Channel-major order matches the generated feature names.
        output[row_number] = np.stack([calculated[name] for name in stats], axis=1).reshape(-1)
    if not np.isfinite(output).all():
        raise ValueError("Handcrafted feature extraction produced NaN/Inf")
    return output, feature_names


@dataclass(frozen=True)
class FeatureStandardizer:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    fitted_split: str = "train"

    @classmethod
    def fit(
        cls, features: np.ndarray, feature_names: tuple[str, ...]
    ) -> "FeatureStandardizer":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(feature_names):
            raise ValueError("Feature matrix/name mismatch")
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        return cls(feature_names=feature_names, mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("Feature matrix has an unexpected shape")
        return ((values - self.mean) / self.scale).astype(np.float32)

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "train_only_standard",
            "fitted_split": self.fitted_split,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStandardizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            feature_names=tuple(payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            fitted_split=str(payload["fitted_split"]),
        )


def labels_from_manifest(manifest: pd.DataFrame) -> np.ndarray:
    return manifest[[f"m{motor}_label" for motor in range(1, 7)]].to_numpy(dtype=np.int8)
