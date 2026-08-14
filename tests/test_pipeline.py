from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from motor_fault.data.archive import RunData
from motor_fault.data.scaler import ChannelStandardizer
from motor_fault.data.split import SplitDefinition
from motor_fault.data.windowing import WindowSpec, build_run_window_manifest


def make_run(run_id: str, values: np.ndarray, labels: np.ndarray) -> RunData:
    channels = tuple(f"c{i}" for i in range(values.shape[1]))
    return RunData(
        run_id=run_id,
        time=np.arange(len(values), dtype=np.float64),
        features=values.astype(np.float32),
        labels=labels.astype(np.int8),
        channel_names=channels,
        activity="synthetic",
        declared_failures=(),
    )


class DummyArchive:
    def __init__(self, runs: dict[str, RunData]):
        self.runs = runs
        self.channel_names = next(iter(runs.values())).channel_names
        self.archive_sha256 = "synthetic"

    def run_ids(self, split: str = "training") -> list[str]:
        return sorted(self.runs)

    def load_run(self, run_id: str) -> RunData:
        return self.runs[run_id]


class WindowingTests(unittest.TestCase):
    def test_any_rule_keeps_early_fault_inside_long_window(self) -> None:
        x = np.zeros((8, 18), dtype=np.float32)
        y = np.zeros((8, 6), dtype=np.int8)
        y[1:3, 0] = 1
        run = make_run("r1", x, y)
        manifest = build_run_window_manifest(run, "test", WindowSpec(length=4, stride=4))
        self.assertEqual(int(manifest.iloc[0]["m1_label"]), 1)
        self.assertEqual(int(manifest.iloc[0]["m1_fault_points"]), 2)

    def test_endpoint_rule_does_not_use_any_point(self) -> None:
        x = np.zeros((8, 18), dtype=np.float32)
        y = np.zeros((8, 6), dtype=np.int8)
        y[1:3, 0] = 1
        run = make_run("r1", x, y)
        spec = WindowSpec(length=4, stride=4, label_rule="endpoint")
        manifest = build_run_window_manifest(run, "test", spec)
        self.assertEqual(int(manifest.iloc[0]["m1_label"]), 0)


class LeakageTests(unittest.TestCase):
    def test_split_overlap_is_rejected(self) -> None:
        split = SplitDefinition("bad", ("r1",), ("r1",), ("r2",), "shared")
        with self.assertRaisesRegex(ValueError, "Run leakage"):
            split.validate(["r1", "r2"])

    def test_scaler_fits_only_declared_training_run(self) -> None:
        y = np.zeros((2, 6), dtype=np.int8)
        train = make_run("train", np.vstack([np.zeros(18), np.full(18, 2)]), y)
        test = make_run("test", np.full((2, 18), 1000), y)
        archive = DummyArchive({"train": train, "test": test})
        scaler = ChannelStandardizer.fit(archive, ["train"])
        self.assertEqual(scaler.fitted_run_ids, ("train",))
        np.testing.assert_allclose(scaler.mean, np.ones(18))
        np.testing.assert_allclose(scaler.scale, np.ones(18))
        np.testing.assert_allclose(scaler.transform(train.features).mean(axis=0), np.zeros(18))


if __name__ == "__main__":
    unittest.main()
