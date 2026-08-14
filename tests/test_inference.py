from __future__ import annotations

import unittest

import numpy as np

from motor_fault.data.archive import RunData
from motor_fault.inference import build_window_spans, extract_run_features


class InferenceTests(unittest.TestCase):
    def test_window_spans_match_frozen_contract(self) -> None:
        spans = build_window_spans(450, window_length=200, stride=10)
        self.assertEqual(spans[0], (0, 199))
        self.assertEqual(spans[-1], (250, 449))
        self.assertEqual(len(spans), 26)
        self.assertEqual(build_window_spans(199), [])

    def test_run_feature_shape_matches_18_channels_times_8_statistics(self) -> None:
        rng = np.random.default_rng(7)
        run = RunData(
            run_id="synthetic",
            time=np.arange(210, dtype=np.float64),
            features=rng.normal(size=(210, 18)).astype(np.float32),
            labels=None,
            channel_names=tuple(f"m{motor}_{signal}" for motor in range(1, 7) for signal in ("position", "temperature", "voltage")),
            activity="synthetic",
            declared_failures=(),
        )
        features, names = extract_run_features(run, [(0, 199), (10, 209)])
        self.assertEqual(features.shape, (2, 144))
        self.assertEqual(len(names), 144)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
