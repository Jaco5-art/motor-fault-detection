from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from motor_fault.data.handcrafted import _window_statistics
from motor_fault.evaluation.metrics import (
    evaluate_multimotor,
    select_f1_threshold,
    select_support_aware_thresholds,
)
from motor_fault.models.traditional import EmpiricalScoreCalibrator
from motor_fault.models.xgboost_baseline import per_motor_class_weights, stack_motor_targets


class HandcraftedFeatureTests(unittest.TestCase):
    def test_original_signal_statistics_are_finite_and_correct(self) -> None:
        window = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        stats = _window_statistics(window)
        np.testing.assert_allclose(stats["mean"], [2.0, 0.0])
        np.testing.assert_allclose(stats["range"], [2.0, 0.0])
        np.testing.assert_allclose(stats["last_first_diff"], [2.0, 0.0])
        np.testing.assert_allclose(stats["energy"], [14.0, 0.0])
        self.assertTrue(all(np.isfinite(values).all() for values in stats.values()))


class MetricTests(unittest.TestCase):
    def test_f1_threshold_is_selected_only_from_given_scores(self) -> None:
        y = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        result = select_f1_threshold(y, scores)
        self.assertEqual(result.f1, 1.0)
        self.assertEqual(result.threshold, 0.8)

    def test_missing_motor_positives_return_null_eligible_metric(self) -> None:
        y = np.zeros((3, 6), dtype=np.int8)
        y[:, 5] = [0, 1, 0]
        scores = y.astype(float)
        metrics = evaluate_multimotor(y, scores, 0.5)
        self.assertIsNone(metrics["per_motor"]["m1"]["pr_auc"])
        self.assertEqual(metrics["per_motor"]["m6"]["f1"], 1.0)

    def test_support_aware_threshold_uses_percentile_without_positives(self) -> None:
        y = np.zeros((4, 6), dtype=np.int8)
        y[:, 5] = [0, 0, 1, 1]
        scores = np.tile(np.asarray([0.1, 0.2, 0.8, 0.9])[:, None], (1, 6))
        thresholds, policies = select_support_aware_thresholds(y, scores)
        self.assertEqual(policies["m1"], "validation_normal_percentile_95")
        self.assertAlmostEqual(thresholds[0], 0.885)
        self.assertEqual(policies["m6"], "validation_f1_optimal")
        self.assertEqual(thresholds[5], 0.8)


class ModelUtilityTests(unittest.TestCase):
    def test_empirical_calibration_is_per_motor(self) -> None:
        train = np.column_stack([np.arange(4) + motor * 10 for motor in range(6)])
        calibrator = EmpiricalScoreCalibrator.fit(train)
        transformed = calibrator.transform(train)
        np.testing.assert_allclose(transformed[:, 0], [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(transformed[:, 5], [0.25, 0.5, 0.75, 1.0])

    def test_shared_xgboost_stack_and_train_only_weights(self) -> None:
        x = np.arange(12, dtype=np.float32).reshape(2, 6)
        y = np.asarray([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=np.int8)
        stacked_x, stacked_y, motors = stack_motor_targets(x, y)
        self.assertEqual(stacked_x.shape, (12, 12))
        self.assertEqual(stacked_y.shape, (12,))
        weights, summary = per_motor_class_weights(stacked_y, motors)
        np.testing.assert_allclose(weights, np.ones(12))
        self.assertEqual(summary["m1"], 1.0)


if __name__ == "__main__":
    unittest.main()
