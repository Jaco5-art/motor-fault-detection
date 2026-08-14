from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from motor_fault.evaluation.events import (
    contiguous_intervals,
    event_counts_for_motor,
    greedy_nonoverlap_indices,
    maximum_overlap_matches,
    merge_intervals,
)


class EventMetricTests(unittest.TestCase):
    def test_contiguous_and_merged_intervals_are_inclusive(self) -> None:
        self.assertEqual(
            contiguous_intervals(np.array([0, 1, 1, 0, 1]), offset=10),
            [(11, 12), (14, 14)],
        )
        self.assertEqual(merge_intervals([(0, 4), (5, 7), (10, 12)]), [(0, 7), (10, 12)])

    def test_one_prediction_cannot_match_two_truth_events(self) -> None:
        truth = [(10, 12), (16, 18)]
        predicted = [(8, 20)]
        self.assertEqual(maximum_overlap_matches(truth, predicted), 1)

    def test_event_metrics_do_not_point_adjust_window_predictions(self) -> None:
        manifest = pd.DataFrame(
            {
                "run_id": ["r1", "r1", "r1"],
                "start_idx": [0, 5, 10],
                "end_idx": [9, 14, 19],
            }
        )
        labels = np.zeros((20, 6), dtype=np.int8)
        labels[7:9, 0] = 1
        labels[16:18, 0] = 1
        result = event_counts_for_motor(
            manifest, {"r1": labels}, np.array([1, 0, 0]), motor_index=0
        )
        self.assertEqual(result["true_events"], 2)
        self.assertEqual(result["matched_events"], 1)
        self.assertEqual(result["missed_events"], 1)
        self.assertEqual(result["event_recall"], 0.5)

    def test_greedy_nonoverlap_is_run_local(self) -> None:
        manifest = pd.DataFrame(
            {
                "run_id": ["a", "a", "a", "b"],
                "start_idx": [0, 5, 10, 0],
                "end_idx": [9, 14, 19, 9],
            }
        )
        self.assertEqual(greedy_nonoverlap_indices(manifest).tolist(), [0, 2, 3])


if __name__ == "__main__":
    unittest.main()
