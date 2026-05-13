"""Tests for range calibration aggregation helpers."""

import unittest

import numpy as np
import pandas as pd

from utils.eval.range_evaluation_helpers import (
    aggregate_range_calibration_by_hand,
    aggregate_range_calibration_row_stats,
)


class TestRangeCalibrationAggregate(unittest.TestCase):
    def test_row_stats_means_and_counts(self) -> None:
        df = pd.DataFrame(
            {
                "street": ["pre-flop", "flop", "flop"],
                "brier": [0.2, np.nan, 0.4],
                "combo_nll": [1.0, 2.0, 3.0],
            }
        )
        s = aggregate_range_calibration_row_stats(df)
        self.assertEqual(s["n_rows"], 3)
        self.assertEqual(s["n_by_street"]["pre-flop"], 1)
        self.assertEqual(s["n_by_street"]["flop"], 2)
        self.assertAlmostEqual(s["mean_brier"], 0.3)
        self.assertEqual(s["n_finite_brier"], 2)
        self.assertAlmostEqual(s["mean_combo_nll"], 2.0)
        self.assertEqual(s["n_finite_combo_nll"], 3)

    def test_by_hand_two_stage_mean(self) -> None:
        df = pd.DataFrame(
            {
                "session": ["a", "a", "b"],
                "hand_number": [1, 1, 1],
                "observer": ["O", "O", "O"],
                "target": ["T", "T", "T"],
                "brier": [0.0, 1.0, 1.0],
            }
        )
        h = aggregate_range_calibration_by_hand(df)
        self.assertEqual(h["n_hands"], 2)
        self.assertAlmostEqual(h["mean_across_hands_of_mean_brier_per_hand"], 0.75)


if __name__ == "__main__":
    unittest.main()
