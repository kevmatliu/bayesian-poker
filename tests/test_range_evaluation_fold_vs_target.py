"""Tests for observer fold vs target realized strength (range evaluation helper)."""

import unittest
from pathlib import Path

import pandas as pd

from utils.eval.range_evaluation_helpers import evaluate_observer_fold_vs_target_realized_strength


class TestFoldVsTargetStrength(unittest.TestCase):
    def test_missing_required_columns_raises(self) -> None:
        df = pd.DataFrame(
            [{"session": "x", "hand_number": 1, "street": "flop", "observer": "A", "target": "B"}]
        )
        with self.assertRaises(ValueError) as ctx:
            evaluate_observer_fold_vs_target_realized_strength(
                df, Path("pluribus"), require_target_alive_at_street_end=False
            )
        self.assertIn("community_cards", str(ctx.exception))

    def test_nonexistent_hand_returns_empty_summary(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "session": "nonexistent_session_xyz",
                    "hand_number": 1,
                    "street": "flop",
                    "observer": "A",
                    "target": "B",
                    "community_cards": "AhKdQc2s3h",
                    "target_still_in_hand": True,
                }
            ]
        )
        r = evaluate_observer_fold_vs_target_realized_strength(
            df, Path("pluribus"), require_target_alive_at_street_end=False
        )
        self.assertEqual(r["summary"]["n_eligible_observer_folds"], 0)
        self.assertIsNone(r.get("detail"))


if __name__ == "__main__":
    unittest.main()
