"""Tests for evaluation / Brier utilities."""

import unittest

import numpy as np

from utils.eval.brier import (
    brier_postflop1326,
    brier_preflop169,
    brier_preflop_from_combo1326,
    collapse_combo_distribution_to_169,
    multiclass_brier,
)
from utils.filter import all_combo_keys, combo_key
from utils.strength.common import parse_card
from utils.strength.preflop import all_169_classes, get_equivalence_class


class TestMulticlassBrier(unittest.TestCase):
    def test_perfect_calibration(self) -> None:
        p = np.zeros(5)
        p[2] = 1.0
        self.assertAlmostEqual(multiclass_brier(p, 2), 0.0)

    def test_uniform_wrong_class(self) -> None:
        p = np.ones(4) / 4.0
        # one-hot at index 0; predicted uniform
        self.assertGreater(multiclass_brier(p, 0), 0.0)


class TestPreflop169From1326(unittest.TestCase):
    def test_collapse_one_combo(self) -> None:
        ah = parse_card("Ah")
        kh = parse_card("Kh")
        k = combo_key(ah, kh)
        collapsed = collapse_combo_distribution_to_169({k: 1.0})
        self.assertEqual(len(collapsed), 1)
        cls = next(iter(collapsed))
        self.assertEqual(cls, get_equivalence_class([ah, kh]))
        self.assertAlmostEqual(collapsed[cls], 1.0)

    def test_explode_then_collapse_brier_matches(self) -> None:
        from utils.filter import ComboRangeFilter

        pre = {"AKs": 1.0}
        exploded = ComboRangeFilter.explode_preflop_to_combos(pre, "", "")
        b1 = brier_preflop169(pre, "AKs")
        b2 = brier_preflop_from_combo1326(exploded, "AKs")
        self.assertAlmostEqual(b1, b2)


class TestBrier1326(unittest.TestCase):
    def test_combo_key_order(self) -> None:
        keys = all_combo_keys()
        k0 = keys[0]
        dist = {k0: 1.0}
        self.assertAlmostEqual(brier_postflop1326(dist, keys, k0), 0.0)


if __name__ == "__main__":
    unittest.main()
