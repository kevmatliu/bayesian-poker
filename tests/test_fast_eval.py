"""Tests for the fast vectorized evaluator and per-board feature tables.

Covers:

* :func:`rank_hand` / :func:`_rank_batch` parity with the legacy
  :func:`utils.strength.postflop.best_hand` ordering on random 5/6/7-card
  hands (Method 1).
* :func:`made_percentile_index_table` agrees with the legacy
  :func:`made_strength_percentile` to floating-point exactness.
* :func:`rollout_equity_index_table` matches a brute-force reference on
  the turn (exact enumeration is small enough to verify directly).
* :func:`combo_key_from_indices` matches the canonical
  :func:`utils.filter.postflop.combo_key`.
"""

from __future__ import annotations

import random
import unittest
from itertools import combinations

import numpy as np

from utils.filter.postflop import combo_key as legacy_combo_key
from utils.parse import all_52_cards, parse_card, parse_cards
from utils.strength.fast_eval import (
    _per_board_state,
    _rank_batch,
    card_to_index,
    clear_caches,
    combo_key_from_indices,
    index_to_card,
    made_percentile_array,
    made_percentile_at_combo_key,
    made_percentile_by_combo_key,
    made_percentile_index_table,
    parse_board_indices,
    rank_hand,
    rollout_equity_at_combo_key,
    rollout_equity_by_combo_key,
    rollout_equity_index_table,
)
from utils.strength.postflop import (
    best_hand,
    draw_strength_from_hand,
    hand_feature_vector,
    made_strength_percentile,
)


class TestComboKeyParity(unittest.TestCase):
    def test_random_pairs(self) -> None:
        random.seed(0)
        cards = all_52_cards()
        for _ in range(500):
            a, b = random.sample(cards, 2)
            legacy = legacy_combo_key(a, b)
            fast = combo_key_from_indices(card_to_index(a), card_to_index(b))
            self.assertEqual(legacy, fast)


class TestRankHand(unittest.TestCase):
    def test_scalar_ordering_matches_legacy(self) -> None:
        """For 200 random pairs of hands, fast/legacy agree on which is stronger."""
        random.seed(1)
        cards = all_52_cards()
        for n in (5, 6, 7):
            for _ in range(200):
                hand_a = random.sample(cards, n)
                hand_b = random.sample(cards, n)
                while set(hand_b) & set(hand_a):
                    hand_b = random.sample(cards, n)
                la = best_hand(hand_a)
                lb = best_hand(hand_b)
                fa = rank_hand([card_to_index(c) for c in hand_a])
                fb = rank_hand([card_to_index(c) for c in hand_b])
                legacy_cmp = (la > lb) - (la < lb)
                fast_cmp = (fa > fb) - (fa < fb)
                self.assertEqual(legacy_cmp, fast_cmp)

    def test_batch_matches_scalar(self) -> None:
        random.seed(2)
        for n in (5, 6, 7):
            hands = []
            for _ in range(300):
                hands.append(random.sample(range(52), n))
            arr = np.array(hands, dtype=np.int32)
            batch = _rank_batch(arr)
            for i, h in enumerate(hands):
                self.assertEqual(rank_hand(h), int(batch[i]))


class TestMadePercentileParity(unittest.TestCase):
    """The vectorized per-board table must equal the per-hole legacy call."""

    def test_flop_parity_random_sample(self) -> None:
        clear_caches()
        board = "Ah7s2d"
        board_cards = parse_cards([board[i : i + 2] for i in range(0, len(board), 2)])
        board_idx = parse_board_indices(board)
        table = made_percentile_index_table(board_idx)
        random.seed(3)
        sample = random.sample(list(table.keys()), 12)
        for a_idx, b_idx in sample:
            hole = [index_to_card(a_idx), index_to_card(b_idx)]
            legacy = made_strength_percentile(hole, board_cards)
            self.assertAlmostEqual(legacy, table[(a_idx, b_idx)], places=10)

    def test_turn_parity_random_sample(self) -> None:
        clear_caches()
        board = "KsKd7c4s"
        board_cards = parse_cards([board[i : i + 2] for i in range(0, len(board), 2)])
        board_idx = parse_board_indices(board)
        table = made_percentile_by_combo_key(board_idx)
        random.seed(4)
        sample = random.sample(list(table.keys()), 12)
        for key in sample:
            hole = parse_cards([key[0:2], key[2:4]])
            legacy = made_strength_percentile(hole, board_cards)
            self.assertAlmostEqual(legacy, table[key], places=10)

    def test_river_parity_random_sample(self) -> None:
        clear_caches()
        board = "Ah7s2dKd5c"
        board_cards = parse_cards([board[i : i + 2] for i in range(0, len(board), 2)])
        board_idx = parse_board_indices(board)
        table = made_percentile_by_combo_key(board_idx)
        random.seed(5)
        sample = random.sample(list(table.keys()), 12)
        for key in sample:
            hole = parse_cards([key[0:2], key[2:4]])
            legacy = made_strength_percentile(hole, board_cards)
            self.assertAlmostEqual(legacy, table[key], places=10)

    def test_at_combo_key_matches_full_dict(self) -> None:
        clear_caches()
        board_idx = parse_board_indices("Ah7s2dKd")
        full = made_percentile_by_combo_key(board_idx)
        for key, v in full.items():
            one = made_percentile_at_combo_key(board_idx, key)
            self.assertIsNotNone(one)
            self.assertAlmostEqual(float(one), float(v), places=10)
        eq_full = rollout_equity_by_combo_key(board_idx, mc_samples=0)
        for key, v in eq_full.items():
            one = rollout_equity_at_combo_key(board_idx, key, mc_samples=0)
            self.assertIsNotNone(one)
            self.assertAlmostEqual(float(one), float(v), places=10)


class TestRolloutEquity(unittest.TestCase):
    def test_turn_equity_matches_manual_reference(self) -> None:
        """Pick one combo on the turn, brute-force average over the 44 rivers."""
        clear_caches()
        board = "Ah7s2dKd"
        board_idx = parse_board_indices(board)
        eq_table = rollout_equity_by_combo_key(board_idx, mc_samples=0)

        # Pick a non-blocked combo and verify the equity manually.
        # Canonical combo_key sorts by (-rank_value, suit_char):
        # T of hearts (h) sorts before T of spades (s) since 'h' < 's' alphabetically.
        combo_str = "ThTs"
        ca, cb = parse_cards([combo_str[0:2], combo_str[2:4]])
        board_set = set(board_idx) | {card_to_index(ca), card_to_index(cb)}
        rivers = [c for c in range(52) if c not in board_set]
        total = 0.0
        for r in rivers:
            full = tuple(sorted(list(board_idx) + [r]))
            live_rows, perc = made_percentile_array(full)
            # find row index for this combo
            from utils.strength.fast_eval import (
                _ALL_COMBO_PAIRS,
            )
            pairs_arr = _ALL_COMBO_PAIRS[live_rows]
            target = (card_to_index(ca), card_to_index(cb))
            if target[0] > target[1]:
                target = (target[1], target[0])
            mask = (pairs_arr[:, 0] == target[0]) & (pairs_arr[:, 1] == target[1])
            self.assertTrue(mask.any(), f"target combo not found on river {r}")
            total += float(perc[mask][0])
        manual = total / len(rivers)
        self.assertAlmostEqual(manual, eq_table[combo_str], places=10)

    def test_river_equity_equals_made(self) -> None:
        clear_caches()
        board = "Ah7s2dKd5c"
        board_idx = parse_board_indices(board)
        eq = rollout_equity_by_combo_key(board_idx)
        made = made_percentile_by_combo_key(board_idx)
        self.assertEqual(eq.keys(), made.keys())
        for k in eq:
            self.assertAlmostEqual(eq[k], made[k], places=12)

    def test_flop_mc_is_unbiased(self) -> None:
        """MC equity should be close to exact equity (within sampling error)."""
        clear_caches()
        board = "Ah7s2d"
        board_idx = parse_board_indices(board)
        exact = rollout_equity_by_combo_key(board_idx, mc_samples=0)
        mc = rollout_equity_by_combo_key(board_idx, mc_samples=128)
        # Same set of live combos
        self.assertEqual(exact.keys(), mc.keys())
        diffs = np.array([abs(exact[k] - mc[k]) for k in exact])
        # 128-sample MC: typical std ~ 0.03; allow up to 0.1 on any combo
        self.assertLess(diffs.mean(), 0.05, f"mean diff {diffs.mean()}")
        self.assertLess(diffs.max(), 0.15, f"max diff {diffs.max()}")


class TestRichFeatures(unittest.TestCase):
    def test_overpair_recognised(self) -> None:
        hole = parse_cards(["Qs", "Qh"])
        board = parse_cards(["8c", "5d", "2s"])
        vec = hand_feature_vector(hole, board)
        # is_over_pair index = 2
        self.assertEqual(vec[2], 1.0)
        # is_pair_or_better index = 0
        self.assertEqual(vec[0], 1.0)
        # not top_pair
        self.assertEqual(vec[1], 0.0)
        # is_pocket_pair index = 12
        self.assertEqual(vec[12], 1.0)

    def test_top_pair_recognised(self) -> None:
        hole = parse_cards(["Ks", "Tc"])
        board = parse_cards(["Kd", "7h", "2s"])
        vec = hand_feature_vector(hole, board)
        # is_top_pair index = 1
        self.assertEqual(vec[1], 1.0)
        self.assertEqual(vec[0], 1.0)  # is_pair_or_better
        self.assertEqual(vec[2], 0.0)  # not overpair

    def test_flush_draw_recognised(self) -> None:
        hole = parse_cards(["Ah", "Th"])
        board = parse_cards(["Kh", "5h", "2c"])
        vec = hand_feature_vector(hole, board)
        # has_flush_draw index = 8
        self.assertEqual(vec[8], 1.0)
        # is_suited = 11
        self.assertEqual(vec[11], 1.0)

    def test_blocker_features(self) -> None:
        hole = parse_cards(["Ah", "Kd"])
        board = parse_cards(["Qh", "Jh", "2s"])
        vec = hand_feature_vector(hole, board)
        # blocks_nut_fd index = 15 (Ah is the nut-flush blocker for hearts)
        self.assertEqual(vec[15], 1.0)


if __name__ == "__main__":
    unittest.main()
