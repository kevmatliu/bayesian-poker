"""Tests for the post-flop ``ComboRangeFilter`` (1,326-combo Bayesian filter)."""

from __future__ import annotations

import math
import unittest
from itertools import combinations

from utils.filter import ComboRangeFilter, all_combo_keys, combo_key, parse_combo_key
from utils.filter.helpers import initial_class_prior
from utils.filter.postflop import _coerce_cards
from utils.prior.postflop import CALL, FOLD, PostflopFeatures, PostflopPrior, RAISE
from utils.strength.common import all_52_cards, parse_card
from utils.strength.preflop import all_169_classes


def _features_combo(
    *,
    made: float = 0.5,
    draw: float = 0.3,
    facing_bet: bool = True,
) -> PostflopFeatures:
    return PostflopFeatures(
        made=made,
        draw=draw,
        bet_frac_pot=0.4 if facing_bet else 0.0,
        pot_odds=0.22 if facing_bet else 0.0,
        in_position=True,
        multiway=False,
        spr=8.0,
        street="flop",
        board_wetness=0.45,
        facing_bet=facing_bet,
    )


class TestExplodePreflopToCombos(unittest.TestCase):
    def test_uniform_169_explodes_to_uniform_1326(self) -> None:
        uniform = initial_class_prior()
        combos = ComboRangeFilter.explode_preflop_to_combos(uniform, "", "")
        self.assertEqual(len(combos), 1326)
        self.assertAlmostEqual(sum(combos.values()), 1.0, places=10)
        # Uniform-over-combos prior puts equal mass on every combo.
        for prob in combos.values():
            self.assertAlmostEqual(prob, 1.0 / 1326, places=10)

    def test_class_mass_distributed_uniformly_across_valid_combos(self) -> None:
        # Concentrate all mass on AKs; only suited Ace-King combos should be alive.
        rng = {cls: 0.0 for cls in all_169_classes()}
        rng["AKs"] = 1.0
        combos = ComboRangeFilter.explode_preflop_to_combos(rng, "", "")
        # 4 suits => 4 valid suited combos
        nonzero = {k: v for k, v in combos.items() if v > 0}
        self.assertEqual(len(nonzero), 4)
        for prob in nonzero.values():
            self.assertAlmostEqual(prob, 0.25, places=10)
        for combo in nonzero:
            ca, cb = parse_combo_key(combo)
            self.assertEqual({ca.rank, cb.rank}, {"A", "K"})
            self.assertEqual(ca.suit, cb.suit)

    def test_observer_blocks_conflicting_combos(self) -> None:
        # Observer holds AhKh – any AK combo containing the Ace or King of hearts is impossible.
        rng = {cls: 0.0 for cls in all_169_classes()}
        rng["AKs"] = 1.0
        combos = ComboRangeFilter.explode_preflop_to_combos(rng, "AhKh", "")
        # Suited hearts requires both Ah & Kh which are dead -> only spades / diamonds / clubs.
        nonzero = {k: v for k, v in combos.items() if v > 0}
        self.assertEqual(len(nonzero), 3)
        self.assertNotIn(combo_key(parse_card("Ah"), parse_card("Kh")), nonzero)
        # Equal share over 3 surviving combos.
        for p in nonzero.values():
            self.assertAlmostEqual(p, 1.0 / 3.0, places=10)

    def test_board_zeroes_blocked_combos(self) -> None:
        rng = {cls: 0.0 for cls in all_169_classes()}
        rng["AA"] = 1.0
        # Ace of spades is on the board: only AhAd / AhAc / AdAc remain.
        combos = ComboRangeFilter.explode_preflop_to_combos(rng, "", "As2c3d")
        nonzero = {k: v for k, v in combos.items() if v > 0}
        self.assertEqual(len(nonzero), 3)
        for combo in nonzero:
            ca, cb = parse_combo_key(combo)
            self.assertEqual(ca.rank, "A")
            self.assertEqual(cb.rank, "A")
            self.assertNotIn("S", {ca.suit, cb.suit})
        for p in nonzero.values():
            self.assertAlmostEqual(p, 1.0 / 3.0, places=10)

    def test_total_probability_preserved_across_explode(self) -> None:
        # Two classes with arbitrary masses should yield a distribution summing to 1.
        rng = {cls: 0.0 for cls in all_169_classes()}
        rng["AA"] = 0.3
        rng["KK"] = 0.7
        combos = ComboRangeFilter.explode_preflop_to_combos(rng, "", "")
        self.assertAlmostEqual(sum(combos.values()), 1.0, places=10)
        aa = [v for k, v in combos.items() if parse_combo_key(k)[0].rank == "A" and parse_combo_key(k)[1].rank == "A"]
        kk = [v for k, v in combos.items() if parse_combo_key(k)[0].rank == "K" and parse_combo_key(k)[1].rank == "K"]
        self.assertEqual(len(aa), 6)
        self.assertEqual(len(kk), 6)
        self.assertAlmostEqual(sum(aa), 0.3, places=10)
        self.assertAlmostEqual(sum(kk), 0.7, places=10)


class TestNarrowComboDistribution(unittest.TestCase):
    def test_turn_card_removes_conflicting_mass(self) -> None:
        rng = {cls: 0.0 for cls in all_169_classes()}
        rng["AA"] = 1.0
        flop = ComboRangeFilter.explode_preflop_to_combos(rng, "", "As2c3d")
        self.assertEqual(len(flop), 3)
        turn = ComboRangeFilter.narrow_combo_distribution(
            flop,
            observer_hole_cards="",
            board_cards="As2c3dAd",
        )
        # Ad on board kills AhAd and AdAc; only AhAc remains among the three AA combos.
        self.assertEqual(len(turn), 1)
        self.assertAlmostEqual(sum(turn.values()), 1.0, places=10)
        for combo in turn:
            ca, cb = parse_combo_key(combo)
            self.assertNotIn(parse_card("Ad"), (ca, cb))


class TestComboRangeFilterUpdate(unittest.TestCase):
    def _make_filter(self) -> ComboRangeFilter:
        f = ComboRangeFilter(
            observer_name="hero",
            target_name="villain",
            observer_hole_cards="",
            board_cards="",
            prior_model=PostflopPrior(floor=0.0),
        )
        f.explode_from_preflop(initial_class_prior())
        return f

    def test_update_renormalizes_to_one(self) -> None:
        f = self._make_filter()
        # Pretend every combo has the same features for this action.
        feats = {combo: _features_combo() for combo in f.combos}
        f.update(CALL, feats, state_key="flop|0")
        self.assertAlmostEqual(sum(f.combos.values()), 1.0, places=10)

    def test_update_proportional_to_action_likelihood(self) -> None:
        # When all combos share identical features, posterior should equal prior.
        f = self._make_filter()
        feats = {combo: _features_combo() for combo in f.combos}
        prior_combos = dict(f.combos)
        f.update(CALL, feats)
        for c, p in f.combos.items():
            self.assertAlmostEqual(p, prior_combos[c], places=10)

    def test_update_concentrates_on_strong_combos(self) -> None:
        # Two-buckets feature mock: half the combos have a "strong made" feature
        # (high made), half have "air" (low made). After observing RAISE the
        # strong half should outweigh the weak half.
        f = self._make_filter()
        all_combos = list(f.combos.keys())
        strong_set = set(all_combos[: len(all_combos) // 2])

        feats = {
            combo: _features_combo(made=0.95, draw=0.05, facing_bet=False)
            if combo in strong_set
            else _features_combo(made=0.05, draw=0.05, facing_bet=False)
            for combo in all_combos
        }
        f.update(RAISE, feats)
        strong_mass = sum(p for c, p in f.combos.items() if c in strong_set)
        weak_mass = sum(p for c, p in f.combos.items() if c not in strong_set)
        self.assertGreater(strong_mass, weak_mass)
        self.assertAlmostEqual(strong_mass + weak_mass, 1.0, places=10)

    def test_update_logs_filter_step(self) -> None:
        f = self._make_filter()
        feats = {combo: _features_combo() for combo in f.combos}
        f.update(CALL, feats, state_key="flop|0")
        self.assertEqual(len(f.steps), 1)
        step = f.steps[0]
        self.assertEqual(step.layer, "combo")
        self.assertEqual(step.state_key, "flop|0")
        self.assertGreater(step.evidence, 0.0)

    def test_zero_evidence_raises(self) -> None:
        f = self._make_filter()
        # No combos have features available -> evidence is zero.
        with self.assertRaises(ValueError):
            f.update(CALL, feature_by_combo={})

    def test_log_likelihood_matches_evidence_sum(self) -> None:
        f = self._make_filter()
        feats = {combo: _features_combo() for combo in f.combos}
        f.update(CALL, feats)
        f.update(CALL, feats)
        expected = sum(math.log(s.evidence) for s in f.steps)
        self.assertAlmostEqual(f.log_likelihood(), expected, places=10)


class TestSetBoard(unittest.TestCase):
    def test_set_board_zeroes_new_blockers_and_renormalizes(self) -> None:
        f = ComboRangeFilter(
            observer_name="hero",
            target_name="villain",
            prior_model=PostflopPrior(floor=0.0),
        )
        f.explode_from_preflop(initial_class_prior(), board="2c3d4h")
        self.assertAlmostEqual(sum(f.combos.values()), 1.0, places=10)
        # Reveal the turn (5s); combos that contain 5s should disappear.
        f.set_board("2c3d4h5s")
        for combo in f.combos:
            ca, cb = parse_combo_key(combo)
            self.assertNotIn(parse_card("5s"), {ca, cb})
        self.assertAlmostEqual(sum(f.combos.values()), 1.0, places=10)

    def test_set_board_rejects_overlap_with_observer_cards(self) -> None:
        f = ComboRangeFilter(
            observer_name="hero",
            target_name="villain",
            observer_hole_cards="AhKh",
            prior_model=PostflopPrior(floor=0.0),
        )
        f.explode_from_preflop(initial_class_prior(), board="2c3d4h")
        with self.assertRaises(ValueError):
            f.set_board("2c3d4hAh")  # observer holds Ah, illegal new board card.


class TestClassMarginal(unittest.TestCase):
    def test_class_marginal_recovers_input_when_uniform(self) -> None:
        f = ComboRangeFilter(
            observer_name="hero",
            target_name="villain",
            prior_model=PostflopPrior(floor=0.0),
        )
        uniform = initial_class_prior()
        f.explode_from_preflop(uniform)
        marginal = f.class_marginal()
        # Uniform-over-combos induces uniform-over-classes weighted by combo counts,
        # which is exactly the original ``initial_class_prior`` distribution.
        for cls, prob in uniform.items():
            self.assertAlmostEqual(marginal[cls], prob, places=10)


class TestComboKeyHelpers(unittest.TestCase):
    def test_combo_key_canonical_ordering(self) -> None:
        a, b = parse_card("Ah"), parse_card("Kh")
        self.assertEqual(combo_key(a, b), combo_key(b, a))
        self.assertEqual(combo_key(a, b)[0:2], "Ah")  # higher rank first

    def test_all_combo_keys_count(self) -> None:
        keys = all_combo_keys()
        self.assertEqual(len(keys), 1326)
        self.assertEqual(len(set(keys)), 1326)


if __name__ == "__main__":
    unittest.main()
