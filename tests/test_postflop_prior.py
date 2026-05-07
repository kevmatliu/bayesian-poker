"""Tests for post-flop baseline policy and theta tilting."""

from __future__ import annotations

import unittest

import numpy as np

from utils.prior.postflop import (
    CALL,
    FOLD,
    RAISE,
    PostflopFeatures,
    PostflopPrior,
    feature_vector,
    legal_actions,
    train_baseline_facing_bet,
    train_baseline_no_bet,
)
from utils.em import PostflopThetaObservation, single_hand_em_gradient_sample


def _feat_facing(m: float = 0.45, d: float = 0.35) -> PostflopFeatures:
    return PostflopFeatures(
        made=m,
        draw=d,
        bet_frac_pot=0.5,
        pot_odds=0.25,
        in_position=True,
        multiway=False,
        spr=8.0,
        street="flop",
        board_wetness=0.4,
        facing_bet=True,
    )


def _feat_no_bet(m: float = 0.5, d: float = 0.4) -> PostflopFeatures:
    return PostflopFeatures(
        made=m,
        draw=d,
        bet_frac_pot=0.0,
        pot_odds=0.0,
        in_position=False,
        multiway=True,
        spr=12.0,
        street="turn",
        board_wetness=0.55,
        facing_bet=False,
    )


class TestPostflopPrior(unittest.TestCase):
    def test_legal_actions_facing(self) -> None:
        f = _feat_facing()
        self.assertEqual(legal_actions(f), (FOLD, CALL, RAISE))

    def test_legal_actions_no_bet(self) -> None:
        f = _feat_no_bet()
        self.assertEqual(legal_actions(f), (CALL, RAISE))

    def test_base_probs_sum_to_one_facing(self) -> None:
        p = PostflopPrior(floor=0.0)
        probs = p.base_probs(_feat_facing())
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertEqual(set(probs.keys()), {FOLD, CALL, RAISE})

    def test_base_probs_sum_to_one_no_bet(self) -> None:
        p = PostflopPrior(floor=0.0)
        probs = p.base_probs(_feat_no_bet())
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertEqual(set(probs.keys()), {CALL, RAISE})

    def test_action_probs_sum_to_one(self) -> None:
        p = PostflopPrior(theta_post=(0.2, -0.1, 0.05), floor=0.0)
        probs = p.action_probs(_feat_facing())
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_theta_zero_matches_base(self) -> None:
        p = PostflopPrior(theta_post=(0.0, 0.0, 0.0), floor=0.0)
        f = _feat_facing()
        base = p.base_probs(f)
        tilted = p.action_probs(f)
        for a in base:
            self.assertAlmostEqual(tilted[a], base[a], places=6)

    def test_fold_theta_increases_fold_when_facing(self) -> None:
        f = _feat_facing(m=0.35, d=0.25)
        p0 = PostflopPrior(theta_post=(0.0, 0.0, 0.0), floor=0.0)
        p1 = PostflopPrior(theta_post=(0.4, 0.0, 0.0), floor=0.0)
        self.assertGreater(p1.action_probs(f)[FOLD], p0.action_probs(f)[FOLD])

    def test_call_theta_increases_call(self) -> None:
        f = _feat_facing(m=0.55, d=0.3)
        p0 = PostflopPrior(theta_post=(0.0, 0.0, 0.0), floor=0.0)
        p1 = PostflopPrior(theta_post=(0.0, 0.35, 0.0), floor=0.0)
        self.assertGreater(p1.action_probs(f)[CALL], p0.action_probs(f)[CALL])

    def test_raise_theta_increases_raise(self) -> None:
        f = _feat_no_bet(m=0.6, d=0.35)
        p0 = PostflopPrior(theta_post=(0.0, 0.0, 0.0), floor=0.0)
        p1 = PostflopPrior(theta_post=(0.0, 0.0, 0.45), floor=0.0)
        self.assertGreater(p1.action_probs(f)[RAISE], p0.action_probs(f)[RAISE])

    def test_no_bet_fold_probability_zero(self) -> None:
        p = PostflopPrior(floor=0.0)
        f = _feat_no_bet()
        self.assertEqual(p.action_probability(f, FOLD), 0.0)
        self.assertNotIn(FOLD, p.action_probs(f))

    def test_em_gradient_shape_finite(self) -> None:
        feat = _feat_facing()
        obs = PostflopThetaObservation(
            combo_key="AsKs",
            log_prior_range=-5.0,
            decisions=((feat, CALL),),
        )
        g = single_hand_em_gradient_sample([obs], prior_floor=0.0)
        self.assertEqual(g.shape, (3,))
        self.assertTrue(np.all(np.isfinite(g)))

    def test_phi_dimension(self) -> None:
        v = feature_vector(_feat_facing())
        self.assertEqual(v.shape, (13,))

    def test_train_baselines_run(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 13))
        y_f = rng.integers(0, 3, size=40)
        bf = train_baseline_facing_bet(X, y_f, learning_rate=0.2, max_epochs=100)
        self.assertEqual(bf.shape, (3, 13))

        y_nb = rng.choice([CALL, RAISE], size=30)
        bn = train_baseline_no_bet(X[:30], y_nb, learning_rate=0.2, max_epochs=100)
        self.assertEqual(bn.shape, (2, 13))


if __name__ == "__main__":
    unittest.main()
