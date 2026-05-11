"""Tests for post-flop baseline policy and theta tilting."""

from __future__ import annotations

import unittest

import numpy as np

from utils.prior.postflop import (
    CALL,
    FOLD,
    PHI_DIM,
    RAISE,
    PostflopFeatures,
    PostflopPrior,
    feature_vector,
    legal_actions,
    train_baseline_facing_bet,
    train_baseline_no_bet,
)
from utils.em import (
    PostflopEMHandBundle,
    PostflopEMTimestep,
    PostflopThetaObservation,
    e_step_postflop_bundle,
    single_hand_em_gradient_sample,
)


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

    def test_em_bundle_e_step_posterior(self) -> None:
        f = _feat_facing()
        bundle = PostflopEMHandBundle(
            decisions=(
                PostflopEMTimestep(
                    action=CALL,
                    features_by_combo=(("AhKh", f), ("QsJs", f)),
                ),
            ),
            initial_combo_range={"AhKh": 0.5, "QsJs": 0.5},
        )
        prior = PostflopPrior(theta_post=(0.0, 0.0, 0.0), floor=0.0)
        q = e_step_postflop_bundle(bundle, prior)
        self.assertAlmostEqual(sum(q.values()), 1.0, places=6)
        self.assertAlmostEqual(q["AhKh"], 0.5, places=5)
        self.assertAlmostEqual(q["QsJs"], 0.5, places=5)

    def test_phi_dimension(self) -> None:
        v = feature_vector(_feat_facing())
        self.assertEqual(v.shape, (PHI_DIM,))

    def test_train_baselines_run(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, PHI_DIM))
        y_f = rng.integers(0, 3, size=40)
        bf = train_baseline_facing_bet(X, y_f, learning_rate=0.2, max_epochs=100)
        self.assertEqual(bf.shape, (3, PHI_DIM))

        y_nb = rng.choice([CALL, RAISE], size=30)
        bn = train_baseline_no_bet(X[:30], y_nb, learning_rate=0.2, max_epochs=100)
        self.assertEqual(bn.shape, (2, PHI_DIM))


class TestPostflopPriorBatchAPI(unittest.TestCase):
    """The vectorized matrix API must match the per-row API exactly (Method D)."""

    def _build_random_features(self, n: int = 32, seed: int = 0) -> list:
        rng = np.random.default_rng(seed)
        streets = ("flop", "turn", "river")
        feats = []
        for i in range(n):
            facing = bool(i % 3)
            feats.append(
                PostflopFeatures(
                    made=float(rng.random()),
                    draw=float(rng.random()),
                    bet_frac_pot=float(rng.random()) if facing else 0.0,
                    pot_odds=float(rng.random() * 0.5) if facing else 0.0,
                    in_position=bool(rng.integers(0, 2)),
                    multiway=bool(rng.integers(0, 2)),
                    spr=float(rng.random() * 30),
                    street=streets[int(rng.integers(0, 3))],
                    board_wetness=float(rng.random()),
                    facing_bet=facing,
                    rich=(rng.random(16) > 0.5).astype(float),
                    equity=(float(rng.random()) if rng.random() < 0.5 else -1.0),
                )
            )
        return feats

    def test_action_probs_matrix_matches_per_row(self) -> None:
        from utils.prior.postflop import feature_vector

        prior = PostflopPrior(theta_post=(0.2, -0.1, 0.3), floor=1e-4)
        feats = self._build_random_features(n=64, seed=1)
        phi = np.stack([feature_vector(f) for f in feats], axis=0)
        facing = np.array([f.facing_bet for f in feats], dtype=bool)

        batch = prior.action_probs_matrix(phi, facing)
        max_diff = 0.0
        for i, f in enumerate(feats):
            row = prior.action_probs(f)
            ref = np.array([row.get(0, 0.0), row.get(1, 0.0), row.get(2, 0.0)])
            max_diff = max(max_diff, float(np.max(np.abs(ref - batch[i]))))
        self.assertLess(max_diff, 1e-12)

    def test_action_probs_matrix_floor_zero(self) -> None:
        from utils.prior.postflop import feature_vector

        prior = PostflopPrior(theta_post=(0.0, 0.0, 0.0), floor=0.0)
        feats = self._build_random_features(n=32, seed=2)
        phi = np.stack([feature_vector(f) for f in feats], axis=0)
        facing = np.array([f.facing_bet for f in feats], dtype=bool)
        batch = prior.action_probs_matrix(phi, facing)
        for i, f in enumerate(feats):
            row = prior.action_probs(f)
            ref = np.array([row.get(0, 0.0), row.get(1, 0.0), row.get(2, 0.0)])
            self.assertTrue(np.allclose(batch[i], ref, atol=1e-12))
        # FOLD column must be exactly zero on no-bet rows.
        no_bet_idx = ~facing
        self.assertTrue(np.all(batch[no_bet_idx, 0] == 0.0))

    def test_legacy_beta_shape_auto_padded(self) -> None:
        """A 13-column beta loaded from disk should pad up to PHI_DIM."""
        legacy_facing = np.random.default_rng(0).normal(size=(3, 13))
        legacy_no_bet = np.random.default_rng(1).normal(size=(2, 13))
        prior = PostflopPrior(
            beta_facing=legacy_facing,
            beta_no_bet=legacy_no_bet,
        )
        self.assertEqual(prior.beta_facing_matrix.shape, (3, PHI_DIM))
        self.assertEqual(prior.beta_no_bet_matrix.shape, (2, PHI_DIM))
        # The first 13 columns are preserved verbatim.
        np.testing.assert_allclose(
            prior.beta_facing_matrix[:, :13], legacy_facing
        )
        np.testing.assert_allclose(
            prior.beta_facing_matrix[:, 13:], 0.0
        )


if __name__ == "__main__":
    unittest.main()
