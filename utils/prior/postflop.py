"""Population postflop baseline: facing / no-bet ``beta`` matrices only (no tendency theta)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from utils.action.common import ActionPhase
from utils.action.postflop_context import (
    CALL,
    FOLD,
    HEURISTIC_BETA_FACING,
    HEURISTIC_BETA_NO_BET,
    PHI_DIM,
    RAISE,
    PostflopFeatures,
    _coerce_beta_to_phi_dim,
    _floor_row_probs,
    _row_softmax,
    _softmax_dict,
    feature_vector,
    legal_actions,
    maybe_floor_action_probs,
)


@dataclass
class PostflopPrior:
    """Frozen ``beta_facing`` / ``beta_no_bet``; per-row probs are baseline softmax only."""

    floor: float = 1e-6
    beta_facing: Optional[np.ndarray] = None
    beta_no_bet: Optional[np.ndarray] = None
    _beta_facing_store: np.ndarray = field(init=False, repr=False)
    _beta_no_bet_store: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bf = self.beta_facing if self.beta_facing is not None else HEURISTIC_BETA_FACING.copy()
        bn = self.beta_no_bet if self.beta_no_bet is not None else HEURISTIC_BETA_NO_BET.copy()
        bf = _coerce_beta_to_phi_dim(bf, n_rows=3, name="beta_facing")
        bn = _coerce_beta_to_phi_dim(bn, n_rows=2, name="beta_no_bet")
        object.__setattr__(self, "_beta_facing_store", bf)
        object.__setattr__(self, "_beta_no_bet_store", bn)

    @property
    def mode(self) -> ActionPhase:
        return ActionPhase.POSTFLOP

    @property
    def beta_facing_matrix(self) -> np.ndarray:
        return np.asarray(self._beta_facing_store, dtype=float).copy()

    @property
    def beta_no_bet_matrix(self) -> np.ndarray:
        return np.asarray(self._beta_no_bet_store, dtype=float).copy()

    def feature_row(self, features: PostflopFeatures) -> np.ndarray:
        return feature_vector(features)

    def legal_actions(self, features: PostflopFeatures) -> Tuple[int, ...]:
        return legal_actions(features)

    def base_scores(self, features: PostflopFeatures) -> Dict[int, float]:
        phi = self.feature_row(features)
        if features.facing_bet:
            beta = self._beta_facing_store
            return {
                FOLD: float(beta[0] @ phi),
                CALL: float(beta[1] @ phi),
                RAISE: float(beta[2] @ phi),
            }
        beta = self._beta_no_bet_store
        return {
            CALL: float(beta[0] @ phi),
            RAISE: float(beta[1] @ phi),
        }

    def base_probs(self, features: PostflopFeatures) -> Dict[int, float]:
        scores = self.base_scores(features)
        legal = self.legal_actions(features)
        sub = {a: scores[a] for a in legal}
        probs = _softmax_dict(sub)
        return maybe_floor_action_probs(probs, self.floor)

    @staticmethod
    def behavior_vector(action: int) -> np.ndarray:
        if action == FOLD:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if action == CALL:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        if action == RAISE:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        raise ValueError(f"unknown action {action}")

    def action_utility_vectors(self, features: PostflopFeatures) -> Dict[int, np.ndarray]:
        p_base = self.base_probs(features)
        legal = self.legal_actions(features)
        expected = np.zeros(3, dtype=float)
        for a in legal:
            expected += float(p_base[a]) * self.behavior_vector(a)
        return {a: self.behavior_vector(a) - expected for a in legal}

    def baseline_probs_matrix(self, phi: np.ndarray, facing: np.ndarray) -> np.ndarray:
        """Vectorized baseline softmax probabilities ``(N, 3)`` (no tendency tilt)."""
        phi = np.asarray(phi, dtype=float)
        facing = np.asarray(facing, dtype=bool)
        if phi.ndim != 2 or phi.shape[1] != PHI_DIM:
            raise ValueError(f"phi must have shape (N, {PHI_DIM}), got {phi.shape}")
        if facing.shape != (phi.shape[0],):
            raise ValueError(
                f"facing must be a 1-D bool array of length {phi.shape[0]}, got shape {facing.shape}"
            )

        beta_f = self._beta_facing_store
        beta_n = self._beta_no_bet_store
        scores_f = phi @ beta_f.T
        scores_n = phi @ beta_n.T
        base_facing = _row_softmax(scores_f)
        base_no_bet = _row_softmax(scores_n)

        probs = np.zeros((phi.shape[0], 3), dtype=float)
        probs[facing] = base_facing[facing]
        no_bet_idx = ~facing
        if no_bet_idx.any():
            probs[no_bet_idx, FOLD] = 0.0
            probs[no_bet_idx, CALL] = base_no_bet[no_bet_idx, 0]
            probs[no_bet_idx, RAISE] = base_no_bet[no_bet_idx, 1]

        if self.floor > 0.0:
            probs = _floor_row_probs(probs, self.floor, facing)
        return probs


def fit_heuristic_postflop_prior() -> PostflopPrior:
    """Return a :class:`PostflopPrior` with default heuristic ``beta`` matrices."""
    return PostflopPrior()
