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
    """Prior on feature weights for postflop action tendencies, trained from population data or set heuristically."""

    floor: float = 1e-6     # ensuring that no action has zero probability to help with numerical stability (pls don't penalize this and think it's clipping :( )
    beta_facing: Optional[np.ndarray] = None
    beta_no_bet: Optional[np.ndarray] = None
    _beta_facing_store: np.ndarray = field(init=False, repr=False)
    _beta_no_bet_store: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bf = self.beta_facing if self.beta_facing is not None else HEURISTIC_BETA_FACING.copy()  # fold/call/raise logits
        bn = self.beta_no_bet if self.beta_no_bet is not None else HEURISTIC_BETA_NO_BET.copy()  # call/raise when checked to us
        bf = _coerce_beta_to_phi_dim(bf, n_rows=3, name="beta_facing")                           # pad/truncate rows to PHI_DIM
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
        phi = self.feature_row(features)    # fixed-length context vector
        if features.facing_bet:
            beta = self._beta_facing_store  # three rows: one per facing action
            return {
                FOLD: float(beta[0] @ phi),
                CALL: float(beta[1] @ phi),
                RAISE: float(beta[2] @ phi),
            }
        beta = self._beta_no_bet_store      # two rows: aggression when no bet to face
        return {
            CALL: float(beta[0] @ phi),
            RAISE: float(beta[1] @ phi),
        }

    def base_probs(self, features: PostflopFeatures) -> Dict[int, float]:
        scores = self.base_scores(features)                 # unnormalized logits per global action id
        legal = self.legal_actions(features)                # e.g. no fold when not facing a bet
        sub = {a: scores[a] for a in legal}                 # restrict softmax support to legal set
        probs = _softmax_dict(sub)                          # baseline population policy before tilt
        return maybe_floor_action_probs(probs, self.floor)  # numerical floor on tiny probs

    @staticmethod
    def behavior_vector(action: int) -> np.ndarray:
        if action == FOLD:
            return np.array([1.0, 0.0, 0.0], dtype=float)  # one-hot in global 3-action space
        if action == CALL:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        if action == RAISE:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        raise ValueError(f"unknown action {action}")

    def action_utility_vectors(self, features: PostflopFeatures) -> Dict[int, np.ndarray]:
        p_base = self.base_probs(features)                             # reference distribution for contrast
        legal = self.legal_actions(features)
        expected = np.zeros(3, dtype=float)                            # E_{a~p_base}[ u(a) ] in R^3
        for a in legal:
            expected += float(p_base[a]) * self.behavior_vector(a)     # one-hot vectors averaged
        return {a: self.behavior_vector(a) - expected for a in legal}  # centered utilities for M-step

    def baseline_probs_matrix(self, phi: np.ndarray, facing: np.ndarray) -> np.ndarray:
        """Vectorized baseline softmax probabilities ``(N, 3)`` (no tendency tilt)."""
        phi = np.asarray(phi, dtype=float)                       # stack of feature rows (N, PHI_DIM)
        facing = np.asarray(facing, dtype=bool)                  # parallel bool: facing bet vs free play
        if phi.ndim != 2 or phi.shape[1] != PHI_DIM:
            raise ValueError(f"phi must have shape (N, {PHI_DIM}), got {phi.shape}")
        if facing.shape != (phi.shape[0],):
            raise ValueError(
                f"facing must be a 1-D bool array of length {phi.shape[0]}, got shape {facing.shape}"
            )                                                    # guard against ragged batching

        beta_f = self._beta_facing_store
        beta_n = self._beta_no_bet_store
        scores_f = phi @ beta_f.T                                # (N, 3) logits when facing a bet
        scores_n = phi @ beta_n.T                                # (N, 2) logits in no-bet branch
        base_facing = _row_softmax(scores_f)                     # per-row distribution over fold/call/raise
        base_no_bet = _row_softmax(scores_n)                     # per-row distribution over call/raise only

        probs = np.zeros((phi.shape[0], 3), dtype=float)         # always 3 columns for downstream matmuls
        probs[facing] = base_facing[facing]                      # copy facing rows wholesale
        no_bet_idx = ~facing
        if no_bet_idx.any():                                     # stitch 2-way softmax into 3-way layout (fold mass stays 0)
            probs[no_bet_idx, FOLD] = 0.0
            probs[no_bet_idx, CALL] = base_no_bet[no_bet_idx, 0]
            probs[no_bet_idx, RAISE] = base_no_bet[no_bet_idx, 1]

        if self.floor > 0.0:
            probs = _floor_row_probs(probs, self.floor, facing)  # row-wise epsilon floor respecting legality
        return probs


def fit_heuristic_postflop_prior() -> PostflopPrior:
    """Simple heuristic prior that can serve as a baseline."""
    return PostflopPrior()
