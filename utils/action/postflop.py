"""Postflop action model: population :class:`~utils.prior.postflop.PostflopPrior` × ``theta_post`` tilt."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from utils.action.postflop_context import (
    ACTION_BUCKETS,
    CALL,
    FOLD,
    HEURISTIC_BETA_FACING,
    HEURISTIC_BETA_NO_BET,
    PHI_DIM,
    PHI_DIM_BASE,
    PHI_DIM_EQUITY,
    PHI_DIM_RICH,
    RAISE,
    PostflopFeatures,
    _floor_row_probs,
    _softmax_log_probs,
    feature_vector,
    features_matrix,
    legal_actions,
    maybe_floor_action_probs,
    phi_column_labels,
    train_baseline_facing_bet,
    train_baseline_no_bet,
)
from utils.prior.postflop import PostflopPrior, fit_heuristic_postflop_prior


def _tilt_probs_matrix(
    probs: np.ndarray,
    facing: np.ndarray,
    theta: np.ndarray,
    floor: float,
) -> np.ndarray:
    log_p = np.log(np.maximum(probs, 1e-300))
    log_scores = log_p + theta[None, :]
    no_bet_idx = ~facing
    if no_bet_idx.any():
        log_scores[no_bet_idx, FOLD] = -np.inf
    m = log_scores.max(axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    w = np.exp(log_scores - m)
    if no_bet_idx.any():
        w[no_bet_idx, FOLD] = 0.0
    s = w.sum(axis=1, keepdims=True)
    out = w / np.maximum(s, 1e-300)
    if floor > 0.0:
        out = _floor_row_probs(out, floor, facing)
    return out


@dataclass
class PostflopActionModel:
    """Baseline :class:`PostflopPrior` with multinomial tilt ``theta_post``."""

    prior: PostflopPrior
    theta_post: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta_post", tuple(float(x) for x in self.theta_post))

    @property
    def floor(self) -> float:
        return self.prior.floor

    @property
    def theta_vec(self) -> np.ndarray:
        return np.asarray(self.theta_post, dtype=float)

    def with_theta(self, theta_post: Tuple[float, float, float]) -> PostflopActionModel:
        return PostflopActionModel(
            self.prior,
            tuple(float(x) for x in theta_post),
        )

    def feature_vector(self, features: PostflopFeatures) -> np.ndarray:
        return feature_vector(features)

    def legal_actions(self, features: PostflopFeatures) -> Tuple[int, ...]:
        return legal_actions(features)

    def base_probs(self, features: PostflopFeatures) -> Dict[int, float]:
        return self.prior.base_probs(features)

    def action_utility_vectors(self, features: PostflopFeatures) -> Dict[int, np.ndarray]:
        return self.prior.action_utility_vectors(features)

    def action_probs(self, features: PostflopFeatures) -> Dict[int, float]:
        legal = self.legal_actions(features)
        p_base = self.base_probs(features)
        utilities = self.action_utility_vectors(features)
        theta = self.theta_vec
        log_scores: Dict[int, float] = {}
        eps = max(self.floor, 1e-300)
        for a in legal:
            log_scores[a] = math.log(max(p_base[a], eps)) + float(theta @ utilities[a])
        probs = _softmax_log_probs(log_scores)
        return maybe_floor_action_probs(probs, self.floor)

    def action_probability(self, features: PostflopFeatures, action: int) -> float:
        if action not in self.legal_actions(features):
            return 0.0
        return self.action_probs(features)[action]

    def action_probs_matrix(self, phi: np.ndarray, facing: np.ndarray) -> np.ndarray:
        base = self.prior.baseline_probs_matrix(phi, facing)
        return _tilt_probs_matrix(base, facing, self.theta_vec, self.floor)

    def action_probs_matrix_given_theta(
        self,
        phi: np.ndarray,
        facing: np.ndarray,
        theta_post: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        base = self.prior.baseline_probs_matrix(phi, facing)
        th = np.asarray(theta_post, dtype=float).reshape(3)
        return _tilt_probs_matrix(base, facing, th, self.floor)

    def action_log_probs_matrix(self, phi: np.ndarray, facing: np.ndarray) -> np.ndarray:
        probs = self.action_probs_matrix(phi, facing)
        with np.errstate(divide="ignore"):
            return np.log(probs)

    def action_utilities_matrix(self, phi: np.ndarray, facing: np.ndarray) -> np.ndarray:
        probs = self.action_probs_matrix(phi, facing)
        eye = np.eye(3, dtype=float)
        expected = probs
        return eye[None, :, :] - expected[:, None, :]


__all__ = [
    "ACTION_BUCKETS",
    "CALL",
    "FOLD",
    "HEURISTIC_BETA_FACING",
    "HEURISTIC_BETA_NO_BET",
    "PHI_DIM",
    "PHI_DIM_BASE",
    "PHI_DIM_EQUITY",
    "PHI_DIM_RICH",
    "RAISE",
    "PostflopActionModel",
    "PostflopFeatures",
    "PostflopPrior",
    "feature_vector",
    "features_matrix",
    "fit_heuristic_postflop_prior",
    "legal_actions",
    "phi_column_labels",
    "train_baseline_facing_bet",
    "train_baseline_no_bet",
]
