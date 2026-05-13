"""Postflop action model: population :class:`~utils.prior.postflop.PostflopPrior` * ``theta_post`` tilt."""

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
    probs: np.ndarray,                          # baseline probs (N, 3)
    facing: np.ndarray,                         # bool mask: row faces a bet
    theta: np.ndarray,                          # length-3 tilt vector
    floor: float,                               # minimum per-action mass
) -> np.ndarray:
    log_p = np.log(np.maximum(probs, 1e-300))   # safe log baseline
    log_scores = log_p + theta[None, :]         # add same tilt to every row
    no_bet_idx = ~facing                        # rows where fold is illegal
    if no_bet_idx.any():                        # only mask where applicable
        log_scores[no_bet_idx, FOLD] = -np.inf  # forbid fold when no bet to face

    m = log_scores.max(axis=1, keepdims=True)   # per-row max for stability
    m = np.where(np.isfinite(m), m, 0.0)        # avoid NaN if all -inf
    w = np.exp(log_scores - m)                  # unnormalized weights

    if no_bet_idx.any():                        # zero out illegal fold mass after exp
        w[no_bet_idx, FOLD] = 0.0
    s = w.sum(axis=1, keepdims=True)            # row normalization factor
    out = w / np.maximum(s, 1e-300)             # probabilities

    if floor > 0.0:                             # optional epsilon mixing toward uniform legal
        out = _floor_row_probs(out, floor, facing)

    return out                                  # tilted distribution matrix


@dataclass
class PostflopActionModel:
    """Baseline :class:`PostflopPrior` with multinomial tilt ``theta_post``."""

    prior: PostflopPrior                                        # population baseline over PostflopFeatures
    theta_post: Tuple[float, float, float] = (0.0, 0.0, 0.0)    # (fold, call, raise) tilt tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta_post", tuple(float(x) for x in self.theta_post))  # frozen dataclass field fix

    @property
    def floor(self) -> float:
        return self.prior.floor  # delegate min mass

    @property
    def theta_vec(self) -> np.ndarray:
        return np.asarray(self.theta_post, dtype=float)  # ndarray view for matmul

    def with_theta(self, theta_post: Tuple[float, float, float]) -> PostflopActionModel:
        return PostflopActionModel(  # immutable-style copy
            self.prior,
            tuple(float(x) for x in theta_post),
        )

    def feature_vector(self, features: PostflopFeatures) -> np.ndarray:
        return feature_vector(features)  # passthrough to context helper

    def legal_actions(self, features: PostflopFeatures) -> Tuple[int, ...]:
        return legal_actions(features)  # facing-aware legal indices

    def base_probs(self, features: PostflopFeatures) -> Dict[int, float]:
        return self.prior.base_probs(features)  # population dict P(a|phi)

    def action_utility_vectors(self, features: PostflopFeatures) -> Dict[int, np.ndarray]:
        return self.prior.action_utility_vectors(features)  # u_a vectors for tilt

    def action_probs(self, features: PostflopFeatures) -> Dict[int, float]:
        legal = self.legal_actions(features)                                             # which actions exist this spot
        p_base = self.base_probs(features)                                               # baseline masses
        utilities = self.action_utility_vectors(features)                                # utility of each action
        theta = self.theta_vec                                                           
        log_scores: Dict[int, float] = {}                                                # accumulate logit scores
        eps = max(self.floor, 1e-300)                                                    # avoid log(0) using a very small pertubation (pls don't mark down :( )
        for a in legal:                                                                  # only score legal branches
            log_scores[a] = math.log(max(p_base[a], eps)) + float(theta @ utilities[a])  # logit = log p + theta * u
        probs = _softmax_log_probs(log_scores)                                           
        return maybe_floor_action_probs(probs, self.floor)                               # optional floor mix

    def action_probability(self, features: PostflopFeatures, action: int) -> float:
        if action not in self.legal_actions(features):  # illegal actions get zero mass
            return 0.0
        return self.action_probs(features)[action]      # lookup single bucket

    def action_probs_matrix(self, phi: np.ndarray, facing: np.ndarray) -> np.ndarray:
        base = self.prior.baseline_probs_matrix(phi, facing)                 # population (N,3) or masked rows
        return _tilt_probs_matrix(base, facing, self.theta_vec, self.floor)  # apply player tilt

    def action_probs_matrix_given_theta(
        self,
        phi: np.ndarray,
        facing: np.ndarray,
        theta_post: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        base = self.prior.baseline_probs_matrix(phi, facing)     # baseline batch
        th = np.asarray(theta_post, dtype=float).reshape(3)      # explicit tilt for counterfactuals
        return _tilt_probs_matrix(base, facing, th, self.floor)  # tilt with given theta

    def action_log_probs_matrix(self, phi: np.ndarray, facing: np.ndarray) -> np.ndarray:
        probs = self.action_probs_matrix(phi, facing)  # tilted probs
        with np.errstate(divide="ignore"):             # log(0) -> -inf allowed downstream
            return np.log(probs)

    def action_utilities_matrix(self, phi: np.ndarray, facing: np.ndarray) -> np.ndarray:
        probs = self.action_probs_matrix(phi, facing)   # (N, 3) probs
        eye = np.eye(3, dtype=float)                    # one-hot rows for each action
        expected = probs                                # E[u] weights rows by P(a)
        return eye[None, :, :] - expected[:, None, :]   # stacked deviation Jacobians / identities


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
