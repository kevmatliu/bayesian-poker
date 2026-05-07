"""Post-flop action prior: baseline multinomial logit + player tendency tilting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from utils.prior.common import PriorMode
from utils.prior.training import train_multinomial_2_class, train_multinomial_3_class

FOLD = 0
CALL = 1
RAISE = 2

ACTION_BUCKETS = (FOLD, CALL, RAISE)

PHI_DIM = 13

HEURISTIC_BETA_FACING: np.ndarray = np.array(
    [
        [
            0.4,
            -3.0,
            -1.4,
            -0.5,
            0.3,
            1.7,
            0.8,
            -0.3,
            0.3,
            0.1,
            0.0,
            0.2,
            0.2,
        ],
        [
            0.2,
            1.1,
            1.2,
            0.2,
            -0.6,
            -1.1,
            -0.7,
            0.3,
            0.1,
            0.1,
            0.1,
            -0.1,
            0.1,
        ],
        [
            -1.0,
            1.8,
            1.5,
            0.8,
            0.9,
            -0.4,
            -0.2,
            0.3,
            -0.7,
            -0.1,
            0.0,
            -0.2,
            0.3,
        ],
    ],
    dtype=float,
)

HEURISTIC_BETA_NO_BET: np.ndarray = np.array(
    [
        [
            0.3,
            -1.2,
            -1.0,
            -0.2,
            -0.6,
            0.0,
            0.0,
            -0.2,
            0.2,
            0.1,
            0.0,
            0.1,
            -0.1,
        ],
        [
            -0.1,
            1.6,
            1.3,
            0.7,
            0.9,
            0.0,
            0.0,
            0.4,
            -0.6,
            -0.1,
            0.0,
            -0.1,
            0.3,
        ],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class PostflopFeatures:
    made: float
    draw: float
    bet_frac_pot: float
    pot_odds: float
    in_position: bool
    multiway: bool
    spr: float
    street: str
    board_wetness: float
    facing_bet: bool


def feature_vector(features: PostflopFeatures) -> np.ndarray:
    """phi(m, d, state) as a length-PHI_DIM column vector."""
    m = float(features.made)
    d = float(features.draw)
    polar_m = 4.0 * (m - 0.5) ** 2
    spr_log = math.log(1.0 + max(float(features.spr), 0.0))
    st = 1.0 if features.street == "turn" else 0.0
    sr = 1.0 if features.street == "river" else 0.0
    ip = 1.0 if features.in_position else 0.0
    mw = 1.0 if features.multiway else 0.0
    return np.array(
        [
            1.0,
            m,
            d,
            m * d,
            polar_m,
            float(features.bet_frac_pot),
            float(features.pot_odds),
            ip,
            mw,
            spr_log,
            st,
            sr,
            float(features.board_wetness),
        ],
        dtype=float,
    )


def legal_actions(features: PostflopFeatures) -> Tuple[int, ...]:
    if features.facing_bet:
        return (FOLD, CALL, RAISE)
    return (CALL, RAISE)


def _softmax_dict(scores: Mapping[int, float]) -> Dict[int, float]:
    if not scores:
        raise ValueError("empty scores")
    actions = list(scores.keys())
    vals = np.array([scores[a] for a in actions], dtype=float)
    m = float(np.max(vals))
    w = np.exp(vals - m)
    s = float(np.sum(w))
    return {a: float(wi / s) for a, wi in zip(actions, w)}


def _softmax_log_probs(log_p: Mapping[int, float], floor_log: float = -1e300) -> Dict[int, float]:
    scores = {a: max(float(v), floor_log) for a, v in log_p.items()}
    return _softmax_dict(scores)


@dataclass
class PostflopPrior:
    """Population baseline (beta) × session/player tilt (theta_post)."""

    theta_post: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    floor: float = 1e-6
    beta_facing: Optional[np.ndarray] = None
    beta_no_bet: Optional[np.ndarray] = None
    _beta_facing_store: np.ndarray = field(init=False, repr=False)
    _beta_no_bet_store: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta_post", tuple(float(x) for x in self.theta_post))
        bf = self.beta_facing if self.beta_facing is not None else HEURISTIC_BETA_FACING.copy()
        bn = self.beta_no_bet if self.beta_no_bet is not None else HEURISTIC_BETA_NO_BET.copy()
        _validate_beta_shape(bf, (3, PHI_DIM), "beta_facing")
        _validate_beta_shape(bn, (2, PHI_DIM), "beta_no_bet")
        object.__setattr__(self, "_beta_facing_store", np.asarray(bf, dtype=float))
        object.__setattr__(self, "_beta_no_bet_store", np.asarray(bn, dtype=float))

    @property
    def mode(self) -> PriorMode:
        return PriorMode.POSTFLOP

    @property
    def theta_vec(self) -> np.ndarray:
        return np.asarray(self.theta_post, dtype=float)

    @property
    def beta_facing_matrix(self) -> np.ndarray:
        return np.asarray(self._beta_facing_store, dtype=float).copy()

    @property
    def beta_no_bet_matrix(self) -> np.ndarray:
        return np.asarray(self._beta_no_bet_store, dtype=float).copy()

    def feature_vector(self, features: PostflopFeatures) -> np.ndarray:
        return feature_vector(features)

    def legal_actions(self, features: PostflopFeatures) -> Tuple[int, ...]:
        return legal_actions(features)

    def base_scores(self, features: PostflopFeatures) -> Dict[int, float]:
        phi = self.feature_vector(features)
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
        return self._maybe_floor_probs(probs)

    def _maybe_floor_probs(self, probs: Dict[int, float]) -> Dict[int, float]:
        if self.floor <= 0:
            return probs
        n = len(probs)
        if self.floor * n >= 1.0:
            raise ValueError("floor too large for number of legal actions")
        out = {a: (1.0 - self.floor * n) * p + self.floor for a, p in probs.items()}
        tot = sum(out.values())
        return {a: p / tot for a, p in out.items()}

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
        return self._maybe_floor_probs(probs)

    def action_probability(self, features: PostflopFeatures, action: int) -> float:
        if action not in self.legal_actions(features):
            return 0.0
        return self.action_probs(features)[action]

    def with_theta(self, theta_post: Tuple[float, float, float]) -> PostflopPrior:
        return PostflopPrior(
            theta_post=theta_post,
            floor=self.floor,
            beta_facing=self.beta_facing_matrix,
            beta_no_bet=self.beta_no_bet_matrix,
        )


def _validate_beta_shape(arr: np.ndarray, shape: Tuple[int, int], name: str) -> None:
    a = np.asarray(arr, dtype=float)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")


def train_baseline_facing_bet(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.15,
    max_epochs: int = 2000,
    tol: float = 1e-7,
    l2: float = 0.0,
) -> np.ndarray:
    """Fit beta_facing (3, PHI_DIM) for labels in {FOLD=0, CALL=1, RAISE=2}."""
    return train_multinomial_3_class(
        X,
        y,
        PHI_DIM,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        tol=tol,
        l2=l2,
    )


def train_baseline_no_bet(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.15,
    max_epochs: int = 2000,
    tol: float = 1e-7,
    l2: float = 0.0,
) -> np.ndarray:
    """Fit beta_no_bet (2, PHI_DIM). Labels must be CALL=1 or RAISE=2."""
    return train_multinomial_2_class(
        X,
        y,
        PHI_DIM,
        label_a=CALL,
        label_b=RAISE,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        tol=tol,
        l2=l2,
    )


def features_matrix(rows: Iterable[PostflopFeatures]) -> np.ndarray:
    return np.stack([feature_vector(f) for f in rows], axis=0)


def fit_heuristic_postflop_prior() -> PostflopPrior:
    return PostflopPrior()
