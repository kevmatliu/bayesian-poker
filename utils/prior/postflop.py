"""Post-flop action prior: baseline multinomial logit + player tendency tilting.

The feature vector ``phi`` has two halves:

* **Base block** (``PHI_DIM_BASE = 13``): the original ``[bias, made, draw,
  made*draw, polar_m, bet_frac_pot, pot_odds, in_position, multiway,
  log(1+spr), street_turn, street_river, board_wetness]`` features.
* **Rich block** (``PHI_DIM_RICH = 16``, Method A): board-relative
  per-combo categorical indicators (top pair / overpair / flush draw / …)
  produced by :func:`utils.strength.postflop.hand_feature_vector`.
* **Equity slot** (``+1``, Method E): the multi-street rollout equity
  ``E_{turn,river}[made_percentile]`` produced by
  :func:`utils.strength.fast_eval.rollout_equity_index_table`.

This brings ``PHI_DIM`` from 13 to 30. Saved beta matrices trained on the
old 13-dim phi load transparently — :func:`_coerce_beta_to_phi_dim`
right-pads them with zero columns so legacy populations behave as if the
new features have zero weight until they are explicitly retrained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.prior.common import PriorMode
from utils.prior.training import train_multinomial_2_class, train_multinomial_3_class
from utils.strength.postflop import RICH_FEAT_DIM, RICH_FEAT_KEYS

FOLD = 0
CALL = 1
RAISE = 2

ACTION_BUCKETS = (FOLD, CALL, RAISE)

# Original 13-dim base block: [bias, made, draw, made*draw, polar_m,
# bet_frac_pot, pot_odds, in_position, multiway, log(1+spr), street_turn,
# street_river, board_wetness].
PHI_DIM_BASE = 13
# Method A: board-relative per-combo categorical features.
PHI_DIM_RICH = RICH_FEAT_DIM
# Method E: multi-street rollout equity slot (single scalar).
PHI_DIM_EQUITY = 1

PHI_DIM = PHI_DIM_BASE + PHI_DIM_RICH + PHI_DIM_EQUITY  # 30

# Offsets into phi for slicing batched feature blocks.
PHI_OFFSET_BASE = 0
PHI_OFFSET_RICH = PHI_DIM_BASE
PHI_OFFSET_EQUITY = PHI_DIM_BASE + PHI_DIM_RICH


_BASE_PHI_LABELS: Tuple[str, ...] = (
    "bias",
    "made",
    "draw",
    "made_x_draw",
    "polar_m",
    "bet_frac_pot",
    "pot_odds",
    "in_position",
    "multiway",
    "log_1_plus_spr",
    "street_turn",
    "street_river",
    "board_wetness",
)


def phi_column_labels() -> Tuple[str, ...]:
    """Names for each entry of the length-``PHI_DIM`` feature vector."""
    return _BASE_PHI_LABELS + RICH_FEAT_KEYS + ("equity",)

_HEURISTIC_BETA_FACING_BASE: np.ndarray = np.array(
    [
        [0.4, -3.0, -1.4, -0.5, 0.3, 1.7, 0.8, -0.3, 0.3, 0.1, 0.0, 0.2, 0.2],
        [0.2, 1.1, 1.2, 0.2, -0.6, -1.1, -0.7, 0.3, 0.1, 0.1, 0.1, -0.1, 0.1],
        [-1.0, 1.8, 1.5, 0.8, 0.9, -0.4, -0.2, 0.3, -0.7, -0.1, 0.0, -0.2, 0.3],
    ],
    dtype=float,
)

_HEURISTIC_BETA_NO_BET_BASE: np.ndarray = np.array(
    [
        [0.3, -1.2, -1.0, -0.2, -0.6, 0.0, 0.0, -0.2, 0.2, 0.1, 0.0, 0.1, -0.1],
        [-0.1, 1.6, 1.3, 0.7, 0.9, 0.0, 0.0, 0.4, -0.6, -0.1, 0.0, -0.1, 0.3],
    ],
    dtype=float,
)


def _zero_pad_to_phi_dim(beta: np.ndarray) -> np.ndarray:
    """Right-pad ``beta`` with zero columns up to :data:`PHI_DIM`."""
    arr = np.asarray(beta, dtype=float)
    if arr.shape[1] >= PHI_DIM:
        return arr
    pad = np.zeros((arr.shape[0], PHI_DIM - arr.shape[1]), dtype=float)
    return np.concatenate([arr, pad], axis=1)


# Heuristic priors, padded to the new PHI_DIM (rich + equity columns = 0).
HEURISTIC_BETA_FACING: np.ndarray = _zero_pad_to_phi_dim(_HEURISTIC_BETA_FACING_BASE)
HEURISTIC_BETA_NO_BET: np.ndarray = _zero_pad_to_phi_dim(_HEURISTIC_BETA_NO_BET_BASE)


_ZERO_RICH = np.zeros(RICH_FEAT_DIM, dtype=float)


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
    # Method A: optional board-relative categorical features per combo.
    rich: Optional[np.ndarray] = None
    # Method E: multi-street rollout equity. ``-1`` sentinel means
    # "fall back to ``made``" so legacy call-sites work unchanged.
    equity: float = -1.0


def _rich_block(features: PostflopFeatures) -> np.ndarray:
    arr = features.rich
    if arr is None:
        return _ZERO_RICH
    a = np.asarray(arr, dtype=float)
    if a.shape == (RICH_FEAT_DIM,):
        return a
    if a.size >= RICH_FEAT_DIM:
        return a[:RICH_FEAT_DIM]
    out = np.zeros(RICH_FEAT_DIM, dtype=float)
    out[: a.size] = a
    return out


def _equity_value(features: PostflopFeatures) -> float:
    eq = float(features.equity)
    if eq < 0.0:
        return float(features.made)
    return eq


def feature_vector(features: PostflopFeatures) -> np.ndarray:
    """``phi(m, d, state)`` as a length-:data:`PHI_DIM` row vector."""
    m = float(features.made)
    d = float(features.draw)
    polar_m = 4.0 * (m - 0.5) ** 2
    spr_log = math.log(1.0 + max(float(features.spr), 0.0))
    st = 1.0 if features.street == "turn" else 0.0
    sr = 1.0 if features.street == "river" else 0.0
    ip = 1.0 if features.in_position else 0.0
    mw = 1.0 if features.multiway else 0.0
    base = (
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
    )
    rich = _rich_block(features)
    equity = _equity_value(features)
    return np.concatenate(
        (np.asarray(base, dtype=float), rich, np.array([equity], dtype=float))
    )


def legal_actions(features: PostflopFeatures) -> Tuple[int, ...]:
    if features.facing_bet:
        return (FOLD, CALL, RAISE)
    return (CALL, RAISE)


def _row_softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax over a ``(N, K)`` score matrix."""
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    s = e.sum(axis=1, keepdims=True)
    return e / np.maximum(s, 1e-300)


def _floor_row_probs(
    probs: np.ndarray,
    floor: float,
    facing: np.ndarray,
) -> np.ndarray:
    """Vectorized analogue of :meth:`PostflopPrior._maybe_floor_probs`.

    Facing-bet rows have 3 legal actions; no-bet rows only 2 (FOLD must
    stay at 0). The flooring rule used in the per-row path mixes the
    softmax output with a uniform-over-legal-actions distribution.
    """
    out = probs.copy()
    n_legal = np.where(facing, 3.0, 2.0)
    mix_alpha = floor * n_legal
    if (mix_alpha >= 1.0).any():
        raise ValueError("floor too large for number of legal actions")
    out = (1.0 - mix_alpha)[:, None] * out + floor
    no_bet_idx = ~facing
    if no_bet_idx.any():
        out[no_bet_idx, FOLD] = 0.0
    row_sum = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sum, 1e-300)


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
        bf = _coerce_beta_to_phi_dim(bf, n_rows=3, name="beta_facing")
        bn = _coerce_beta_to_phi_dim(bn, n_rows=2, name="beta_no_bet")
        object.__setattr__(self, "_beta_facing_store", bf)
        object.__setattr__(self, "_beta_no_bet_store", bn)

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

    # ------------------------------------------------------------------
    # Method D: vectorized batch API over a (N, PHI_DIM) feature matrix.
    # All per-combo loops in the filter, the EM E-step, and the EM
    # gradient go through these to avoid Python-level softmax/dot per
    # combo.
    # ------------------------------------------------------------------

    def action_probs_matrix(
        self,
        phi: np.ndarray,
        facing: np.ndarray,
    ) -> np.ndarray:
        """Vectorized action probabilities for ``N`` feature rows.

        Returns a ``(N, 3)`` array indexed by ``(FOLD, CALL, RAISE)``.
        For rows where ``facing[i] == False``, the ``FOLD`` column is
        forced to 0 and ``CALL`` / ``RAISE`` are renormalized over the
        two legal actions, matching the per-row :meth:`action_probs`.

        Equivalent to looping over rows and calling :meth:`action_probs`,
        but ~50–200x faster: one ``(N, PHI_DIM) @ (PHI_DIM, 3)`` matmul
        plus a vectorized softmax instead of N Python softmaxes.
        """
        phi = np.asarray(phi, dtype=float)
        facing = np.asarray(facing, dtype=bool)
        if phi.ndim != 2 or phi.shape[1] != PHI_DIM:
            raise ValueError(
                f"phi must have shape (N, {PHI_DIM}), got {phi.shape}"
            )
        if facing.shape != (phi.shape[0],):
            raise ValueError(
                f"facing must be a 1-D bool array of length {phi.shape[0]}, "
                f"got shape {facing.shape}"
            )

        beta_f = self._beta_facing_store  # (3, PHI_DIM)
        beta_n = self._beta_no_bet_store  # (2, PHI_DIM)
        theta = self.theta_vec

        # Base scores under both heads. The no-bet head is computed for
        # every row; we mask FOLD afterwards for the no-bet subset.
        # Facing head scores: (N, 3)
        scores_f = phi @ beta_f.T
        # No-bet head scores: (N, 2), columns = CALL, RAISE.
        scores_n = phi @ beta_n.T

        # Stable softmax → base probs.
        base_facing = _row_softmax(scores_f)
        base_no_bet = _row_softmax(scores_n)

        # Combine into a (N, 3) probability matrix.
        probs = np.zeros((phi.shape[0], 3), dtype=float)
        probs[facing] = base_facing[facing]
        no_bet_idx = ~facing
        if no_bet_idx.any():
            probs[no_bet_idx, FOLD] = 0.0
            probs[no_bet_idx, CALL] = base_no_bet[no_bet_idx, 0]
            probs[no_bet_idx, RAISE] = base_no_bet[no_bet_idx, 1]

        if self.floor > 0.0:
            probs = _floor_row_probs(probs, self.floor, facing)

        # Tendency tilt: log p_base + theta @ u(a)
        # u(a) = e_a - E[e_a | base] so theta @ u(a) = theta_a - sum_b p_b * theta_b.
        # log score: log p_a + theta_a - mean_theta (constant) -> drops in softmax.
        # We compute log p + theta_action then renormalize per row.
        log_p = np.log(np.maximum(probs, 1e-300))
        log_scores = log_p + theta[None, :]
        # Mask FOLD to -inf on no-bet rows so it stays at 0 after softmax.
        if no_bet_idx.any():
            log_scores[no_bet_idx, FOLD] = -np.inf

        m = log_scores.max(axis=1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        w = np.exp(log_scores - m)
        # Mask FOLD explicitly to 0 (exp of -inf already gives 0 but guard nans).
        if no_bet_idx.any():
            w[no_bet_idx, FOLD] = 0.0
        s = w.sum(axis=1, keepdims=True)
        out = w / np.maximum(s, 1e-300)

        if self.floor > 0.0:
            out = _floor_row_probs(out, self.floor, facing)

        return out

    def action_log_probs_matrix(
        self,
        phi: np.ndarray,
        facing: np.ndarray,
    ) -> np.ndarray:
        """``log P(a | x)`` matrix of shape ``(N, 3)``. ``-inf`` on illegal entries."""
        probs = self.action_probs_matrix(phi, facing)
        with np.errstate(divide="ignore"):
            return np.log(probs)

    def action_utilities_matrix(
        self,
        phi: np.ndarray,
        facing: np.ndarray,
    ) -> np.ndarray:
        """``u(a) - E_p[u]`` deviation matrix of shape ``(N, 3, 3)``.

        Used by the EM gradient: for each row ``i`` and each action
        ``a``, returns the deviation vector ``e_a - sum_b p(b) e_b``.
        For no-bet rows the FOLD-row deviation is the zero vector (the
        gradient masks it via the action labels anyway).
        """
        probs = self.action_probs_matrix(phi, facing)
        eye = np.eye(3, dtype=float)
        expected = probs  # since u(a) = e_a, E[u] = sum_b p_b e_b = probs row.
        return eye[None, :, :] - expected[:, None, :]


def _validate_beta_shape(arr: np.ndarray, shape: Tuple[int, int], name: str) -> None:
    a = np.asarray(arr, dtype=float)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")


def _coerce_beta_to_phi_dim(
    beta: np.ndarray,
    *,
    n_rows: int,
    name: str,
) -> np.ndarray:
    """Force ``beta`` into shape ``(n_rows, PHI_DIM)``, padding with zeros if needed.

    Saved beta matrices trained against ``PHI_DIM_BASE = 13`` (or any
    intermediate dim) are right-padded with zero columns so they continue
    to work as if the additional features have zero coefficient. The row
    dimension is strict; only the feature dimension is permissive.
    """
    a = np.asarray(beta, dtype=float)
    if a.ndim != 2 or a.shape[0] != n_rows:
        raise ValueError(
            f"{name} must have shape (rows={n_rows}, cols=*), got {a.shape}"
        )
    if a.shape[1] == PHI_DIM:
        return a
    if a.shape[1] > PHI_DIM:
        raise ValueError(
            f"{name} has {a.shape[1]} columns but PHI_DIM={PHI_DIM}; "
            "cannot truncate without losing information."
        )
    pad = np.zeros((n_rows, PHI_DIM - a.shape[1]), dtype=float)
    return np.concatenate([a, pad], axis=1)


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
