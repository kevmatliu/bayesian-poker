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
right-pads them with zero columns so older checkpoints behave as if the
new features have zero weight until they are explicitly retrained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
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
    rich: Optional[np.ndarray] = None   # board-relative categorical features
    equity: float = -1.0    # equity for draw hands


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
    """Return the ordered legal action indices for ``features.facing_bet``.

    When not facing a bet, **fold is illegal** (caller should not emit fold
    labels in the no-bet training head). Facing a bet uses the full 3-way set.
    """
    if features.facing_bet:
        return (FOLD, CALL, RAISE)
    return (CALL, RAISE)


def _row_softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax over a ``(N, K)`` score matrix."""
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    s = e.sum(axis=1, keepdims=True)
    return e / np.maximum(s, 1e-300)


def maybe_floor_action_probs(probs: Dict[int, float], floor: float) -> Dict[int, float]:
    """Mix dict-shaped action probabilities toward uniform over legal actions."""
    if floor <= 0:
        return probs
    n = len(probs)
    if floor * n >= 1.0:
        raise ValueError("floor too large for number of legal actions")
    out = {a: (1.0 - floor * n) * p + floor for a, p in probs.items()}
    tot = sum(out.values())
    return {a: p / tot for a, p in out.items()}


def _floor_row_probs(
    probs: np.ndarray,
    floor: float,
    facing: np.ndarray,
) -> np.ndarray:
    """Vectorized row-wise floor mixing (facing vs no-bet legal action counts)."""
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
    """Stable softmax over an arbitrary finite action set keyed by ``int``."""
    if not scores:
        raise ValueError("empty scores")
    actions = list(scores.keys())
    vals = np.array([scores[a] for a in actions], dtype=float)
    m = float(np.max(vals))
    w = np.exp(vals - m)
    s = float(np.sum(w))
    return {a: float(wi / s) for a, wi in zip(actions, w)}


def _softmax_log_probs(log_p: Mapping[int, float], floor_log: float = -1e300) -> Dict[int, float]:
    """Softmax treating inputs as **log** scores, with a floor to avoid ``-inf``."""
    scores = {a: max(float(v), floor_log) for a, v in log_p.items()}
    return _softmax_dict(scores)

def _validate_beta_shape(arr: np.ndarray, shape: Tuple[int, int], name: str) -> None:
    """Raise ``ValueError`` if ``arr`` is not exactly ``shape`` (used by loaders/tests)."""
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
    """Stack :func:`feature_vector` for many feature rows → ``(N, PHI_DIM)``."""
    return np.stack([feature_vector(f) for f in rows], axis=0)
