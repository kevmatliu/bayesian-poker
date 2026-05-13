"""
Context for postflop action modeling

Feature vector with two halves:
- Base block of 13 original features
 ``[bias, made, draw, made*draw, polar_m, bet_frac_pot, pot_odds, in_position, 
    multiway, log(1+spr), street_turn, street_river, board_wetness]`` features.
- Richer block of 16 more features (patching) based on board-relative indicators.
  Attempting to make the features more human.
- Draw equity ``E_{turn,river}[made_percentile]`` produced by
  `utils.strength.fast_eval.rollout_equity_index_table`.

Postflop prior has two heads:
- facing bet: fold/call/raise over the full feature set (3 * PHI weights)
- no bet: call/raise over the full feature set (2 * PHI weights, fold illegal)

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


PHI_DIM_BASE = 13
PHI_DIM_RICH = RICH_FEAT_DIM
PHI_DIM_EQUITY = 1

PHI_DIM = PHI_DIM_BASE + PHI_DIM_RICH + PHI_DIM_EQUITY


_BASE_PHI_LABELS: Tuple[str, ...] = (
    "bias",             # 1 for intercept term
    "made",             # made percentile of hand [0, 1]
    "draw",             # outs-based draw strength proxy [0, 1]
    "made_x_draw",      # interaction of made and draw
    "polar_m",          # polarization of made around 0.5, 4(m - 0.5)^2
    "bet_frac_pot",     # bet size as fraction of pot (0 for no bet, >1 for overbet)
    "pot_odds",         # price to continue, bet size / (pot + bet)
    "in_position",      # 1 if acting after opponent, 0 if before
    "multiway",         # 1 if more than 2 players contesting the pot, else 0
    "log_1_plus_spr",   # log(1 + stack-to-pot ratio), clipped to avoid log(0)
    "street_turn",      # 1 if on the turn, 0 if on the flop (river is separate indicator)
    "street_river",     # 1 if on the river, 0 if on the flop (turn is separate indicator)
    "board_wetness",    # scalar measure of board texture, higher = wetter (more coordinated, more draws)
)


def phi_column_labels() -> Tuple[str, ...]:
    return _BASE_PHI_LABELS + RICH_FEAT_KEYS + ("equity",)

_HEURISTIC_BETA_FACING_BASE: np.ndarray = np.array(                                 # general heuristics made
    [
        [0.4, -3.0, -1.4, -0.5, 0.3, 1.7, 0.8, -0.3, 0.3, 0.1, 0.0, 0.2, 0.2],
        [0.2, 1.1, 1.2, 0.2, -0.6, -1.1, -0.7, 0.3, 0.1, 0.1, 0.1, -0.1, 0.1],
        [-1.0, 1.8, 1.5, 0.8, 0.9, -0.4, -0.2, 0.3, -0.7, -0.1, 0.0, -0.2, 0.3],
    ],
    dtype=float,
)

_HEURISTIC_BETA_NO_BET_BASE: np.ndarray = np.array(                                 # general heuristics no bet 
    [
        [0.3, -1.2, -1.0, -0.2, -0.6, 0.0, 0.0, -0.2, 0.2, 0.1, 0.0, 0.1, -0.1],
        [-0.1, 1.6, 1.3, 0.7, 0.9, 0.0, 0.0, 0.4, -0.6, -0.1, 0.0, -0.1, 0.3],
    ],
    dtype=float,
)


def _zero_pad_to_phi_dim(beta: np.ndarray) -> np.ndarray:
    """For intercept purposes"""
    arr = np.asarray(beta, dtype=float)
    if arr.shape[1] >= PHI_DIM:                                          # already full width
        return arr
    pad = np.zeros((arr.shape[0], PHI_DIM - arr.shape[1]), dtype=float)  # new feature cols = 0
    return np.concatenate([arr, pad], axis=1)                            # append zeros on the right


HEURISTIC_BETA_FACING: np.ndarray = _zero_pad_to_phi_dim(_HEURISTIC_BETA_FACING_BASE)  # 3×PHI
HEURISTIC_BETA_NO_BET: np.ndarray = _zero_pad_to_phi_dim(_HEURISTIC_BETA_NO_BET_BASE)  # 2×PHI


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
    equity: float = -1.0                # equity for draw hands


def _rich_block(features: PostflopFeatures) -> np.ndarray:
    """Extracting the board-relative features"""
    arr = features.rich
    a = np.asarray(arr, dtype=float)
    if a.shape == (RICH_FEAT_DIM,):
        return a
    if a.size >= RICH_FEAT_DIM:                 # longer → truncate head
        return a[:RICH_FEAT_DIM]
    out = np.zeros(RICH_FEAT_DIM, dtype=float)  # shorter → pad tail
    out[: a.size] = a
    return out


def _equity_value(features: PostflopFeatures) -> float:
    eq = float(features.equity)
    if eq < 0.0:
        return float(features.made)
    return eq                           # use rollout / provided equity


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
        1.0,                          # bias
        m,
        d,
        m * d,                        # interaction
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
    rich = _rich_block(features)      # board-relative tail
    equity = _equity_value(features)  # equity scalar (or made fallback)
    return np.concatenate(
        (np.asarray(base, dtype=float), rich, np.array([equity], dtype=float))
    )


def legal_actions(features: PostflopFeatures) -> Tuple[int, ...]:
    """
    Return legal action buckets for this state, as a tuple of ints. Fold is legal iff facing bet.
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
    if floor <= 0:                                                      # disabled fast path
        return probs
    n = len(probs)                                                      # number of legal actions in dict
    if floor * n >= 1.0:                                                # would invert mixing weights
        raise ValueError("floor too large for number of legal actions")
    out = {a: (1.0 - floor * n) * p + floor for a, p in probs.items()}  # Dirichlet-style mix
    tot = sum(out.values())                                             # renormalization constant
    return {a: p / tot for a, p in out.items()}                         # sum to 1


def _floor_row_probs(
    probs: np.ndarray,
    floor: float,
    facing: np.ndarray,
) -> np.ndarray:
    """Vectorized row-wise floor mixing (facing vs no-bet legal action counts)."""
    out = probs.copy()                            
    n_legal = np.where(facing, 3.0, 2.0)           
    mix_alpha = floor * n_legal                     # total floor mass budget per row
    if (mix_alpha >= 1.0).any():                    # invalid mixing coefficient
        raise ValueError("floor too large for number of legal actions")
    out = (1.0 - mix_alpha)[:, None] * out + floor  # blend toward uniform
    no_bet_idx = ~facing                            # rows with only call/raise legal
    if no_bet_idx.any():                            # zero illegal fold column
        out[no_bet_idx, FOLD] = 0.0
    row_sum = out.sum(axis=1, keepdims=True)        # renorm denominator
    return out / np.maximum(row_sum, 1e-300)        # safe divide


def _softmax_dict(scores: Mapping[int, float]) -> Dict[int, float]:
    """Stable softmax over an arbitrary finite action set keyed by ``int``."""
    if not scores:                                              
        raise ValueError("empty scores")
    actions = list(scores.keys())                               # preserve arbitrary key order
    vals = np.array([scores[a] for a in actions], dtype=float)  # score vector
    m = float(np.max(vals))                                     # log-sum-exp
    w = np.exp(vals - m)                                       
    s = float(np.sum(w))                                        
    return {a: float(wi / s) for a, wi in zip(actions, w)}      


def _softmax_log_probs(log_p: Mapping[int, float], floor_log: float = -1e300) -> Dict[int, float]:
    """Softmax treating inputs as **log** scores, with a floor to avoid ``-inf``."""
    scores = {a: max(float(v), floor_log) for a, v in log_p.items()}  # clamp log inputs
    return _softmax_dict(scores)                                      # delegate to linear-domain softmax

def _validate_beta_shape(arr: np.ndarray, shape: Tuple[int, int], name: str) -> None:
    """Raise ``ValueError`` if ``arr`` is not exactly ``shape`` (used by loaders/tests)."""
    a = np.asarray(arr, dtype=float)  # normalize type
    if a.shape != shape:              # strict shape check
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")


def _coerce_beta_to_phi_dim(
    beta: np.ndarray,
    *,
    n_rows: int,
    name: str,
) -> np.ndarray:
    """
    Force ``beta`` into shape ``(n_rows, PHI_DIM)``, padding with zeros if needed.
    """
    a = np.asarray(beta, dtype=float)                            # normalize
    if a.ndim != 2 or a.shape[0] != n_rows:                      # row count must match head (3 or 2)
        raise ValueError(
            f"{name} must have shape (rows={n_rows}, cols=*), got {a.shape}"
        )
    if a.shape[1] == PHI_DIM:                                    # already target width
        return a
    if a.shape[1] > PHI_DIM:                                     # would require truncation — refuse
        raise ValueError(
            f"{name} has {a.shape[1]} columns but PHI_DIM={PHI_DIM}; "
            "cannot truncate without losing information."
        )
    pad = np.zeros((n_rows, PHI_DIM - a.shape[1]), dtype=float)  # zero-fill new cols
    return np.concatenate([a, pad], axis=1)                      # right-pad


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
    """Stack :func:`feature_vector` for many feature rows, ``(N, PHI_DIM)``."""
    return np.stack([feature_vector(f) for f in rows], axis=0)  # batch design matrix
