"""Preflop feature context: ``StateKey``, design matrix rows, and training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.action.common import (
    UtilityVector,
    dot3,
    softmax_dict_float,
    tendency_deviation_vector as _tendency_deviation_generic,
)
from utils.prior.training import train_multinomial_3_class
from utils.strength.preflop import HandClassFeatures, all_169_classes, hand_class_features

FOLD = 0            
CHECK_CALL = 1      
RAISE = 2           

ACTION_BUCKETS = (FOLD, CHECK_CALL, RAISE)  # ordered legal set

POSITIONS = {  # 1-based seat index → human label (HU order)
    1: "sb",
    2: "bb",
    3: "utg",
    4: "hj",
    5: "co",
    6: "btn",
}

_POSITION_ONEHOT_ORDER = ("sb", "bb", "utg", "hj", "co", "btn")     # stable one-hot axis order of position
_ACTIVE_ONEHOT_ORDER = ("heads_up", "three_way", "multiway")        # active count buckets
_RAISE_ONEHOT_ORDER = ("raises_0", "raises_1", "raises_2plus")      # aggression depth buckets
_SPR_ONEHOT_ORDER = ("deep", "medium", "shallow")                   # stack/pot ratio buckets


@dataclass(frozen=True)
class StateKey:
    """
    Discrete context for preflop policy features via string buckets.

    Each field is a string. Keys are separated using `|`.
    """

    position: str           # seat label bucket
    active_players: str     # HU / 3-way / multiway bucket
    facing_bet: str         # "facing_bet" vs "no_bet"
    raise_count: str        # prior raise depth bucket
    spr: str                # stack-to-pot bucket

    def as_string(self) -> str:
        return "|".join(
            [
                self.position,
                self.active_players,
                self.facing_bet,
                self.raise_count,
                self.spr,
            ]
        )

    @staticmethod
    def from_string(s: str) -> StateKey:
        fields = ("position", "active_players", "facing_bet", "raise_count", "spr")  # field order for zip
        parts = s.split("|")                                                         # tokenize serialized key
        if len(parts) != len(fields):                                                # malformed if arity mismatch
            raise ValueError(f"Invalid StateKey string: {s!r}")
        return StateKey(**dict(zip(fields, parts)))                                  # reconstruct dataclass


def _spr_str(stack: float, pot: float) -> str:
    """Bucket stack-to-pot ratio into ``deep`` / ``medium`` / ``shallow`` labels."""
    spr = (stack / pot) if pot > 0 else float("inf")    # avoid div-by-zero; treat as infinite SPR
    if spr > 10:                                        # very deep relative to pot
        return "deep"
    if spr > 3:                                         # middling commitment depth
        return "medium"
    return "shallow"                                    # short or committed


def _active_str(n: int) -> str:
    """Map active player count to ``heads_up`` / ``three_way`` / ``multiway``."""
    if n == 2:         # heads-up
        return "heads_up"
    if n == 3:         # three-handed
        return "three_way"
    return "multiway"  # four or more


def _raise_count_str(n: int) -> str:
    """Discretize the max raise level seen in parse betting history."""
    if n == 0:             # unopened pot
        return "raises_0"
    if n == 1:             # single raise
        return "raises_1"
    return "raises_2plus"  # reraise wars


def _seat_position(player: str, player_order: list[str]) -> str:
    """Seat label (``sb`` … ``btn``) from 1-based seat index in ``player_order``."""
    idx = player_order.index(player) + 1  # convert list index to 1-based seat
    return POSITIONS.get(idx, "unknown")  # fallback if table size ≠ 6 mapping


def state_key_from_parse_state(state, player: str) -> StateKey:
    """
    From the parse class, derive the StateKey.

    Note that there is a special case for the big blind before any raises,
    they have option to check and are not facing a bet. 
    """
    num_active = sum(1 for _, alive in state.players_in_hand if alive)  # still in pot
    stack = state.current_stacks.get(player, 0.0)                       # hero stack for SPR
    pot = state.pot_size                                                # current pot for SPR

    history = state.betting_history or []                               # street-local betting tuples
    raise_count = 0                                                     # default before parsing history

    bb_player = (                                                       # big blind seat for facing_bet heuristics
        state.player_order[1]
        if len(state.player_order) > 1
        else state.player_order[0]
    )

    if history:                                                         # action already occurred this street
        raise_count = max(lvl for _, (_, lvl), _ in history)            # deepest raise level observed

        if raise_count > 0:                                             # someone opened → always facing a bet to continue
            facing_bet = True
        else:                                                           # limped pot: BB can check, others face limp as bet
            facing_bet = player != bb_player
    else:                                                               # first action preflop: only BB may be not facing
        facing_bet = player != bb_player

    return StateKey(
        position=_seat_position(player, state.player_order),
        active_players=_active_str(num_active),
        facing_bet="facing_bet" if facing_bet else "no_bet",
        raise_count=_raise_count_str(raise_count),
        spr=_spr_str(stack, pot),
    )


def build_state_key(
    position: str,
    num_active: int,
    facing_bet: bool,
    raise_count: int,
    stack: float,
    pot: float,
) -> StateKey:
    """Construct a :class:`StateKey` from already-discretized quantities (tests / tooling)."""
    return StateKey(
        position=position,
        active_players=_active_str(num_active),
        facing_bet="facing_bet" if facing_bet else "no_bet",
        raise_count=_raise_count_str(raise_count),
        spr=_spr_str(stack, pot),
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))  # clip to [lo, hi]


def _state_playability_adjustment(state_key: StateKey) -> float:
    pos_adj = {
        "sb": -0.02,
        "bb": 0.01,
        "utg": -0.08,
        "hj": -0.02,
        "co": 0.05,
        "btn": 0.08,
    }.get(state_key.position, 0.0)                                        # positional looseness prior

    multiway_adj = {
        "heads_up": 0.04,
        "three_way": -0.02,
        "multiway": -0.06,
    }.get(state_key.active_players, 0.0)                                  # multiway tightens ranges

    raise_adj = {
        "raises_0": 0.0,
        "raises_1": -0.12,
        "raises_2plus": -0.24,
    }.get(state_key.raise_count, 0.0)                                     # more raises → defend less wide

    spr_adj = {
        "deep": 0.0,
        "medium": 0.03,
        "shallow": 0.07,
    }.get(state_key.spr, 0.0)                                             # shallow stacks push thinner value

    facing_adj = -0.05 if state_key.facing_bet == "facing_bet" else 0.02  # cold vs open spot

    return pos_adj + multiway_adj + raise_adj + spr_adj + facing_adj      # scalar playability shift


def _playability(features: HandClassFeatures, state_key: StateKey) -> float:
    return _clamp(
        features.strength + _state_playability_adjustment(state_key),  # combine hand + context
        0.0,
        1.0,
    )


def _one_hot(value: str, order: Tuple[str, ...]) -> list[float]:
    return [1.0 if value == x else 0.0 for x in order]  # indicator vector along order


def preflop_feature_vector(hand_class: str, state_key: StateKey | str) -> np.ndarray:
    """Fixed-length features for multinomial logit baseline P_preflop(a | h, s)."""
    if isinstance(state_key, str):                                                                              # accept serialized keys
        state_key = StateKey.from_string(state_key)

    f = hand_class_features(hand_class)                                                                         # 169-class shape stats
    sk = state_key                                                                                              # local alias
    p = _playability(f, sk)                                                                                     # [0,1] scalar summary
    facing = sk.facing_bet == "facing_bet"                                                                      # binary flag for model
    premium = float((f.pair and f.high >= 10) or (f.high == 14 and f.low >= 11))                                # TT+ / broadway Ax
    speculative = float(f.suited and f.gap <= 2 and f.low >= 5)                                                 # suited connectors/gappers

    pos_oh = _one_hot(sk.position if sk.position in _POSITION_ONEHOT_ORDER else "btn", _POSITION_ONEHOT_ORDER)  # unknown → btn default
    act_oh = _one_hot(sk.active_players, _ACTIVE_ONEHOT_ORDER)
    raise_oh = _one_hot(sk.raise_count, _RAISE_ONEHOT_ORDER)
    spr_oh = _one_hot(sk.spr, _SPR_ONEHOT_ORDER)

    parts = [
        1.0,                                                                                                    # bias term
        f.strength,                                                                                             # normalized combo strength
        f.high / 14.0,                                                                                          # high card rank scaled
        f.low / 14.0,                                                                                           # low card rank scaled
        float(f.pair),                                                                                          # pocket pair indicator
        float(f.suited),                                                                                        # suitedness
        f.gap / 12.0,                                                                                           # rank gap scaled
        f.broadways / 2.0,                                                                                      # broadway count scaled
        float(f.has_ace),                                                                                       # contains an ace
        p,                                                                                                      # playability scalar
        premium,                                                                                                # premium bucket mass
        speculative,                                                                                            # speculative suited structure
        float(facing),                                                                                          # facing bet flag
    ] + pos_oh + act_oh + raise_oh + spr_oh                                                                     # concatenate one-hot blocks

    return np.asarray(parts, dtype=float)                                                                       # row vector for regression


PREFLOP_PHI_DIM = int(preflop_feature_vector("AKs", build_state_key("btn", 2, False, 0, 100.0, 1.5)).shape[0])  # inferring the dimension from a sample vector


def reference_preflop_logits(features: HandClassFeatures, state_key: StateKey) -> Dict[int, float]:
    """Getting preflop logits used to ridge-fit :data:`HEURISTIC_BETA_PREFLOP` against ``phi``."""
    p = _playability(features, state_key)           # context-adjusted playability scalar in [0,1] for interpolation between fold/call/raise tendencies
    facing = state_key.facing_bet == "facing_bet"

    premium = float(
        (features.pair and features.high >= 10)
        or (features.high == 14 and features.low >= 11)
    )

    speculative = float(
        features.suited
        and features.gap <= 2
        and features.low >= 5
    )

    spr_aggression = {
        "deep": 0.0,
        "medium": 0.04,
        "shallow": 0.10,
    }.get(state_key.spr, 0.0)                      # shallow --> more raise logits

    # some heuristics to capture intuitive strategic effects of hand strength and context on fold/call/raise tendencies,
    # used as a teacher for ridge regression of the heuristic beta

    if facing:                                     # defend / continue spot
        return {
            FOLD: 1.5 - 3.0 * p,
            CHECK_CALL: 0.2 + 1.4 * p + 0.15 * speculative,
            RAISE: -1.0 + (1.35 + spr_aggression) * p + 0.55 * premium,
        }

    return {
        FOLD: 1.1 - 2.4 * p,                       # still nonzero but discouraged open-fold
        CHECK_CALL: 0.25 + 0.9 * p + 0.25 * speculative,
        RAISE: -0.55 + (1.15 + spr_aggression) * p + 0.35 * premium,
    }


def _ridge_fit_preflop_beta(
    n_samples: int = 5000,
    ridge: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Monte-carlo ridge regression to fit a single global beta vector for preflop action tendencies.

    Using monte-carlo because feature space is so large and sparse
    """
    rng = np.random.default_rng(seed)                             # reproducible synthetic data
    hands = all_169_classes()                                     # universe of labels
    rows: list[np.ndarray] = []                                   # design matrix rows
    targets: list[list[float]] = []                               # 3-way logit targets

    for _ in range(n_samples):                                    # Monte Carlo ridge target
        h = hands[int(rng.integers(len(hands)))]                  # getting a random combo of hands
        sk = StateKey(
            position=rng.choice(_POSITION_ONEHOT_ORDER),
            active_players=rng.choice(_ACTIVE_ONEHOT_ORDER),
            facing_bet=rng.choice(("facing_bet", "no_bet")),
            raise_count=rng.choice(_RAISE_ONEHOT_ORDER),
            spr=rng.choice(_SPR_ONEHOT_ORDER),
        )
        phi = preflop_feature_vector(h, sk)                       # features for (h,s)
        feat = hand_class_features(h)                             # structured stats for teacher
        leg = reference_preflop_logits(feat, sk)                  # closed-form logits
        rows.append(phi)
        targets.append([leg[FOLD], leg[CHECK_CALL], leg[RAISE]])  # row of Y

    X = np.stack(rows)                                            # (N, D)
    Y = np.array(targets)                                         # (N, 3)
    d = X.shape[1]                                                # feature dimension
    xt_x = X.T @ X + ridge * np.eye(d)                            # Normalizing the covariance with ridge for stability purposes
    beta_dt = np.linalg.solve(xt_x, X.T @ Y)                      # (D,3) coefficient block

    return beta_dt.T                                              # (3,D) rows = action-specific weights


HEURISTIC_BETA_PREFLOP: np.ndarray = _ridge_fit_preflop_beta()      # module-level cached ridge solution


def tendency_deviation_vector(
    candidate_action: int,
    p_base: Mapping[int, float],
) -> UtilityVector:
    """u_k = 1[a=k] - P_base(k) for preflop FOLD / CHECK_CALL / RAISE."""
    return _tendency_deviation_generic(
        candidate_action,
        p_base,
        fold=FOLD,
        call=CHECK_CALL,
        raise_=RAISE,
    )


def _canonical_action(action_bucket: int) -> int:
    if action_bucket == FOLD:                                      # explicit fold code
        return FOLD
    if action_bucket == CHECK_CALL:                                # call/check bucket
        return CHECK_CALL
    if action_bucket in (RAISE, 3, 4):                             # legacy multi-raise indices → single raise
        return RAISE
    raise ValueError(f"Unknown action bucket: {action_bucket!r}")  # invalid storage


def canonical_preflop_action(action_bucket: int) -> int:
    """Map stored preflop bucket indices to ``{FOLD, CHECK_CALL, RAISE}``."""
    return _canonical_action(action_bucket)


def theta_pre_from_phi(phi: float) -> Tuple[float, float, float]:
    """Map a scalar aggressiveness shim ``phi`` to a length-3 ``theta_pre`` tuple."""
    p = float(phi)     # ensure scalar float
    return (-p, p, p)  # downweight fold, upweight call+raise symmetrically


def train_baseline_preflop(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.15,
    max_epochs: int = 2000,
    tol: float = 1e-7,
    l2: float = 0.0,
) -> np.ndarray:
    """Population multinomial logistic regression; labels {FOLD,CHECK_CALL,RAISE} = {0,1,2}."""
    return train_multinomial_3_class(
        X,
        y,
        PREFLOP_PHI_DIM,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        tol=tol,
        l2=l2,
    )


def features_matrix_preflop(pairs: Sequence[Tuple[str, StateKey | str]]) -> np.ndarray:
    """Batch :func:`preflop_feature_vector` for ``(hand_class, state_key)`` pairs → ``(N, D)``."""
    return np.stack([preflop_feature_vector(h, sk) for h, sk in pairs], axis=0)  # stack rows
