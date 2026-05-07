"""Preflop action prior: multinomial-logit baseline + tendency tilting."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.prior.common import (
    PriorMode,
    UtilityVector,
    dot3,
    softmax_dict_legacy,
    tendency_deviation_vector as _tendency_deviation_generic,
)
from utils.prior.training import train_multinomial_3_class
from utils.strength.preflop import HandClassFeatures, all_169_classes, hand_class_features

FOLD = 0
CHECK_CALL = 1
RAISE = 2

ACTION_BUCKETS = (FOLD, CHECK_CALL, RAISE)

CBR_SMALL = RAISE
CBR_MEDIUM = RAISE
CBR_LARGE = RAISE

OLD_TO_THREE_ACTION = {
    0: FOLD,
    1: CHECK_CALL,
    2: RAISE,
    3: RAISE,
    4: RAISE,
}

POSITIONS = {
    1: "sb",
    2: "bb",
    3: "utg",
    4: "hj",
    5: "co",
    6: "btn",
}

_POSITION_ONEHOT_ORDER = ("sb", "bb", "utg", "hj", "co", "btn")
_ACTIVE_ONEHOT_ORDER = ("heads_up", "three_way", "multiway")
_RAISE_ONEHOT_ORDER = ("raises_0", "raises_1", "raises_2plus")
_SPR_ONEHOT_ORDER = ("deep", "medium", "shallow")


@dataclass(frozen=True)
class StateKey:
    position: str
    active_players: str
    facing_bet: str
    raise_count: str
    spr: str

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
        fields = ("position", "active_players", "facing_bet", "raise_count", "spr")
        parts = s.split("|")
        if len(parts) != len(fields):
            raise ValueError(f"Invalid StateKey string: {s!r}")
        return StateKey(**dict(zip(fields, parts)))


def _spr_str(stack: float, pot: float) -> str:
    spr = (stack / pot) if pot > 0 else float("inf")
    if spr > 10:
        return "deep"
    if spr > 3:
        return "medium"
    return "shallow"


def _active_str(n: int) -> str:
    if n == 2:
        return "heads_up"
    if n == 3:
        return "three_way"
    return "multiway"


def _raise_count_str(n: int) -> str:
    if n == 0:
        return "raises_0"
    if n == 1:
        return "raises_1"
    return "raises_2plus"


def _seat_position(player: str, player_order: list[str]) -> str:
    idx = player_order.index(player) + 1
    return POSITIONS.get(idx, "unknown")


def state_key_from_parse_state(state, player: str) -> StateKey:
    num_active = sum(1 for _, alive in state.players_in_hand if alive)
    stack = state.current_stacks.get(player, 0.0)
    pot = state.pot_size

    history = state.betting_history or []
    raise_count = 0

    bb_player = (
        state.player_order[1]
        if len(state.player_order) > 1
        else state.player_order[0]
    )

    if history:
        raise_count = max(lvl for _, (_, lvl), _ in history)

        if raise_count > 0:
            facing_bet = True
        else:
            facing_bet = player != bb_player
    else:
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
    return StateKey(
        position=position,
        active_players=_active_str(num_active),
        facing_bet="facing_bet" if facing_bet else "no_bet",
        raise_count=_raise_count_str(raise_count),
        spr=_spr_str(stack, pot),
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _state_playability_adjustment(state_key: StateKey) -> float:
    pos_adj = {
        "sb": -0.02,
        "bb": 0.01,
        "utg": -0.08,
        "hj": -0.02,
        "co": 0.05,
        "btn": 0.08,
    }.get(state_key.position, 0.0)

    multiway_adj = {
        "heads_up": 0.04,
        "three_way": -0.02,
        "multiway": -0.06,
    }.get(state_key.active_players, 0.0)

    raise_adj = {
        "raises_0": 0.0,
        "raises_1": -0.12,
        "raises_2plus": -0.24,
    }.get(state_key.raise_count, 0.0)

    spr_adj = {
        "deep": 0.0,
        "medium": 0.03,
        "shallow": 0.07,
    }.get(state_key.spr, 0.0)

    facing_adj = -0.05 if state_key.facing_bet == "facing_bet" else 0.02

    return pos_adj + multiway_adj + raise_adj + spr_adj + facing_adj


def _playability(features: HandClassFeatures, state_key: StateKey) -> float:
    return _clamp(
        features.strength + _state_playability_adjustment(state_key),
        0.0,
        1.0,
    )


def _one_hot(value: str, order: Tuple[str, ...]) -> list[float]:
    return [1.0 if value == x else 0.0 for x in order]


def preflop_feature_vector(hand_class: str, state_key: StateKey | str) -> np.ndarray:
    """Fixed-length features for multinomial logit baseline P_preflop(a | h, s)."""
    if isinstance(state_key, str):
        state_key = StateKey.from_string(state_key)

    f = hand_class_features(hand_class)
    sk = state_key
    p = _playability(f, sk)
    facing = sk.facing_bet == "facing_bet"
    premium = float((f.pair and f.high >= 10) or (f.high == 14 and f.low >= 11))
    speculative = float(f.suited and f.gap <= 2 and f.low >= 5)

    pos_oh = _one_hot(sk.position if sk.position in _POSITION_ONEHOT_ORDER else "btn", _POSITION_ONEHOT_ORDER)
    act_oh = _one_hot(sk.active_players, _ACTIVE_ONEHOT_ORDER)
    raise_oh = _one_hot(sk.raise_count, _RAISE_ONEHOT_ORDER)
    spr_oh = _one_hot(sk.spr, _SPR_ONEHOT_ORDER)

    parts = [
        1.0,
        f.strength,
        f.high / 14.0,
        f.low / 14.0,
        float(f.pair),
        float(f.suited),
        f.gap / 12.0,
        f.broadways / 2.0,
        float(f.has_ace),
        p,
        premium,
        speculative,
        float(facing),
    ] + pos_oh + act_oh + raise_oh + spr_oh

    return np.asarray(parts, dtype=float)


PREFLOP_PHI_DIM = int(preflop_feature_vector("AKs", build_state_key("btn", 2, False, 0, 100.0, 1.5)).shape[0])


def legacy_preflop_logits(features: HandClassFeatures, state_key: StateKey) -> Dict[int, float]:
    """Previous closed-form baseline logits (used to ridge-fit HEURISTIC_BETA_PREFLOP)."""
    p = _playability(features, state_key)
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
    }.get(state_key.spr, 0.0)

    if facing:
        return {
            FOLD: 1.5 - 3.0 * p,
            CHECK_CALL: 0.2 + 1.4 * p + 0.15 * speculative,
            RAISE: -1.0 + (1.35 + spr_aggression) * p + 0.55 * premium,
        }

    return {
        FOLD: 1.1 - 2.4 * p,
        CHECK_CALL: 0.25 + 0.9 * p + 0.25 * speculative,
        RAISE: -0.55 + (1.15 + spr_aggression) * p + 0.35 * premium,
    }


def _ridge_fit_preflop_beta(
    n_samples: int = 5000,
    ridge: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hands = all_169_classes()
    rows: list[np.ndarray] = []
    targets: list[list[float]] = []

    for _ in range(n_samples):
        h = hands[int(rng.integers(len(hands)))]
        sk = StateKey(
            position=rng.choice(_POSITION_ONEHOT_ORDER),
            active_players=rng.choice(_ACTIVE_ONEHOT_ORDER),
            facing_bet=rng.choice(("facing_bet", "no_bet")),
            raise_count=rng.choice(_RAISE_ONEHOT_ORDER),
            spr=rng.choice(_SPR_ONEHOT_ORDER),
        )
        phi = preflop_feature_vector(h, sk)
        feat = hand_class_features(h)
        leg = legacy_preflop_logits(feat, sk)
        rows.append(phi)
        targets.append([leg[FOLD], leg[CHECK_CALL], leg[RAISE]])

    X = np.stack(rows)
    Y = np.array(targets)
    d = X.shape[1]
    xt_x = X.T @ X + ridge * np.eye(d)
    beta_dt = np.linalg.solve(xt_x, X.T @ Y)
    return beta_dt.T


HEURISTIC_BETA_PREFLOP: np.ndarray = _ridge_fit_preflop_beta()


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
    try:
        return OLD_TO_THREE_ACTION[action_bucket]
    except KeyError as exc:
        raise ValueError(f"Unknown action bucket: {action_bucket!r}") from exc


def canonical_preflop_action(action_bucket: int) -> int:
    return _canonical_action(action_bucket)


@dataclass
class PreflopPrior:
    """Multinomial-logit baseline × player/session tilt (same math as former GTOPrior)."""

    theta_pre: Sequence[float] | None = None
    floor: float = 0.01
    phi: float | None = None
    beta_preflop: Optional[np.ndarray] = None
    _beta_store: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.theta_pre is None:
            if self.phi is None:
                self.theta_pre = (0.0, 0.0, 0.0)
            else:
                self.theta_pre = (
                    -float(self.phi),
                    float(self.phi),
                    float(self.phi),
                )

        if len(self.theta_pre) != 3:
            raise ValueError("theta_pre must have length 3")

        self.theta_pre = tuple(float(x) for x in self.theta_pre)

        bf = self.beta_preflop if self.beta_preflop is not None else HEURISTIC_BETA_PREFLOP.copy()
        if bf.shape != (3, PREFLOP_PHI_DIM):
            raise ValueError(f"beta_preflop must be (3, {PREFLOP_PHI_DIM}), got {bf.shape}")
        self._beta_store = np.asarray(bf, dtype=float)

    @property
    def mode(self) -> PriorMode:
        return PriorMode.PREFLOP

    def _base_logits(self, hand_class: str, state_key: StateKey) -> Dict[int, float]:
        phi = preflop_feature_vector(hand_class, state_key)
        b = self._beta_store
        return {
            FOLD: float(b[0] @ phi),
            CHECK_CALL: float(b[1] @ phi),
            RAISE: float(b[2] @ phi),
        }

    def gto_action_probs(
        self,
        hand_class: str,
        state_key: StateKey | str,
    ) -> Dict[int, float]:
        if isinstance(state_key, str):
            state_key = StateKey.from_string(state_key)
        return softmax_dict_legacy(self._base_logits(hand_class, state_key), floor=self.floor)

    def action_utility_vectors(
        self,
        hand_class: str,
        state_key: StateKey | str,
    ) -> Dict[int, UtilityVector]:
        if isinstance(state_key, str):
            state_key = StateKey.from_string(state_key)

        p_base = self.gto_action_probs(hand_class, state_key)
        return {
            a: _tendency_deviation_generic(
                a,
                p_base,
                fold=FOLD,
                call=CHECK_CALL,
                raise_=RAISE,
            )
            for a in ACTION_BUCKETS
        }

    def action_probs_with_theta(
        self,
        hand_class: str,
        state_key: StateKey | str,
        theta_pre: Sequence[float],
    ) -> Dict[int, float]:
        tilted = PreflopPrior(theta_pre=theta_pre, floor=self.floor, phi=None, beta_preflop=self._beta_store.copy())
        return tilted.action_probs(hand_class, state_key)

    def action_probs(
        self,
        hand_class: str,
        state_key: StateKey | str,
    ) -> Dict[int, float]:
        if isinstance(state_key, str):
            state_key = StateKey.from_string(state_key)

        gto_probs = self.gto_action_probs(hand_class, state_key)

        modulated_scores = {
            a: log(gto_probs[a])
            + dot3(
                self.theta_pre,
                _tendency_deviation_generic(
                    a,
                    gto_probs,
                    fold=FOLD,
                    call=CHECK_CALL,
                    raise_=RAISE,
                ),
            )
            for a in ACTION_BUCKETS
        }

        return softmax_dict_legacy(modulated_scores, floor=self.floor)

    def action_probability(
        self,
        hand_class: str,
        state_key: StateKey | str,
        action_bucket: int,
    ) -> float:
        action = _canonical_action(action_bucket)
        return self.action_probs(hand_class, state_key)[action]

    def full_table_for_state(
        self,
        state_key: StateKey | str,
    ) -> Dict[str, Dict[int, float]]:
        return {
            h: self.action_probs(h, state_key)
            for h in all_169_classes()
        }


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
    return np.stack([preflop_feature_vector(h, sk) for h, sk in pairs], axis=0)


def fit_heuristic_preflop_prior() -> PreflopPrior:
    return PreflopPrior()


GTOPrior = PreflopPrior
