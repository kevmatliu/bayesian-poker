"""GTO-inspired pre-flop action prior.

p_GTO(a | h, s) gives the baseline solver frequency for action a given
hand class h and state s.  This is the anchor for the range filter.

phi is the *temperature* of the range: it controls how wide or tight a
player deviates from GTO.  phi > 0 means wider (looser) than GTO;
phi < 0 means tighter (more nit-like).  phi = 0 recovers the pure GTO
prior.  Formally,

    P(a | h, s, phi) ∝ p_GTO(a | h, s) * exp(phi * u(h, s, a))

where u(h, s, a) is a hand-state-action utility that encodes whether a
is a "wider" action (call/raise with weak hands) or a "tighter" action
(fold with marginal hands).  phi is learned by EM across hands in a
session.

StateKey is constructed directly from parse.State objects via
``state_key_from_parse_state``.  The raw ``build_state_key`` helper is
still available for synthetic / test usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict, Optional

from utils.hand_map import RANK_TO_VALUE, all_169_classes

# ── action bucket constants (mirrors action_map.py) ──────────────────────────
FOLD       = 0
CHECK_CALL = 1
CBR_SMALL  = 2
CBR_MEDIUM = 3
CBR_LARGE  = 4

ACTION_BUCKETS = (FOLD, CHECK_CALL, CBR_SMALL, CBR_MEDIUM, CBR_LARGE)

POSITIONS = {
    1: "sb",
    2: "bb",
    3: "utg",
    4: "hj",
    5: "co",
    6: "btn",
}


# ── state key ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StateKey:
    """Compact encoding of the public for pre-flop

    position        : sb | bb | utg | hj | co | btn
    active_players  : heads_up | three_way | multiway
    facing_bet      : facing_bet | no_bet
    raise_count     : raises_0 | raises_1 | raises_2plus
    spr             : deep (>10) | medium (3-10) | shallow (<3) # even though stacks reset to 10000 every time
    """
    position       : str
    active_players : str
    facing_bet     : str
    raise_count    : str
    spr            : str

    def as_string(self) -> str:
        return "|".join([
            self.position, self.active_players, self.facing_bet, self.raise_count, self.spr
        ])

    @staticmethod
    def from_string(s: str) -> "StateKey":
        fields = ("position", "active_players", "facing_bet", "raise_count", "spr")
        parts = s.split("|")
        if len(parts) != len(fields):
            raise ValueError(f"Invalid StateKey string: {s!r}")
        return StateKey(**dict(zip(fields, parts)))


def _spr_str(stack: float, pot: float) -> str:
    spr = (stack / pot) if pot > 0 else float("inf")
    if spr > 10: return "deep"
    if spr > 3:  return "medium"
    return "shallow"

def _active_str(n: int) -> str:
    if n == 2: return "heads_up"
    if n == 3: return "three_way"
    return "multiway"

def _raise_count_str(n: int) -> str:
    if n == 0: return "raises_0"
    if n == 1: return "raises_1"
    return "raises_2plus"


def _seat_position(player: str, player_order: list[str]) -> str:
    idx = player_order.index(player) + 1
    return POSITIONS.get(idx, "unknown")

# ── primary builder: from parse.State ─────────────────────────────────────────

def state_key_from_parse_state(
    state      ,      # parse.State — the snapshot *before* the player acts
    player     : str, # player name
) -> StateKey:
    """
    Building StateKey from parse.State.
    """
    num_active = sum(1 for _, alive in state.players_in_hand if alive)
    stack = state.current_stacks.get(player, 0.0)
    pot = state.pot_size

    history = state.betting_history   # may be empty list
    raise_count = 0
    facing_bet = False
    bb_player = state.player_order[1] if len(state.player_order) > 1 else state.player_order[0]

    if history:
        raise_count = max(lvl for _, (_, lvl), _ in history)
        if raise_count > 0:
            facing_bet = True   # at least one cbr
        else:
            facing_bet = (player != bb_player)
    else:
        facing_bet = (player != bb_player)

    return StateKey(
        position=_seat_position(player, state.player_order),
        active_players=_active_str(num_active),
        facing_bet="facing_bet" if facing_bet else "no_bet",
        raise_count=_raise_count_str(raise_count),
        spr=_spr_str(stack, pot),
    )


def build_state_key(
    position    : str,
    num_active  : int,
    facing_bet  : bool,
    raise_count : int,
    stack       : float,
    pot         : float,
) -> StateKey:
    """Construct a StateKey from raw scalar values (tests / synthetic data)."""
    return StateKey(
        position       = position,
        active_players = _active_str(num_active),
        facing_bet     = "facing_bet" if facing_bet else "no_bet",
        raise_count    = _raise_count_str(raise_count),
        spr            = _spr_str(stack, pot),
    )


# ── hand features ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HandClassFeatures:
    high      : int
    low       : int
    pair      : bool
    suited    : bool
    gap       : int
    broadways : int
    has_ace   : bool
    strength  : float   # in [0, 1], baseline pre-flop equity proxy


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def hand_class_features(hand_class: str) -> HandClassFeatures:
    if len(hand_class) == 2:
        high = low = RANK_TO_VALUE[hand_class[0]]
        pair, suited = True, False
    elif len(hand_class) == 3:
        high   = RANK_TO_VALUE[hand_class[0]]
        low    = RANK_TO_VALUE[hand_class[1]]
        pair   = False
        suited = hand_class[2] == "s"
    else:
        raise ValueError(f"Invalid hand class: {hand_class!r}")

    gap       = max(0, high - low - 1)
    broadways = int(high >= 10) + int(low >= 10)
    has_ace   = high == 14 or low == 14

    if pair:
        strength = 0.52 + 0.45 * ((high - 2) / 12)
    else:
        strength = (
            0.10
            + 0.45 * ((high - 2) / 12)
            + 0.25 * ((low - 2) / 12)
            + (0.08 if suited else 0.0)
            + (0.07 if gap <= 1 else 0.03 if gap == 2 else 0.0)
            + 0.04 * broadways
            + (0.05 if has_ace else 0.0)
            - 0.025 * max(0, gap - 2)
        )

    return HandClassFeatures(
        high=high, low=low, pair=pair, suited=suited,
        gap=gap, broadways=broadways, has_ace=has_ace,
        strength=_clamp(strength, 0.01, 0.99),
    )


# ── utility u(h, s, a) ────────────────────────────────────────────────────────

def _action_utility(features: HandClassFeatures, action_bucket: int) -> float:
    """
    Utility of action relative to GTO baseline.

    Positive utility means wider than GTO (looser, calling with weaker hand)
    Negative utility means tighter than GTO (folding a marginal hand that GTO would call).

    phi scales this utility in the likelihood.
    """
    s = features.strength

    if action_bucket == FOLD:
        return -(s - 0.5) * 2.0

    if action_bucket == CHECK_CALL:
        return (0.5 - s) * 1.5

    if action_bucket in (CBR_SMALL, CBR_MEDIUM, CBR_LARGE):
        size_factor = {CBR_SMALL: 0.8, CBR_MEDIUM: 1.2, CBR_LARGE: 1.8}[action_bucket]
        return size_factor * (0.6 - s)

    return 0.0


# ── softmax helper ────────────────────────────────────────────────────────────

def _softmax(scores: Dict[int, float], floor: float = 0.01) -> Dict[int, float]:
    mx      = max(scores.values())
    w       = {a: exp(v - mx) for a, v in scores.items()}
    total   = sum(w.values())
    probs   = {a: v / total for a, v in w.items()}
    floored = {a: (1.0 - floor * len(probs)) * p + floor for a, p in probs.items()}
    norm    = sum(floored.values())
    return {a: p / norm for a, p in floored.items()}


@dataclass
class GTOPrior:
    """
    p_GTO(a | h, s) with phi-temperature deviation from GTO.

    phi: temperature scalar encoding how wide (phi > 0) or tight (phi < 0) based on GTO heuristics
    floor : clipping the likelihood
    """
    phi  : float = 0.0
    floor: float = 0.01

    def _gto_scores(
        self,
        features : HandClassFeatures,
        state_key: StateKey,
    ) -> Dict[int, float]:
        s      = features.strength
        facing = (state_key.facing_bet == "facing_bet")

        pos_adj = {
            "sb": -0.02, "bb": 0.01, "utg": -0.08,
            "hj": -0.02, "co": 0.05, "btn": 0.08,
        }.get(state_key.position, 0.0)

        multiway_adj = {
            "heads_up": 0.04, "three_way": -0.02, "multiway": -0.06,
        }.get(state_key.active_players, 0.0)

        raise_adj = {
            "raises_0": 0.0, "raises_1": -0.12, "raises_2plus": -0.24,
        }.get(state_key.raise_count, 0.0)

        # Shallow SPR → more all-in pressure → effective raise bonus
        spr_aggression = {
            "deep": 0.0, "medium": 0.04, "shallow": 0.10,
        }.get(state_key.spr, 0.0)

        playable = _clamp(s + pos_adj + multiway_adj + raise_adj, 0.0, 1.0)
        premium = int(features.pair and features.high >= 10) or int(
            features.high == 14 and features.low >= 11
        )
        speculative = int(features.suited and features.gap <= 2 and features.low >= 5)

        if facing:
            return {    # penalizing fold more, rewarding call and small cbr more when facing a bet
                FOLD: 1.5 - 3.0 * playable,
                CHECK_CALL: 0.2 + 1.4 * playable + 0.15 * speculative,
                CBR_SMALL : -0.8 + (1.1 + spr_aggression) * playable + 0.25 * premium,
                CBR_MEDIUM: -1.0 + (1.4 + spr_aggression) * playable + 0.50 * premium,
                CBR_LARGE : -1.6 + (1.2 + spr_aggression) * playable + 0.75 * premium,
            }
        else:
            return {    # can be more aggressive when not facing a bet, especially with strong hands
                FOLD: 1.1 - 2.4 * playable,
                CHECK_CALL: 0.25 + 0.9 * playable + 0.25 * speculative,
                CBR_SMALL: -0.35 + (1.0 + spr_aggression) * playable,
                CBR_MEDIUM: -0.55 + (1.2 + spr_aggression) * playable + 0.25 * premium,
                CBR_LARGE: -1.6  + (0.8 + spr_aggression) * playable + 0.55 * premium,
            }

    def action_probs(
        self,
        hand_class: str,
        state_key : StateKey | str,
    ) -> Dict[int, float]:
        """P(a | h, s, phi) ∝ p_GTO(a | h, s) * exp(phi * u(h, s, a))."""
        if isinstance(state_key, str):
            state_key = StateKey.from_string(state_key)
        features   = hand_class_features(hand_class)
        gto_scores = self._gto_scores(features, state_key)
        modulated  = {
            a: score + self.phi * _action_utility(features, a)
            for a, score in gto_scores.items()
        }
        return _softmax(modulated, floor=self.floor)

    def action_probability(
        self,
        hand_class   : str,
        state_key    : StateKey | str,
        action_bucket: int,
    ) -> float:
        return self.action_probs(hand_class, state_key)[action_bucket]

    def dirichlet_alpha(
        self,
        hand_class: str,
        state_key : StateKey | str,
        kappa     : float = 5.0,
    ) -> Dict[int, float]:
        """Dirichlet pseudo-counts alpha = kappa * p_GTO for EM/MAP."""
        probs = self.action_probs(hand_class, state_key)
        return {a: kappa * p for a, p in probs.items()}

    def full_table_for_state(
        self, state_key: StateKey | str,
    ) -> Dict[str, Dict[int, float]]:
        """Materialise probabilities for all 169 hand classes at one state."""
        return {h: self.action_probs(h, state_key) for h in all_169_classes()}
