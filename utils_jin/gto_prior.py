"""Simple GTO-inspired pre-flop action prior.

This is not a solver output.  It is a deliberately transparent heuristic that
plays the role of ``p_GTO(a | h, s)`` in the slides.  The important design
choice is the interface:

    hand class + compact state key -> probability over action buckets

Later, this module can be replaced by real solver frequencies or a better
pre-flop chart without changing the filter or EM code.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from .action_encoder import (
    CHECK_CALL,
    FOLD,
    LARGE_BET_RAISE,
    MEDIUM_BET_RAISE,
    SMALL_BET_RAISE,
)
from .cards import RANK_TO_VALUE, all_169_classes
from .state_encoder import StateKey

ACTION_BUCKETS = (FOLD, CHECK_CALL, SMALL_BET_RAISE, MEDIUM_BET_RAISE, LARGE_BET_RAISE)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _softmax(scores: dict[int, float], floor: float) -> dict[int, float]:
    """Convert scores to probabilities with a small exploration floor."""

    max_score = max(scores.values())
    weights = {action: exp(score - max_score) for action, score in scores.items()}
    total = sum(weights.values())
    probs = {action: weight / total for action, weight in weights.items()}

    # The floor keeps all likelihoods nonzero.  That matters for filtering and
    # EM because a zero likelihood would permanently delete a hand class.
    floored = {action: (1.0 - floor * len(probs)) * prob + floor for action, prob in probs.items()}
    norm = sum(floored.values())
    return {action: prob / norm for action, prob in floored.items()}


@dataclass(frozen=True)
class HandClassFeatures:
    """Small feature set used by the heuristic prior."""

    high: int
    low: int
    pair: bool
    suited: bool
    gap: int
    broadways: int
    has_ace: bool
    strength: float


def hand_class_features(hand_class: str) -> HandClassFeatures:
    """Compute transparent pre-flop features for a 169 hand class."""

    if len(hand_class) == 2:
        high = low = RANK_TO_VALUE[hand_class[0]]
        pair = True
        suited = False
    elif len(hand_class) == 3:
        high = RANK_TO_VALUE[hand_class[0]]
        low = RANK_TO_VALUE[hand_class[1]]
        pair = False
        suited = hand_class[2] == "s"
    else:
        raise ValueError(f"Invalid hand class: {hand_class!r}")

    gap = max(0, high - low - 1)
    broadways = int(high >= 10) + int(low >= 10)
    has_ace = high == 14 or low == 14

    if pair:
        # Pairs are strong because they are already made hands pre-flop.
        strength = 0.52 + 0.45 * ((high - 2) / 12)
    else:
        high_component = 0.45 * ((high - 2) / 12)
        low_component = 0.25 * ((low - 2) / 12)
        suited_bonus = 0.08 if suited else 0.0
        connected_bonus = 0.07 if gap <= 1 else 0.03 if gap == 2 else 0.0
        broadway_bonus = 0.04 * broadways
        ace_bonus = 0.05 if has_ace else 0.0
        gap_penalty = 0.025 * max(0, gap - 2)
        strength = 0.10 + high_component + low_component + suited_bonus
        strength += connected_bonus + broadway_bonus + ace_bonus - gap_penalty

    return HandClassFeatures(
        high=high,
        low=low,
        pair=pair,
        suited=suited,
        gap=gap,
        broadways=broadways,
        has_ace=has_ace,
        strength=_clamp(strength, 0.01, 0.99),
    )


def _get_state_field(state_key: StateKey | str, field: str) -> str:
    """Read a field from either a ``StateKey`` object or its string form."""

    if isinstance(state_key, StateKey):
        return getattr(state_key, field)

    parts = state_key.split("|")
    fields = ("street", "position", "active_players", "facing_bet", "raise_count", "spr", "board_texture")
    if len(parts) != len(fields):
        raise ValueError(f"Invalid StateKey string: {state_key!r}")
    return dict(zip(fields, parts))[field]


@dataclass
class PreflopGTOPrior:
    """Heuristic ``p_GTO(a | hand_class, state_key)`` provider.

    Parameters
    ----------
    floor:
        Minimum probability mixed into every action.  Keep this positive.
    aggression:
        Multiplier on raise scores.  This is not the learned ``theta`` yet, but
        it gives a single knob for sensitivity checks.
    tightness:
        Positive values make the chart fold more marginal hands.
    """

    floor: float = 0.01
    aggression: float = 1.0
    tightness: float = 0.0

    def action_probs(self, hand_class: str, state_key: StateKey | str) -> dict[int, float]:
        """Return probabilities for all action buckets."""

        features = hand_class_features(hand_class)
        position = _get_state_field(state_key, "position")
        facing_bet = _get_state_field(state_key, "facing_bet") == "facing_bet"
        raise_count = _get_state_field(state_key, "raise_count")
        active_players = _get_state_field(state_key, "active_players")

        # Later position can profitably open/continue wider.  Early position is
        # intentionally tighter.
        position_adjustment = {
            "sb": -0.02,
            "bb": 0.01,
            "early": -0.08,
            "middle": -0.02,
            "late": 0.05,
            "button_or_late": 0.08,
        }.get(position, 0.0)

        multiway_penalty = {
            "heads_up": 0.04,
            "three_way": -0.02,
            "multiway": -0.06,
        }.get(active_players, 0.0)

        raise_penalty = {
            "raises_0": 0.0,
            "raises_1": -0.12,
            "raises_2plus": -0.24,
        }.get(raise_count, 0.0)

        playable = features.strength + position_adjustment + multiway_penalty + raise_penalty - self.tightness
        playable = _clamp(playable, 0.0, 1.0)

        premium = int(features.pair and features.high >= 11) or int(features.high == 14 and features.low >= 12)
        speculative = int(features.suited and features.gap <= 2 and features.low >= 5)

        if facing_bet:
            # Facing a bet: weak hands fold, medium hands call, premium hands
            # raise.  This is the key behavior the range filter needs.
            scores = {
                FOLD: 1.5 - 3.0 * playable,
                CHECK_CALL: 0.2 + 1.4 * playable + 0.15 * speculative,
                SMALL_BET_RAISE: -0.8 + self.aggression * (1.1 * playable + 0.25 * premium),
                MEDIUM_BET_RAISE: -1.0 + self.aggression * (1.4 * playable + 0.50 * premium),
                LARGE_BET_RAISE: -1.6 + self.aggression * (1.2 * playable + 0.75 * premium),
            }
        else:
            # Not facing a bet: fold is mostly open-folding/small blind folding,
            # check_call represents check/limp/call-like passive continuations,
            # and raises are opens.
            scores = {
                FOLD: 1.1 - 2.4 * playable,
                CHECK_CALL: 0.25 + 0.9 * playable + 0.25 * speculative,
                SMALL_BET_RAISE: -0.35 + self.aggression * (1.0 * playable),
                MEDIUM_BET_RAISE: -0.55 + self.aggression * (1.2 * playable + 0.25 * premium),
                LARGE_BET_RAISE: -1.6 + self.aggression * (0.8 * playable + 0.55 * premium),
            }

        return _softmax(scores, floor=self.floor)

    def action_probability(self, hand_class: str, state_key: StateKey | str, action_bucket: int) -> float:
        """Convenience accessor for a single likelihood value."""

        return self.action_probs(hand_class, state_key)[action_bucket]

    def dirichlet_alpha(self, hand_class: str, state_key: StateKey | str, kappa: float) -> dict[int, float]:
        """Return prior pseudo-counts ``alpha = kappa * p_GTO`` for EM/MAP."""

        probs = self.action_probs(hand_class, state_key)
        return {action: kappa * prob for action, prob in probs.items()}

    def full_table_for_state(self, state_key: StateKey | str) -> dict[str, dict[int, float]]:
        """Materialize probabilities for every 169 hand class at one state."""

        return {hand_class: self.action_probs(hand_class, state_key) for hand_class in all_169_classes()}

