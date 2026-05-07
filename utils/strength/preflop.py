"""Preflop hole-card taxonomy and heuristic strength encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from utils.strength.common import Card, RANK_TO_VALUE, StrengthMode, VALUE_TO_RANK

MODE = StrengthMode.PREFLOP


@dataclass(frozen=True)
class HandClassFeatures:
    high: int
    low: int
    pair: bool
    suited: bool
    gap: int
    broadways: int
    has_ace: bool
    strength: float


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def hand_class_features(hand_class: str) -> HandClassFeatures:
    """Encode a 169-style label (e.g. ``AKs``, ``TT``) into features + heuristic strength."""
    if len(hand_class) == 2:
        high = low = RANK_TO_VALUE[hand_class[0]]
        pair, suited = True, False
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
        strength = 0.52 + 0.45 * ((high - 2) / 12)
    else:
        connector_bonus = (
            0.10
            if gap == 0
            else 0.07
            if gap == 1
            else 0.03
            if gap == 2
            else 0.0
        )

        strength = (
            0.10
            + 0.45 * ((high - 2) / 12)
            + 0.25 * ((low - 2) / 12)
            + (0.08 if suited else 0.0)
            + connector_bonus
            + 0.04 * broadways
            + (0.05 if has_ace else 0.0)
            - 0.025 * max(0, gap - 2)
        )

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


def get_equivalence_class(cards: List[Card]) -> str:
    """Map a 2-card holding to its 169 equivalence-class string."""
    if len(cards) != 2:
        raise ValueError("Equivalence classes are only defined for 2-card hands.")

    c1, c2 = sorted(cards, key=lambda c: c.value, reverse=True)
    suited = c1.suit == c2.suit
    rank1 = VALUE_TO_RANK[c1.value]
    rank2 = VALUE_TO_RANK[c2.value]

    if rank1 == rank2:
        return f"{rank1}{rank2}"
    if suited:
        return f"{rank1}{rank2}s"
    return f"{rank1}{rank2}o"


def all_169_classes() -> List[str]:
    """All canonical preflop hand labels in a fixed order."""
    ranks = "AKQJT98765432"
    classes: List[str] = []
    for i in range(len(ranks)):
        for j in range(i, len(ranks)):
            r1, r2 = ranks[i], ranks[j]
            if i == j:
                classes.append(f"{r1}{r2}")
            else:
                classes.append(f"{r1}{r2}s")
                classes.append(f"{r1}{r2}o")

    return classes
