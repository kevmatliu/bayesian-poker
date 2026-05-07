"""Shared card / rank primitives for strength coding (preflop & postflop)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List


class StrengthMode(Enum):
    """Which strength feature family applies."""

    PREFLOP = "preflop"
    POSTFLOP = "postflop"


RANK_TO_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
VALUE_TO_RANK = {v: k for k, v in RANK_TO_VALUE.items()}


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    @property
    def value(self) -> int:
        return RANK_TO_VALUE[self.rank]


def parse_card(card: str) -> Card:
    card = card.strip().upper()
    if len(card) != 2:
        raise ValueError(f"Invalid card: {card}")
    rank, suit = card[0], card[1]
    if rank not in RANK_TO_VALUE:
        raise ValueError(f"Invalid rank: {rank}")
    if suit not in {"S", "H", "D", "C"}:
        raise ValueError(f"Invalid suit: {suit}")
    return Card(rank, suit)


def parse_cards(cards: Iterable[str]) -> List[Card]:
    parsed = [parse_card(c) if isinstance(c, str) else c for c in cards]
    if len(set(parsed)) != len(parsed):
        raise ValueError("Duplicate cards detected.")
    return parsed


def all_52_cards() -> List[Card]:
    """Full deck in deterministic order."""
    return [Card(r, s) for s in "SHDC" for r in "23456789TJQKA"]
