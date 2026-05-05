from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from utils.hand_map import all_169_classes

STRENGTH_BUCKETS = [
    "nuts/near-nuts",
    "strong made",
    "medium made",
    "weak made",
    "strong draw",
    "weak draw",
    "air",
]
STRENGTH_INDEX = {w: i for i, w in enumerate(STRENGTH_BUCKETS)}

_COMBO_COUNT = {
    "pair":    6,   # e.g. AA
    "suited":  4,   # e.g. AKs
    "offsuit": 12,  # e.g. AKo
}


def _combo_count(hand_class: str) -> int:
    if len(hand_class) == 2:
        return _COMBO_COUNT["pair"]
    return _COMBO_COUNT["suited"] if hand_class.endswith("s") else _COMBO_COUNT["offsuit"]


def _dead_card_set(dead_cards: str = "") -> set[str]:
    return {
        dead_cards[i:i + 2].upper()
        for i in range(0, len(dead_cards), 2)
        if dead_cards[i:i + 2]
    }


def available_combo_count(hand_class: str, dead_cards: str = "") -> int:
    dead = _dead_card_set(dead_cards)

    if len(hand_class) == 2:
        rank = hand_class[0]
        available_cards = [f"{rank}{suit}" for suit in _SUIT_CHARS if f"{rank}{suit}" not in dead]
        n = len(available_cards)
        return (n * (n - 1)) // 2

    rank1, rank2 = hand_class[0], hand_class[1]
    suited = hand_class.endswith("s")

    if suited:
        return sum(
            1
            for suit in _SUIT_CHARS
            if f"{rank1}{suit}" not in dead and f"{rank2}{suit}" not in dead
        )

    return sum(
        1
        for suit1 in _SUIT_CHARS
        for suit2 in _SUIT_CHARS
        if suit1 != suit2
        and f"{rank1}{suit1}" not in dead
        and f"{rank2}{suit2}" not in dead
    )


def initial_class_prior(dead_cards: str = "") -> Dict[str, float]:
    """Uniform-over-combos prior conditioned on any known dead cards."""
    classes = all_169_classes()
    counts = {hand_class: available_combo_count(hand_class, dead_cards) for hand_class in classes}
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Cannot build an initial prior with zero available combos.")
    return {hand_class: count / total for hand_class, count in counts.items()}


def normalize(d: Dict) -> Dict:
    total = sum(d.values())
    if total <= 0:
        raise ValueError("Cannot normalize a zero distribution.")
    return {k: v / total for k, v in d.items()}


def effective_sample_size(distribution: Dict) -> float:
    """ESS = 1 / sum(p_i^2)."""
    denom = sum(p * p for p in distribution.values())
    return 1.0 / denom if denom > 0 else 0.0


@dataclass
class FilterStep:
    state_key: str
    action_bucket: int
    evidence: float          # sum of unnormalized weights (marginal likelihood)
    ess: float               # effective sample size after update
    top_class: str           # highest-probability class / bucket
    top_prob: float
    layer: str               # "preflop" | "postflop"


_SUIT_CHARS = "SHDC"


def sample_combo_for_class(hand_class: str, board_cards: str) -> Optional[str]:
    """Return a representative concrete 2-card combo for a class, avoiding blockers."""
    board_set = {
        board_cards[i:i + 2].upper()
        for i in range(0, len(board_cards), 2)
    }

    if len(hand_class) == 2:
        rank = hand_class[0]
        suits = list(_SUIT_CHARS)
        for i in range(len(suits)):
            for j in range(i + 1, len(suits)):
                c1 = f"{rank}{suits[i]}"
                c2 = f"{rank}{suits[j]}"
                if c1 not in board_set and c2 not in board_set:
                    return c1 + c2
        return None

    rank1, rank2 = hand_class[0], hand_class[1]
    suited = hand_class.endswith("s")

    if suited:
        for suit in _SUIT_CHARS:
            c1 = f"{rank1}{suit}"
            c2 = f"{rank2}{suit}"
            if c1 not in board_set and c2 not in board_set:
                return c1 + c2
        return None

    for suit1 in _SUIT_CHARS:
        for suit2 in _SUIT_CHARS:
            if suit1 == suit2:
                continue
            c1 = f"{rank1}{suit1}"
            c2 = f"{rank2}{suit2}"
            if c1 not in board_set and c2 not in board_set:
                return c1 + c2
    return None
