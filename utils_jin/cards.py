"""Card, combo, and 169-class utilities.

The model is easiest to reason about with two representations:

* 1326 actual two-card combos, e.g. ``("As", "Kd")``.
* 169 pre-flop equivalence classes, e.g. ``AKs``, ``AKo``, ``AA``.

The filter in this first implementation tracks 169 classes, because that is the
state space in the slides.  We still expose 1326-combo helpers because board
blockers and post-flop mapping need exact cards later.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Iterable

RANKS_LOW_TO_HIGH = "23456789TJQKA"
RANKS_HIGH_TO_LOW = tuple(reversed(RANKS_LOW_TO_HIGH))
SUITS = "shdc"

RANK_TO_VALUE = {rank: i + 2 for i, rank in enumerate(RANKS_LOW_TO_HIGH)}
VALUE_TO_RANK = {value: rank for rank, value in RANK_TO_VALUE.items()}


def normalize_card(card: str) -> str:
    """Normalize a card string to rank-uppercase/suit-lowercase, e.g. ``As``."""

    stripped = card.strip()
    if len(stripped) != 2:
        raise ValueError(f"Invalid card string: {card!r}")

    rank = stripped[0].upper()
    suit = stripped[1].lower()
    if rank not in RANK_TO_VALUE:
        raise ValueError(f"Invalid rank in card: {card!r}")
    if suit not in SUITS:
        raise ValueError(f"Invalid suit in card: {card!r}")
    return rank + suit


def split_cards(cards: str | Iterable[str]) -> list[str]:
    """Split a compact string like ``AsKd`` into normalized card strings."""

    if isinstance(cards, str):
        compact = cards.strip()
        if compact == "":
            return []
        if len(compact) % 2 != 0:
            raise ValueError(f"Invalid compact card string: {cards!r}")
        return [normalize_card(compact[i : i + 2]) for i in range(0, len(compact), 2)]

    return [normalize_card(card) for card in cards]


def card_rank(card: str) -> str:
    return normalize_card(card)[0]


def card_suit(card: str) -> str:
    return normalize_card(card)[1]


def rank_value(rank_or_card: str) -> int:
    """Return numeric rank value for a rank or card string."""

    rank = rank_or_card[0].upper()
    return RANK_TO_VALUE[rank]


def full_deck() -> tuple[str, ...]:
    """Return all 52 cards in a stable order."""

    return tuple(rank + suit for rank in RANKS_HIGH_TO_LOW for suit in SUITS)


def sort_combo(cards: Iterable[str]) -> tuple[str, str]:
    """Return a canonical two-card combo sorted by rank then suit."""

    normalized = split_cards(cards)
    if len(normalized) != 2:
        raise ValueError(f"Expected exactly two cards, got {normalized!r}")
    if normalized[0] == normalized[1]:
        raise ValueError(f"Duplicate card in combo: {normalized!r}")
    return tuple(
        sorted(normalized, key=lambda c: (-rank_value(c), SUITS.index(card_suit(c))))
    )  # type: ignore[return-value]


def combo_to_class(combo: str | Iterable[str]) -> str:
    """Map an exact two-card combo to a 169 pre-flop class.

    Examples:
    ``["As", "Ah"] -> "AA"``, ``["As", "Ks"] -> "AKs"``,
    ``["As", "Kd"] -> "AKo"``.
    """

    cards = sort_combo(split_cards(combo) if isinstance(combo, str) else combo)
    r1, r2 = card_rank(cards[0]), card_rank(cards[1])
    if r1 == r2:
        return r1 + r2

    suffix = "s" if card_suit(cards[0]) == card_suit(cards[1]) else "o"
    return r1 + r2 + suffix


def all_combos() -> tuple[tuple[str, str], ...]:
    """Return all 1326 exact two-card combos."""

    return tuple(sort_combo(combo) for combo in combinations(full_deck(), 2))


def all_169_classes() -> tuple[str, ...]:
    """Return the 169 pre-flop classes in a stable high-card-first order."""

    classes: list[str] = []
    ranks = list(RANKS_HIGH_TO_LOW)
    for i, high in enumerate(ranks):
        for low in ranks[i:]:
            if high == low:
                classes.append(high + low)
            else:
                classes.append(high + low + "s")
                classes.append(high + low + "o")
    return tuple(classes)


def class_multiplicity(hand_class: str) -> int:
    """Return how many exact combos belong to a 169 class."""

    if len(hand_class) == 2:
        return 6
    if hand_class.endswith("s"):
        return 4
    if hand_class.endswith("o"):
        return 12
    raise ValueError(f"Invalid hand class: {hand_class!r}")


def class_multiplicities() -> dict[str, int]:
    return {hand_class: class_multiplicity(hand_class) for hand_class in all_169_classes()}


def initial_class_prior() -> dict[str, float]:
    """Return the no-card-information prior over 169 classes.

    This matches the slide formula ``C(h) / 1326``.
    """

    return {
        hand_class: class_multiplicity(hand_class) / 1326.0
        for hand_class in all_169_classes()
    }


def initial_combo_prior() -> dict[tuple[str, str], float]:
    """Return a uniform prior over exact two-card combos."""

    combos = all_combos()
    prob = 1.0 / len(combos)
    return {combo: prob for combo in combos}


def legal_combos(dead_cards: str | Iterable[str] = ()) -> tuple[tuple[str, str], ...]:
    """Return exact combos that do not contain any dead card.

    Dead cards include known board cards and any known private cards that should
    be blocked from an opponent range.
    """

    dead = set(split_cards(dead_cards))
    return tuple(combo for combo in all_combos() if not dead.intersection(combo))


def combos_by_class(dead_cards: str | Iterable[str] = ()) -> dict[str, list[tuple[str, str]]]:
    """Group legal exact combos by 169 class."""

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for combo in legal_combos(dead_cards):
        grouped[combo_to_class(combo)].append(combo)
    return dict(grouped)


def aggregate_combo_probs(combo_probs: dict[tuple[str, str], float]) -> dict[str, float]:
    """Aggregate exact-combo probabilities into 169-class probabilities."""

    out = {hand_class: 0.0 for hand_class in all_169_classes()}
    for combo, prob in combo_probs.items():
        out[combo_to_class(combo)] += prob
    return out


def normalize_distribution(dist: dict[str, float]) -> dict[str, float]:
    """Normalize a nonnegative dictionary distribution."""

    total = sum(dist.values())
    if total <= 0:
        raise ValueError("Cannot normalize a distribution with non-positive mass.")
    return {key: value / total for key, value in dist.items()}


def count_classes(combos: Iterable[tuple[str, str]]) -> Counter[str]:
    """Count how many exact combos from an iterable fall into each 169 class."""

    return Counter(combo_to_class(combo) for combo in combos)

