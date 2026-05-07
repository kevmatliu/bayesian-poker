"""Postflop hand strength: board-aware made/draw features and ``poker_hand_mapper``."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, List, Set, Tuple

from utils.strength.common import Card, StrengthMode, all_52_cards, parse_cards

MODE = StrengthMode.POSTFLOP


def is_straight(values: List[int]) -> Tuple[bool, int]:
    """
    Returns (is_straight, high_card_of_straight)
    Handles wheel: A-2-3-4-5
    """
    unique = sorted(set(values), reverse=True)
    if len(unique) < 5:
        return False, 0

    for i in range(len(unique) - 4):
        window = unique[i : i + 5]
        if window[0] - window[4] == 4 and len(window) == 5:
            return True, window[0]

    if {14, 5, 4, 3, 2}.issubset(set(unique)):
        return True, 5

    return False, 0


def _remaining_deck(dead: Iterable[Card]) -> List[Card]:
    dset = set(dead)
    return [c for c in all_52_cards() if c not in dset]


def evaluate_5(cards: List[Card]) -> Tuple[int, List[int], str]:
    """
    Higher tuple is better.
    category ranks:
      8 straight flush
      7 quads
      6 full house
      5 flush
      4 straight
      3 trips
      2 two pair
      1 pair
      0 high card
    """
    values = sorted((c.value for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    counts = Counter(values)

    is_flush = len(set(suits)) == 1
    straight, straight_high = is_straight(values)

    if is_flush and straight:
        return 8, [straight_high], "straight_flush"

    count_groups = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))
    freqs = sorted(counts.values(), reverse=True)

    if freqs == [4, 1]:
        quad = count_groups[0][0]
        kicker = count_groups[1][0]
        return 7, [quad, kicker], "quads"

    if freqs == [3, 2]:
        trips = count_groups[0][0]
        pair = count_groups[1][0]
        return 6, [trips, pair], "full_house"

    if is_flush:
        return 5, sorted(values, reverse=True), "flush"

    if straight:
        return 4, [straight_high], "straight"

    if freqs == [3, 1, 1]:
        trips = count_groups[0][0]
        kickers = sorted((v for v, c in counts.items() if c == 1), reverse=True)
        return 3, [trips] + kickers, "trips"

    if freqs == [2, 2, 1]:
        pairs = sorted((v for v, c in counts.items() if c == 2), reverse=True)
        kicker = max(v for v, c in counts.items() if c == 1)
        return 2, pairs + [kicker], "two_pair"

    if freqs == [2, 1, 1, 1]:
        pair = max(v for v, c in counts.items() if c == 2)
        kickers = sorted((v for v, c in counts.items() if c == 1), reverse=True)
        return 1, [pair] + kickers, "pair"

    return 0, sorted(values, reverse=True), "high_card"


def best_hand(cards: List[Card]) -> Tuple[int, List[int], str]:
    best = None
    for combo in combinations(cards, 5):
        score = evaluate_5(list(combo))
        if best is None or score > best:
            best = score
    assert best is not None
    return best


def board_texture(board: List[Card]) -> dict:
    values = sorted((c.value for c in board), reverse=True)
    suits = [c.suit for c in board]
    suit_counts = Counter(suits)
    rank_counts = Counter(values)

    monotone = (
        max(suit_counts.values(), default=0) >= 3
        and len(board) >= 3
        and len(set(suits)) == 1
    )
    two_tone = max(suit_counts.values(), default=0) >= 2
    paired = max(rank_counts.values(), default=0) >= 2

    unique_vals = sorted(set(values))
    connectedness = 0
    if len(unique_vals) >= 2:
        gaps = [
            unique_vals[i + 1] - unique_vals[i]
            for i in range(len(unique_vals) - 1)
        ]
        connectedness = sum(1 for g in gaps if g <= 2)

    return {
        "monotone": monotone,
        "two_tone": two_tone,
        "paired": paired,
        "connected": connectedness >= 2,
        "very_connected": connectedness >= 3,
        "high_card": max(values) if values else None,
    }


def has_flush_draw(hole: List[Card], board: List[Card]) -> bool:
    cards = hole + board
    suit_counts = Counter(c.suit for c in cards)
    return max(suit_counts.values(), default=0) == 4


def has_oesd(hole: List[Card], board: List[Card]) -> bool:
    vals = sorted(set(c.value for c in hole + board))
    if 14 in vals:
        vals = sorted(set(vals + [1]))

    for start in range(1, 11):
        window = set(range(start, start + 5))
        present = window.intersection(vals)
        if len(present) == 4:
            run4a = set(range(start, start + 4))
            run4b = set(range(start + 1, start + 5))
            if run4a.issubset(vals) or run4b.issubset(vals):
                return True
    return False


def has_gutshot(hole: List[Card], board: List[Card]) -> bool:
    vals = sorted(set(c.value for c in hole + board))
    if 14 in vals:
        vals = sorted(set(vals + [1]))

    for start in range(1, 11):
        window = set(range(start, start + 5))
        present = window.intersection(vals)
        if len(present) == 4:
            missing = list(window - present)[0]
            if missing not in {start, start + 4}:
                return True
    return False


def overcards_to_board(hole: List[Card], board: List[Card]) -> int:
    if not board:
        return 0
    board_high = max(c.value for c in board)
    return sum(1 for c in hole if c.value > board_high)


def estimate_outs(hole: List[Card], board: List[Card], hand_name: str) -> int:
    """Heuristics for counting outs."""
    outs = 0

    if hand_name in {
        "straight_flush",
        "quads",
        "full_house",
        "flush",
        "straight",
        "trips",
        "two_pair",
        "pair",
    }:
        return 0

    flush_draw = has_flush_draw(hole, board)
    oesd = has_oesd(hole, board)
    gutshot = has_gutshot(hole, board)
    overcards = overcards_to_board(hole, board)

    if flush_draw:
        outs += 9
    if oesd:
        outs += 8
    elif gutshot:
        outs += 4

    if overcards == 2:
        outs += 6
    elif overcards == 1:
        outs += 3

    return outs


def made_strength_percentile(hole: List[Card], board: List[Card]) -> float:
    """Relative rank of hero showdown strength among all opponent holdings from the remaining deck."""
    dead: Set[Card] = set(hole + board)
    deck = _remaining_deck(dead)
    hero = best_hand(hole + board)
    weaker = tie = 0
    n = 0
    for combo in combinations(deck, 2):
        sc = best_hand(list(combo) + board)
        n += 1
        if sc < hero:
            weaker += 1
        elif sc == hero:
            tie += 1
    if n == 0:
        return 0.5
    return (weaker + 0.5 * tie) / n


def draw_strength_from_hand(hole: List[Card], board: List[Card]) -> float:
    """Heuristic draw strength in ``[0, 1]`` from outs; 0 on the river."""
    if len(board) >= 5:
        return 0.0
    score = best_hand(hole + board)
    hand_name = score[2]
    outs = estimate_outs(hole, board, hand_name)
    return min(1.0, outs / 22.0)


def strength_bucket_from_percentiles(made_p: float, draw_s: float) -> str:
    """Map continuous made/draw scores to discrete buckets for legacy filters."""
    if made_p >= 0.93:
        return "nuts/near-nuts"
    if made_p >= 0.80:
        return "strong made"
    if made_p >= 0.62:
        return "medium made"
    if made_p >= 0.45:
        return "weak made"
    if draw_s >= 0.45:
        return "strong draw"
    if draw_s >= 0.18:
        return "weak draw"
    return "air"


def cards_str_to_list(cards_str: str) -> List[str]:
    cards_str = cards_str.strip()
    if len(cards_str) % 2 != 0:
        raise ValueError(f"Invalid cards string: {cards_str}")
    return list(cards_str[i : i + 2] for i in range(0, len(cards_str), 2))


def poker_hand_mapper(hole_cards, board_cards) -> dict:
    """Map hole + board to made/draw features and a legacy strength bucket."""
    if isinstance(hole_cards, str):
        hole_cards = cards_str_to_list(hole_cards)
    if isinstance(board_cards, str):
        board_cards = cards_str_to_list(board_cards)

    hole = parse_cards(hole_cards)
    board = parse_cards(board_cards)

    if len(hole) != 2:
        raise ValueError("Texas Hold'em hole cards must contain exactly 2 cards.")
    if len(board) < 3 or len(board) > 5:
        raise ValueError("Board must contain 3, 4, or 5 cards.")

    score = best_hand(hole + board)
    hand_name = score[2]
    tex = board_texture(board)

    made_p = made_strength_percentile(hole, board)
    draw_s = draw_strength_from_hand(hole, board)
    outs = estimate_outs(hole, board, hand_name)

    bucket = strength_bucket_from_percentiles(made_p, draw_s)

    return {
        "made": made_p,
        "draw": draw_s,
        "bucket": bucket,
        "score": score,
        "hand_type": hand_name,
        "outs": outs,
        "board_texture": tex,
    }
