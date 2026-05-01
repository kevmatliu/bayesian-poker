from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Tuple


RANK_TO_VALUE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
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


def is_straight(values: List[int]) -> Tuple[bool, int]:
    """
    Returns (is_straight, high_card_of_straight)
    Handles wheel: A-2-3-4-5
    """
    unique = sorted(set(values), reverse=True)
    if len(unique) < 5:
        return False, 0

    for i in range(len(unique) - 4):    # normal straights
        window = unique[i:i + 5]
        if window[0] - window[4] == 4 and len(window) == 5:
            return True, window[0]

    if {14, 5, 4, 3, 2}.issubset(set(unique)):  # wheel
        return True, 5

    return False, 0


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

    monotone = max(suit_counts.values(), default=0) >= 3 and len(board) >= 3 and len(set(suits)) == 1
    two_tone = max(suit_counts.values(), default=0) >= 2
    paired = max(rank_counts.values(), default=0) >= 2

    unique_vals = sorted(set(values))
    connectedness = 0
    if len(unique_vals) >= 2:
        gaps = [unique_vals[i + 1] - unique_vals[i] for i in range(len(unique_vals) - 1)]
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
        vals = sorted(set(vals + [1]))  # Ace low support

    for start in range(1, 11):
        window = set(range(start, start + 5))
        present = window.intersection(vals)
        if len(present) == 4:
            missing = list(window - present)[0]
            # open-ended means missing on one end of a 4-card run
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
            # gutshot means internal missing card
            if missing not in {start, start + 4}:
                return True
    return False


def overcards_to_board(hole: List[Card], board: List[Card]) -> int:
    if not board:
        return 0
    board_high = max(c.value for c in board)
    return sum(1 for c in hole if c.value > board_high)


def estimate_outs(hole: List[Card], board: List[Card], hand_name: str) -> int:
    """
    Heuristics for counting outs
    """
    outs = 0

    if hand_name in {"straight_flush", "quads", "full_house", "flush", "straight", "trips", "two_pair", "pair"}:
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

    if overcards == 2:  # crude overcard heuristics only when still unpaired / air-ish
        outs += 6
    elif overcards == 1:
        outs += 3

    return outs


def made_hand_bucket(
    hole: List[Card],
    board: List[Card],
    hand_name: str,
    score_tuple: Tuple[int, List[int], str],
) -> str:
    texture = board_texture(board)
    board_values = [c.value for c in board]
    hole_values = [c.value for c in hole]
    rank_counts_all = Counter(c.value for c in hole + board)

    top_board = max(board_values) if board_values else None
    second_board = sorted(board_values, reverse=True)[1] if len(board_values) >= 2 else None

    # strong nut classes
    if hand_name in {"straight_flush", "quads"}:
        return "nuts/near-nuts"

    if hand_name == "full_house":
        return "nuts/near-nuts"

    if hand_name == "flush":
        # On monotone or 4-flush boards, non-nut flushes are less absolute.
        flush_suit = next(s for s, cnt in Counter(c.suit for c in hole + board).items() if cnt >= 5)
        # flush_cards = sorted([c.value for c in hole + board if c.suit == flush_suit], reverse=True)
        hole_flush_vals = sorted([c.value for c in hole if c.suit == flush_suit], reverse=True)

        if hole_flush_vals and max(hole_flush_vals) == 14:
            return "nuts/near-nuts"
        if texture["monotone"]:
            return "strong made"
        return "nuts/near-nuts"

    if hand_name == "straight":
        if texture["paired"] or texture["monotone"]:
            return "strong made"
        return "nuts/near-nuts"

    if hand_name == "trips":
        hole_counter = Counter(hole_values)
        set_made = any(cnt == 2 and v in board_values for v, cnt in hole_counter.items())

        if set_made:
            if texture["monotone"] or texture["very_connected"]:    # trips on wet boards are less strong
                return "strong made"
            return "nuts/near-nuts"
        
        return "strong made"    # trips from paired board are usually less strong

    if hand_name == "two_pair":
        if len(board) >= 4 and texture["very_connected"]:
            return "medium made"
        return "strong made"

    if hand_name == "pair":
        pair_rank = score_tuple[1][0]

        # overpair
        if top_board is not None and pair_rank > top_board:
            return "strong made"

        # top pair
        if top_board is not None and pair_rank == top_board:
            kicker_values = sorted([v for v in hole_values if v != pair_rank], reverse=True)
            best_kicker = kicker_values[0] if kicker_values else 0
            if best_kicker >= 11:  # J+ kicker
                return "strong made"
            return "medium made"

        # second pair
        if second_board is not None and pair_rank == second_board:
            return "weak made"

        # any lower pair / underpair
        return "weak made"

    return "air"


def cards_str_to_list(cards_str: str) -> List[str]:
    cards_str = cards_str.strip()
    if len(cards_str) % 2 != 0:
        raise ValueError(f"Invalid cards string: {cards_str}")
    return list(cards_str[i:i + 2] for i in range(0, len(cards_str), 2))


def poker_hand_mapper(hole_cards, board_cards) -> dict:
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

    all_cards = hole + board
    score = best_hand(all_cards)
    hand_name = score[2]

    # made hand?
    if hand_name != "high_card":
        bucket = made_hand_bucket(hole, board, hand_name, score)
        return {
            "bucket": bucket,
            "score": score,
            "hand_type": hand_name,
            "outs": 0,
            "board_texture": board_texture(board),
        }

    # not made -> draw / air
    outs = estimate_outs(hole, board, hand_name)

    if outs >= 10:
        bucket = "strong draw"
    elif outs >= 3:
        bucket = "weak draw"
    else:
        bucket = "air"

    return {
        "bucket": bucket,
        "score": score,
        "hand_type": hand_name,
        "outs": outs,
        "board_texture": board_texture(board),
    }