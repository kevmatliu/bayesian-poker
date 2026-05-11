"""Postflop hand strength: board-aware made/draw features and ``poker_hand_mapper``."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from utils.strength.common import Card, StrengthMode, all_52_cards, parse_cards
from utils.strength.fast_eval import (
    card_to_index,
    combo_key_from_indices,
    hand_category,
    made_percentile_array,
    made_percentile_at_combo_key,
    made_percentile_by_combo_key,
    parse_board_indices,
    rollout_equity_by_combo_key,
)

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
    """Relative rank of hero showdown strength among all opponent holdings.

    Single-combo wrapper around the vectorized per-board percentile table
    in :mod:`utils.strength.fast_eval`. The expensive opponent-enumeration
    is shared across every hole that sits on the same board.
    """
    if len(hole) != 2:
        raise ValueError("Hole must have exactly 2 cards.")
    if not (3 <= len(board) <= 5):
        raise ValueError("Board must have 3..5 cards.")
    board_idx = tuple(sorted(card_to_index(c) for c in board))
    key = combo_key_from_indices(card_to_index(hole[0]), card_to_index(hole[1]))
    perc = made_percentile_at_combo_key(board_idx, key)
    if perc is None:
        # Conflict with the board; fall back to 0.5 to match the legacy contract.
        return 0.5
    return float(perc)


def made_percentile_table_for_board(board: List[Card]) -> Dict[str, float]:
    """``{combo_key: made_percentile}`` for every non-blocked combo on this board."""
    return made_percentile_by_combo_key(tuple(card_to_index(c) for c in board))


def draw_strength_from_hand(hole: List[Card], board: List[Card]) -> float:
    """Heuristic draw strength in ``[0, 1]`` from outs; 0 on the river."""
    if len(board) >= 5:
        return 0.0
    score = best_hand(hole + board)
    hand_name = score[2]
    outs = estimate_outs(hole, board, hand_name)
    return min(1.0, outs / 22.0)


# ---------------------------------------------------------------------------
# Method A: per-combo board-relative categorical features
# ---------------------------------------------------------------------------


# Index into the rich-feature vector returned by :func:`hand_feature_vector`.
RICH_FEAT_KEYS: Tuple[str, ...] = (
    "is_pair_or_better",     # at least one pair using both hole & board
    "is_top_pair",            # pair using highest board rank
    "is_over_pair",           # pocket pair higher than every board card
    "is_middle_pair",         # pair using middle board rank
    "is_under_pair",          # pocket pair below the lowest board card
    "is_two_pair_plus",       # two pair or better
    "is_set_or_better",       # trips or better
    "is_straight_or_better",  # straight or better
    "has_flush_draw",
    "has_oesd",
    "has_gutshot",
    "is_suited",              # hole cards same suit
    "is_pocket_pair",         # hole cards same rank
    "overcard_count",         # 0/1/2 hole cards above the highest board rank
    "blocks_top_pair",        # hole contains the top-board rank (blocker)
    "blocks_nut_flush",       # hole contains the Ace of the flush suit (if any)
)

RICH_FEAT_DIM: int = len(RICH_FEAT_KEYS)


def hand_feature_vector(
    hole: List[Card],
    board: List[Card],
) -> np.ndarray:
    """Length-:data:`RICH_FEAT_DIM` vector of board-relative hand features.

    Encodes information the 2-scalar ``(made, draw)`` representation
    cannot: which exact pair the player has, whether it's an overpair vs.
    the board, draw type, blockers, etc. Cheap (O(7 cards) per combo).
    """
    if len(hole) != 2 or len(board) < 3:
        return np.zeros(RICH_FEAT_DIM, dtype=float)

    hole_vals = sorted((c.value for c in hole), reverse=True)
    board_vals = sorted({c.value for c in board}, reverse=True)
    board_high = board_vals[0]
    board_low = board_vals[-1]
    # "Middle" board rank: use the median of board ranks.
    if len(board_vals) >= 3:
        board_mid = board_vals[len(board_vals) // 2]
    elif len(board_vals) == 2:
        board_mid = board_vals[-1]
    else:
        board_mid = board_high
    board_rank_set = set(board_vals)

    cards = hole + board
    cat = hand_category([card_to_index(c) for c in cards])

    is_pair_or_better = cat >= 1
    is_two_pair_plus = cat >= 2
    is_set_or_better = cat >= 3
    is_straight_or_better = cat >= 4

    hv1, hv2 = hole_vals
    hole_rank_set = {hv1, hv2}
    is_pocket_pair = hv1 == hv2
    is_suited = hole[0].suit == hole[1].suit

    is_top_pair = False
    is_middle_pair = False
    is_over_pair = False
    is_under_pair = False

    if is_pair_or_better and not is_two_pair_plus:
        # Single pair only — figure out which one.
        if is_pocket_pair:
            if hv1 > board_high:
                is_over_pair = True
            elif hv1 < board_low:
                is_under_pair = True
            else:
                is_middle_pair = True
        else:
            paired_with_board = hole_rank_set & board_rank_set
            if board_high in paired_with_board:
                is_top_pair = True
            elif board_mid in paired_with_board:
                is_middle_pair = True
            elif paired_with_board:
                is_under_pair = True

    has_fd = has_flush_draw(hole, board)
    has_oe = has_oesd(hole, board)
    has_gut = has_gutshot(hole, board) and not has_oe

    overcards = overcards_to_board(hole, board)

    blocks_top = (hv1 == board_high) or (hv2 == board_high)

    # Nut-flush blocker: only meaningful when there's at least a 2-tone board.
    suits = [c.suit for c in board]
    suit_counts = Counter(suits)
    blocks_nut_fd = False
    if suit_counts:
        dominant_suit, dom_count = suit_counts.most_common(1)[0]
        if dom_count >= 2:
            for c in hole:
                if c.suit == dominant_suit and c.value == 14:
                    blocks_nut_fd = True
                    break

    return np.array(
        [
            float(is_pair_or_better),
            float(is_top_pair),
            float(is_over_pair),
            float(is_middle_pair),
            float(is_under_pair),
            float(is_two_pair_plus),
            float(is_set_or_better),
            float(is_straight_or_better),
            float(has_fd),
            float(has_oe),
            float(has_gut),
            float(is_suited),
            float(is_pocket_pair),
            float(overcards),
            float(blocks_top),
            float(blocks_nut_fd),
        ],
        dtype=float,
    )


def combo_postflop_features_for_board(
    board: List[Card],
    *,
    equity_mc_samples: Optional[int] = 64,
    equity_rng: Optional[np.random.Generator] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Batch (made, equity, rich-features) for every live combo on ``board``.

    Returns ``{combo_key: {"made": float, "draw": float, "equity": float,
    "rich": np.ndarray (RICH_FEAT_DIM,)}}``. Uses the cached per-board
    percentile table for ``made``, the rollout-equity table for
    ``equity`` (Method E), and recomputes the cheap rich-feature vector
    per combo (~1 µs each).
    """
    if not (3 <= len(board) <= 5):
        raise ValueError("Board must have 3..5 cards.")
    board_idx = tuple(card_to_index(c) for c in board)
    made_tbl = made_percentile_by_combo_key(board_idx)
    if len(board) == 5:
        equity_tbl = made_tbl
    else:
        equity_tbl = rollout_equity_by_combo_key(
            board_idx,
            mc_samples=equity_mc_samples if len(board) == 3 else None,
            rng=equity_rng,
        )

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key, made in made_tbl.items():
        # Decode hole from canonical key
        h0 = key[0:2]
        h1 = key[2:4]
        hole = parse_cards([h0, h1])
        rich = hand_feature_vector(hole, board)
        draw = draw_strength_from_hand(hole, board)
        equity = float(equity_tbl.get(key, made))
        out[key] = {
            "made": float(made),
            "draw": float(draw),
            "equity": equity,
            "rich": rich,
        }
    return out


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
        "rich": hand_feature_vector(hole, board),
    }
