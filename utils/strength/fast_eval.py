"""Fast vectorized poker-hand evaluator and per-board feature tables.

Cards are packed as integers 0..51 with ``idx = rank_idx * 4 + suit_idx``
where ``rank_idx ∈ 0..12`` (mapping ``"23456789TJQKA"``) and
``suit_idx ∈ 0..3`` (``S=0, H=1, D=2, C=3`` — matching ``all_52_cards``).
``idx >> 2`` then recovers the rank index and ``idx & 3`` the suit index,
which lets the classifier run on bit-set rank/suit masks instead of
enumerating C(7,5) 5-card subsets.

The headline routine is :func:`made_percentile_index_table`, which returns
the made-strength percentile (vs uniform opponent holdings) for **every
non-blocked 2-card combo** on a given board in a single pass. For the
flop (board = 3 cards) it costs O(1326) hand evaluations + one
O(1326x1326) NumPy comparison, which replaces the previous O(1326 * 990 *
21) per-combo enumeration in ``made_strength_percentile``.

:func:`rollout_equity_index_table` (Method E) reuses the same machinery
to compute the expected made percentile across all future runouts
(exact on the turn, exact-or-Monte-Carlo on the flop).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils.parse import Card, RANK_TO_VALUE, all_52_cards

_RANK_CHARS = "23456789TJQKA"
_SUIT_CHARS_UPPER = "SHDC"  # suit_idx 0..3 -> 'S', 'H', 'D', 'C'
_SUIT_CHAR_ASCII = tuple(ord(c) for c in _SUIT_CHARS_UPPER)  # for canonical-key tiebreak
_SUIT_TO_IDX = {c: i for i, c in enumerate(_SUIT_CHARS_UPPER)}


def card_to_index(card: Card) -> int:
    """``rank_idx * 4 + suit_idx`` packing (rank_idx 0..12, suit_idx 0..3)."""
    return (RANK_TO_VALUE[card.rank] - 2) * 4 + _SUIT_TO_IDX[card.suit]


def index_to_card(idx: int) -> Card:
    return Card(_RANK_CHARS[idx >> 2], _SUIT_CHARS_UPPER[idx & 3])


def parse_board_indices(board: str) -> Tuple[int, ...]:
    """Encode a Pluribus-style board string (``"AhKsTd"``) into card indices."""
    if not board:
        return ()
    s = board.strip()
    if len(s) % 2 != 0:
        raise ValueError(f"Invalid board string: {board!r}")
    out: List[int] = []
    for i in range(0, len(s), 2):
        tok = s[i : i + 2]
        rank = tok[0].upper()
        suit = tok[1].upper()
        if rank not in RANK_TO_VALUE or suit not in _SUIT_TO_IDX:
            raise ValueError(f"Invalid card token {tok!r} in board {board!r}")
        out.append((RANK_TO_VALUE[rank] - 2) * 4 + _SUIT_TO_IDX[suit])
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate cards in board {board!r}")
    return tuple(out)


def combo_key_from_indices(a_idx: int, b_idx: int) -> str:
    """Canonical 4-char combo key matching ``utils.filter.postflop.combo_key``.

    Sort cards by ``(-rank_value, suit_char)`` so the higher-rank card
    leads, tiebroken by suit alphabetical order (``C < D < H < S``).
    """
    a_rank, a_suit = a_idx >> 2, a_idx & 3
    b_rank, b_suit = b_idx >> 2, b_idx & 3
    key_a = (-a_rank, _SUIT_CHAR_ASCII[a_suit])
    key_b = (-b_rank, _SUIT_CHAR_ASCII[b_suit])
    if key_a <= key_b:
        first, second = a_idx, b_idx
    else:
        first, second = b_idx, a_idx
    fr, fs = first >> 2, first & 3
    sr, ss = second >> 2, second & 3
    return (
        f"{_RANK_CHARS[fr]}{_SUIT_CHARS_UPPER[fs].lower()}"
        f"{_RANK_CHARS[sr]}{_SUIT_CHARS_UPPER[ss].lower()}"
    )


# ---------------------------------------------------------------------------
# 5/6/7-card hand ranking via bit-set classification
# ---------------------------------------------------------------------------

# Highest-rank-first straight masks: (rank_bitmask, top_rank_idx)
_STRAIGHT_MASKS: Tuple[Tuple[int, int], ...] = tuple(
    (0b11111 << i, i + 4) for i in range(8, -1, -1)
)
# Wheel A-2-3-4-5 -> bits 12, 0, 1, 2, 3; top rank index = 3 (the 5)
_WHEEL_MASK = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)


# Pack (category, up to 5 tiebreak nibbles) into a single int.
_CAT_SHIFT = 20
_TB_SHIFTS = (16, 12, 8, 4, 0)


def _highest_straight(rank_mask: int) -> int:
    for m, hi in _STRAIGHT_MASKS:
        if (rank_mask & m) == m:
            return hi
    if (rank_mask & _WHEEL_MASK) == _WHEEL_MASK:
        return 3
    return -1


def _pack(category: int, tiebreaks: Iterable[int]) -> int:
    v = category << _CAT_SHIFT
    for shift, tb in zip(_TB_SHIFTS, tiebreaks):
        v |= (int(tb) & 0xF) << shift
    return v


# Category constants (must equal the legacy ``evaluate_5`` categories).
CAT_HIGH_CARD = 0
CAT_PAIR = 1
CAT_TWO_PAIR = 2
CAT_TRIPS = 3
CAT_STRAIGHT = 4
CAT_FLUSH = 5
CAT_FULL_HOUSE = 6
CAT_QUADS = 7
CAT_STRAIGHT_FLUSH = 8


def rank_hand(cards: Iterable[int]) -> int:
    """Rank a 5/6/7-card hand. Higher int = stronger hand.

    Single pass, no 5-card-subset enumeration: classify via bit masks of
    present ranks and per-suit rank masks.
    """
    rank_count = [0] * 13
    suit_count = [0, 0, 0, 0]
    suit_rank_mask = [0, 0, 0, 0]
    for c in cards:
        r = c >> 2
        s = c & 3
        rank_count[r] += 1
        suit_count[s] += 1
        suit_rank_mask[s] |= (1 << r)

    rank_mask = 0
    for r in range(13):
        if rank_count[r] > 0:
            rank_mask |= (1 << r)

    flush_suit = -1
    for s in range(4):
        if suit_count[s] >= 5:
            flush_suit = s
            break

    if flush_suit >= 0:
        sf_hi = _highest_straight(suit_rank_mask[flush_suit])
        if sf_hi >= 0:
            return _pack(CAT_STRAIGHT_FLUSH, (sf_hi,))

    quad_r = -1
    for r in range(12, -1, -1):
        if rank_count[r] == 4:
            quad_r = r
            break
    if quad_r >= 0:
        for r in range(12, -1, -1):
            if r != quad_r and rank_count[r] >= 1:
                return _pack(CAT_QUADS, (quad_r, r))

    trips: List[int] = []
    pairs: List[int] = []
    for r in range(12, -1, -1):
        c = rank_count[r]
        if c == 3:
            trips.append(r)
        elif c == 2:
            pairs.append(r)

    if trips:
        if pairs:
            return _pack(CAT_FULL_HOUSE, (trips[0], pairs[0]))
        if len(trips) >= 2:
            return _pack(CAT_FULL_HOUSE, (trips[0], trips[1]))

    if flush_suit >= 0:
        srm = suit_rank_mask[flush_suit]
        top: List[int] = []
        for r in range(12, -1, -1):
            if srm & (1 << r):
                top.append(r)
                if len(top) == 5:
                    break
        return _pack(CAT_FLUSH, top)

    s_hi = _highest_straight(rank_mask)
    if s_hi >= 0:
        return _pack(CAT_STRAIGHT, (s_hi,))

    if trips:
        t = trips[0]
        kickers: List[int] = []
        for r in range(12, -1, -1):
            if r != t and rank_count[r] >= 1:
                kickers.append(r)
                if len(kickers) == 2:
                    break
        return _pack(CAT_TRIPS, (t, *kickers))

    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        for r in range(12, -1, -1):
            if r != p1 and r != p2 and rank_count[r] >= 1:
                return _pack(CAT_TWO_PAIR, (p1, p2, r))

    if pairs:
        p = pairs[0]
        kickers = []
        for r in range(12, -1, -1):
            if r != p and rank_count[r] >= 1:
                kickers.append(r)
                if len(kickers) == 3:
                    break
        return _pack(CAT_PAIR, (p, *kickers))

    top = []
    for r in range(12, -1, -1):
        if rank_count[r] >= 1:
            top.append(r)
            if len(top) == 5:
                break
    return _pack(CAT_HIGH_CARD, top)


def hand_category(cards: Iterable[int]) -> int:
    """Return only the hand-category index 0..8 of the best 5-of-N hand."""
    return rank_hand(cards) >> _CAT_SHIFT


# ---------------------------------------------------------------------------
# NumPy-vectorized rank evaluator (Method 1 inner loop)
# ---------------------------------------------------------------------------

# Precomputed straight-mask table for vectorized detection: rank-bit
# patterns ordered from highest top-rank down so we can take the
# arg-of-first-match per row.
_STRAIGHT_MASK_ARR = np.array(
    [m for m, _ in _STRAIGHT_MASKS], dtype=np.int64
)
_STRAIGHT_HI_ARR = np.array(
    [hi for _, hi in _STRAIGHT_MASKS], dtype=np.int8
)
_RANK_POW = (np.int64(1) << np.arange(13, dtype=np.int64))


def _rank_batch(hands: np.ndarray) -> np.ndarray:
    """NumPy vectorized rank evaluator. ``hands`` is ``(N, n)`` of int 0..51.

    Same packed-int output as :func:`rank_hand`, computed for every row
    in parallel. Used by :func:`_per_board_state` to score all ~1326
    combos for a fixed board in a single NumPy pass (~1 ms vs ~6 ms in a
    Python loop).
    """
    N, n = hands.shape
    if not (5 <= n <= 7):
        raise ValueError(f"_rank_batch supports 5..7 cards, got {n}")
    rank = hands >> 2  # (N, n)
    suit = hands & 3

    # (N, 13) counts per rank
    rank_eq = (rank[:, :, None] == np.arange(13)[None, None, :])
    rank_count = rank_eq.sum(axis=1, dtype=np.int32)
    rank_present = rank_count > 0
    rank_mask = (rank_present.astype(np.int64) * _RANK_POW[None, :]).sum(axis=1)

    # (N, 4) counts per suit + suit-rank bitmasks
    suit_eq = (suit[:, :, None] == np.arange(4)[None, None, :])
    suit_count = suit_eq.sum(axis=1, dtype=np.int32)
    # For each suit, mask of which ranks appear: any over the card axis
    # of (rank_eq & suit_eq_broadcast). Shape (N, n, 13) & (N, n, 4) -> (N, n, 4, 13)
    in_suit = suit_eq[:, :, :, None] & rank_eq[:, :, None, :]  # (N, n, 4, 13)
    suit_rank_mask = (in_suit.any(axis=1).astype(np.int64) * _RANK_POW[None, None, :]).sum(axis=2)
    # ^ shape (N, 4)

    has_flush = (suit_count >= 5).any(axis=1)
    # flush suit per row: first suit with >=5
    flush_suit = (suit_count >= 5).argmax(axis=1)

    # Straight detection on rank_mask: True if (rank_mask & m) == m
    sm_check = ((rank_mask[:, None] & _STRAIGHT_MASK_ARR[None, :]) == _STRAIGHT_MASK_ARR[None, :])
    has_straight = sm_check.any(axis=1)
    # top index of first True (masks ordered high->low)
    first_match = sm_check.argmax(axis=1)
    straight_top = np.where(has_straight, _STRAIGHT_HI_ARR[first_match], np.int8(-1))
    wheel_match = (rank_mask & _WHEEL_MASK) == _WHEEL_MASK
    straight_top = np.where(
        (straight_top < 0) & wheel_match, np.int8(3), straight_top
    )
    has_straight_any = straight_top >= 0

    # Straight-flush check: same logic on suit_rank_mask[flush_suit]
    srm_flush = np.take_along_axis(suit_rank_mask, flush_suit[:, None], axis=1)[:, 0]
    sf_check = ((srm_flush[:, None] & _STRAIGHT_MASK_ARR[None, :]) == _STRAIGHT_MASK_ARR[None, :])
    sf_match_any = sf_check.any(axis=1)
    sf_first = sf_check.argmax(axis=1)
    sf_top = np.where(sf_match_any, _STRAIGHT_HI_ARR[sf_first], np.int8(-1))
    sf_wheel = (srm_flush & _WHEEL_MASK) == _WHEEL_MASK
    sf_top = np.where((sf_top < 0) & sf_wheel, np.int8(3), sf_top)
    is_sf = has_flush & (sf_top >= 0)

    # Quads: highest rank with count == 4
    quad_indicator = np.where(
        rank_count == 4, np.arange(13)[None, :], np.int32(-1)
    )
    quad_rank = quad_indicator.max(axis=1)
    has_quad = quad_rank >= 0
    # Quad kicker: highest other present rank
    kicker_mask = (rank_count >= 1) & (np.arange(13)[None, :] != quad_rank[:, None])
    quad_kicker = np.where(kicker_mask, np.arange(13)[None, :], np.int32(-1)).max(axis=1)

    # Trips & pairs
    trip_indicator = np.where(rank_count == 3, np.arange(13)[None, :], np.int32(-1))
    trip_rank = trip_indicator.max(axis=1)
    # second-highest trip (for full house with two trips)
    trip_mask_lo = (rank_count == 3) & (np.arange(13)[None, :] != trip_rank[:, None])
    trip_rank_lo = np.where(trip_mask_lo, np.arange(13)[None, :], np.int32(-1)).max(axis=1)
    pair_indicator = np.where(rank_count == 2, np.arange(13)[None, :], np.int32(-1))
    pair_rank = pair_indicator.max(axis=1)
    pair_mask_lo = (rank_count == 2) & (np.arange(13)[None, :] != pair_rank[:, None])
    pair_rank_lo = np.where(pair_mask_lo, np.arange(13)[None, :], np.int32(-1)).max(axis=1)

    has_trip = trip_rank >= 0
    has_pair = pair_rank >= 0
    fh_pair = np.maximum(pair_rank, trip_rank_lo)
    has_fh = has_trip & (fh_pair >= 0)

    # Top 5 ranks for high-card / flush. Vectorized by iteratively masking
    # out the chosen top rank.
    def _top5_ranks(mask_bits: np.ndarray) -> np.ndarray:
        """For an (N,) int rank-bitmask vector return (N, 5) of top-5 rank idx
        (or -1 in trailing slots if fewer than 5 are present).
        """
        remaining = mask_bits.astype(np.int64).copy()
        out = np.full((mask_bits.shape[0], 5), -1, dtype=np.int8)
        for k in range(5):
            # arg of highest set bit per row
            # iterate ranks high->low, take first present
            bits = (remaining[:, None] & _RANK_POW[None, :]) > 0  # (N, 13)
            any_left = bits.any(axis=1)
            # highest rank index
            top_idx = 12 - bits[:, ::-1].argmax(axis=1)
            top_idx = np.where(any_left, top_idx, -1).astype(np.int8)
            out[:, k] = top_idx
            # clear the chosen bit
            clear = np.where(top_idx >= 0, _RANK_POW[np.maximum(top_idx, 0)], 0)
            remaining = remaining & ~clear
        return out

    # Flush: top-5 ranks in the flush suit
    flush_top5 = _top5_ranks(srm_flush)
    # High-card: top-5 across all ranks
    hc_top5 = _top5_ranks(rank_mask)

    # For trips/two-pair/pair we also need kickers from the non-paired ranks
    def _packed(category: np.ndarray, tb_list) -> np.ndarray:
        """category: (N,) int. tb_list: list of (N,) int columns (high-to-low priority)."""
        v = category.astype(np.int64) << _CAT_SHIFT
        for shift, col in zip(_TB_SHIFTS, tb_list):
            v |= ((col.astype(np.int64) & np.int64(0xF)) << shift)
        return v

    N = hands.shape[0]
    # Default = high card
    out = _packed(
        np.zeros(N, dtype=np.int32),
        [hc_top5[:, k].astype(np.int32) for k in range(5)],
    )

    # Pair
    pair_kicker_mask = (rank_count >= 1) & (np.arange(13)[None, :] != pair_rank[:, None])
    pair_kickers = _top5_ranks(
        (pair_kicker_mask.astype(np.int64) * _RANK_POW[None, :]).sum(axis=1)
    )
    pair_val = _packed(
        np.full(N, CAT_PAIR, dtype=np.int32),
        [
            pair_rank.astype(np.int32),
            pair_kickers[:, 0].astype(np.int32),
            pair_kickers[:, 1].astype(np.int32),
            pair_kickers[:, 2].astype(np.int32),
        ],
    )
    out = np.where(has_pair, pair_val, out)

    # Two pair
    has_two_pair = has_pair & (pair_rank_lo >= 0)
    tp_kicker_mask = (
        (rank_count >= 1)
        & (np.arange(13)[None, :] != pair_rank[:, None])
        & (np.arange(13)[None, :] != pair_rank_lo[:, None])
    )
    tp_kicker = np.where(tp_kicker_mask, np.arange(13)[None, :], np.int32(-1)).max(axis=1)
    tp_val = _packed(
        np.full(N, CAT_TWO_PAIR, dtype=np.int32),
        [
            pair_rank.astype(np.int32),
            pair_rank_lo.astype(np.int32),
            tp_kicker.astype(np.int32),
        ],
    )
    out = np.where(has_two_pair, tp_val, out)

    # Trips
    trip_kicker_mask = (rank_count >= 1) & (np.arange(13)[None, :] != trip_rank[:, None])
    trip_kickers = _top5_ranks(
        (trip_kicker_mask.astype(np.int64) * _RANK_POW[None, :]).sum(axis=1)
    )
    trip_val = _packed(
        np.full(N, CAT_TRIPS, dtype=np.int32),
        [
            trip_rank.astype(np.int32),
            trip_kickers[:, 0].astype(np.int32),
            trip_kickers[:, 1].astype(np.int32),
        ],
    )
    out = np.where(has_trip, trip_val, out)

    # Straight
    straight_val = _packed(
        np.full(N, CAT_STRAIGHT, dtype=np.int32),
        [straight_top.astype(np.int32)],
    )
    out = np.where(has_straight_any, straight_val, out)

    # Flush
    flush_val = _packed(
        np.full(N, CAT_FLUSH, dtype=np.int32),
        [flush_top5[:, k].astype(np.int32) for k in range(5)],
    )
    out = np.where(has_flush, flush_val, out)

    # Full house
    fh_val = _packed(
        np.full(N, CAT_FULL_HOUSE, dtype=np.int32),
        [trip_rank.astype(np.int32), fh_pair.astype(np.int32)],
    )
    out = np.where(has_fh, fh_val, out)

    # Quads
    quad_val = _packed(
        np.full(N, CAT_QUADS, dtype=np.int32),
        [quad_rank.astype(np.int32), quad_kicker.astype(np.int32)],
    )
    out = np.where(has_quad, quad_val, out)

    # Straight flush
    sf_val = _packed(
        np.full(N, CAT_STRAIGHT_FLUSH, dtype=np.int32),
        [sf_top.astype(np.int32)],
    )
    out = np.where(is_sf, sf_val, out)

    return out


# ---------------------------------------------------------------------------
# Static lookup: all 1326 unordered 2-card combos as (a_idx, b_idx) with a<b
# ---------------------------------------------------------------------------


def _build_combo_pairs() -> Tuple[np.ndarray, Tuple[str, ...]]:
    """All 1326 ``(a_idx, b_idx)`` pairs with ``a < b`` plus their canonical keys."""
    pairs = np.array(
        [(a, b) for a, b in combinations(range(52), 2)], dtype=np.int32
    )
    keys = tuple(
        combo_key_from_indices(int(a), int(b)) for a, b in pairs.tolist()
    )
    return pairs, keys


_ALL_COMBO_PAIRS, _ALL_COMBO_KEYS = _build_combo_pairs()
_COMBO_KEY_TO_ROW: Dict[str, int] = {k: i for i, k in enumerate(_ALL_COMBO_KEYS)}


def all_combo_pairs() -> np.ndarray:
    """``(1326, 2) int32`` table of card-index pairs (a < b)."""
    return _ALL_COMBO_PAIRS


def all_combo_keys_fast() -> Tuple[str, ...]:
    """Canonical 4-char combo keys aligned with :func:`all_combo_pairs`."""
    return _ALL_COMBO_KEYS


def combo_key_to_row(combo_key: str) -> int:
    """Row index in the static 1326-pair table for ``combo_key``."""
    return _COMBO_KEY_TO_ROW[combo_key]


# ---------------------------------------------------------------------------
# Per-board state cache: rank vector + pairwise disjointness + percentile
# ---------------------------------------------------------------------------


def _live_combo_mask(board_set: frozenset) -> np.ndarray:
    """``(1326,) bool`` mask of combos that don't conflict with the board."""
    pairs = _ALL_COMBO_PAIRS
    a_in = np.isin(pairs[:, 0], list(board_set))
    b_in = np.isin(pairs[:, 1], list(board_set))
    return ~(a_in | b_in)


@lru_cache(maxsize=8192)
def _per_board_state(board_indices: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    """Per-board cache: rank vector, pairwise disjointness, percentile vector.

    All arrays are indexed by **live combo row** (the subset of the 1326
    static pairs that don't share a card with the board), and the
    ``live_rows`` array gives the mapping back to the canonical
    1326-pair table.

    Cached by sorted board tuple. Larger ``maxsize`` reduces eviction
    thrash when many distinct boards appear in one training run.
    """
    n_board = len(board_indices)
    if n_board < 3 or n_board > 5:
        raise ValueError(f"Board must have 3..5 cards, got {n_board}")
    if len(set(board_indices)) != n_board:
        raise ValueError(f"Duplicate cards on board: {board_indices}")

    board_set = frozenset(board_indices)
    live_mask = _live_combo_mask(board_set)
    live_rows = np.nonzero(live_mask)[0].astype(np.int32)
    live_pairs = _ALL_COMBO_PAIRS[live_rows]  # (M, 2)

    n_board = len(board_indices)
    M = live_pairs.shape[0]
    hands = np.empty((M, 2 + n_board), dtype=np.int32)
    hands[:, :2] = live_pairs
    hands[:, 2:] = np.asarray(board_indices, dtype=np.int32)[None, :]
    ranks = _rank_batch(hands)

    # Pairwise disjoint mask: True where combos i and j share no card.
    # Excludes i == j automatically (since the cards collide with themselves).
    a = live_pairs[:, 0]
    b = live_pairs[:, 1]
    disjoint = (
        (a[:, None] != a[None, :])
        & (a[:, None] != b[None, :])
        & (b[:, None] != a[None, :])
        & (b[:, None] != b[None, :])
    )

    rank_col = ranks[:, None]
    rank_row = ranks[None, :]
    weaker = ((rank_row < rank_col) & disjoint).sum(axis=1)
    tied = ((rank_row == rank_col) & disjoint).sum(axis=1)
    total = disjoint.sum(axis=1)
    perc = (weaker + 0.5 * tied) / np.maximum(total, 1)

    return {
        "live_rows": live_rows,
        "live_pairs": live_pairs,
        "ranks": ranks,
        "percentile": perc.astype(np.float64),
    }


def made_percentile_index_table(
    board_indices: Iterable[int],
) -> Dict[Tuple[int, int], float]:
    """Made-strength percentile for every non-blocked 2-card combo on a board."""
    key = tuple(sorted(int(x) for x in board_indices))
    state = _per_board_state(key)
    pairs = state["live_pairs"]
    perc = state["percentile"]
    return {(int(a), int(b)): float(p) for (a, b), p in zip(pairs, perc)}


def made_percentile_array(
    board_indices: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized variant: returns ``(live_rows (M,), percentile (M,))``.

    ``live_rows[i]`` is the row in :func:`all_combo_pairs` (and
    :func:`all_combo_keys_fast`) of the ``i``\\-th live combo.
    """
    key = tuple(sorted(int(x) for x in board_indices))
    state = _per_board_state(key)
    return state["live_rows"], state["percentile"]


def made_percentile_by_combo_key(
    board_indices: Iterable[int],
) -> Dict[str, float]:
    """``{combo_key: percentile}`` for every non-blocked combo on this board."""
    live_rows, perc = made_percentile_array(board_indices)
    return {
        _ALL_COMBO_KEYS[int(r)]: float(p) for r, p in zip(live_rows, perc)
    }


def made_percentile_at_combo_key(
    board_indices: Iterable[int],
    combo_key: str,
) -> Optional[float]:
    """Made-strength percentile for one combo; uses cached :func:`_per_board_state` only.

    Avoids building the full ``{combo_key: p}`` dict returned by
    :func:`made_percentile_by_combo_key`, which matters when many decisions
    share the same board (e.g. global prior row collection).
    """
    board_tuple = tuple(sorted(int(x) for x in board_indices))
    if len(board_tuple) < 3 or len(board_tuple) > 5:
        return None
    row = _COMBO_KEY_TO_ROW.get(combo_key)
    if row is None:
        return None
    state = _per_board_state(board_tuple)
    live_rows = state["live_rows"]
    perc = state["percentile"]
    idx = int(np.searchsorted(live_rows, row))
    if idx < len(live_rows) and int(live_rows[idx]) == row:
        return float(perc[idx])
    return None


# ---------------------------------------------------------------------------
# Rollout equity (Method E)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def _rollout_equity_cached(
    board_tuple: Tuple[int, ...],
    mc_samples: int,
) -> np.ndarray:
    """Per-board rollout-equity vector aligned with the 1326-pair table.

    ``mc_samples == 0`` means exact enumeration over all future runouts.
    Output is a ``(1326,) float64`` array with ``NaN`` on combos that
    conflict with the starting board. Cached by ``(sorted board,
    mc_samples)``; MC sampling is deterministic in the board so repeated
    queries return identical values without an explicit RNG argument.
    """
    n_board = len(board_tuple)
    if n_board == 5:
        out_arr = np.full(1326, np.nan, dtype=np.float64)
        live_rows, perc = made_percentile_array(board_tuple)
        out_arr[live_rows] = perc
        return out_arr
    if n_board not in (3, 4):
        raise ValueError(f"Board must have 3..5 cards, got {n_board}")

    board_set = set(board_tuple)
    remaining = [c for c in range(52) if c not in board_set]

    if n_board == 4:
        runouts: List[Tuple[int, ...]] = [(c,) for c in remaining]
    else:  # flop
        all_pairs = list(combinations(remaining, 2))
        if mc_samples > 0 and mc_samples < len(all_pairs):
            # Seed the RNG from the (sorted) board so MC samples are
            # reproducible across calls — important for caching downstream.
            seed = abs(hash((board_tuple, "rollout_equity"))) % (2**32)
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(all_pairs), size=mc_samples, replace=False)
            runouts = [all_pairs[int(i)] for i in sel]
        else:
            runouts = all_pairs

    eq_sum = np.zeros(1326, dtype=np.float64)
    eq_count = np.zeros(1326, dtype=np.int32)
    for ro in runouts:
        full_board = tuple(sorted(board_tuple + tuple(int(c) for c in ro)))
        live_rows, perc = made_percentile_array(full_board)
        eq_sum[live_rows] += perc
        eq_count[live_rows] += 1

    out_arr = np.full(1326, np.nan, dtype=np.float64)
    starting_live = _live_combo_mask(frozenset(board_tuple))
    valid = starting_live & (eq_count > 0)
    np.divide(eq_sum, eq_count, out=out_arr, where=valid)
    out_arr[~valid] = np.nan
    return out_arr


def rollout_equity_index_table(
    board_indices: Iterable[int],
    *,
    mc_samples: Optional[int] = 0,
) -> Dict[Tuple[int, int], float]:
    """Expected made percentile averaged over all future runouts.

    * **River** (board = 5): equal to the current made percentile.
    * **Turn** (board = 4): exact enumeration over the ~48 river cards.
    * **Flop** (board = 3): exact over ~C(47, 2) = 1081 runouts when
      ``mc_samples`` is ``0`` (default), or Monte-Carlo with the given
      sample count for live use (~``mc_samples * 15 ms``).

    Cached per board (Method 1 reuses ``_per_board_state`` for each
    runout, and the per-board equity vector is itself memoized).
    """
    board_tuple = tuple(sorted(int(x) for x in board_indices))
    mc = 0 if mc_samples is None else int(mc_samples)
    eq_vec = _rollout_equity_cached(board_tuple, mc)
    pairs = _ALL_COMBO_PAIRS
    out: Dict[Tuple[int, int], float] = {}
    valid = ~np.isnan(eq_vec)
    rows = np.nonzero(valid)[0]
    for r in rows:
        a, b = int(pairs[r, 0]), int(pairs[r, 1])
        out[(a, b)] = float(eq_vec[r])
    return out


def rollout_equity_by_combo_key(
    board_indices: Iterable[int],
    *,
    mc_samples: Optional[int] = 0,
) -> Dict[str, float]:
    """``{combo_key: rollout_equity}`` for every live combo on this board."""
    board_tuple = tuple(sorted(int(x) for x in board_indices))
    mc = 0 if mc_samples is None else int(mc_samples)
    eq_vec = _rollout_equity_cached(board_tuple, mc)
    valid = ~np.isnan(eq_vec)
    rows = np.nonzero(valid)[0]
    return {
        _ALL_COMBO_KEYS[int(r)]: float(eq_vec[int(r)]) for r in rows
    }


def rollout_equity_at_combo_key(
    board_indices: Iterable[int],
    combo_key: str,
    *,
    mc_samples: Optional[int] = 0,
) -> Optional[float]:
    """Rollout / river equity for one combo from the cached 1326-vector (no full dict)."""
    board_tuple = tuple(sorted(int(x) for x in board_indices))
    if len(board_tuple) < 3 or len(board_tuple) > 5:
        return None
    row = _COMBO_KEY_TO_ROW.get(combo_key)
    if row is None:
        return None
    mc = 0 if mc_samples is None else int(mc_samples)
    v = float(_rollout_equity_cached(board_tuple, mc)[row])
    if np.isnan(v):
        return None
    return v


def clear_caches() -> None:
    """Drop memoized per-board state (tests / long-running sessions)."""
    _per_board_state.cache_clear()
    _rollout_equity_cached.cache_clear()
