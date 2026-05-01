"""Compact public state encoders.

The parser's ``State`` object is intentionally rich: it stores stacks, exact
betting history, active players, and board cards.  EM needs a much smaller
state key, otherwise almost every decision becomes its own unique state.  This
module maps a raw parser state to a discrete key used by:

* the GTO-inspired pre-flop prior ``phi[h, s, a]``;
* the pre-flop filter ``R_t(h)``;
* later EM soft-count tables ``N[h, s, a]``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .cards import RANK_TO_VALUE, split_cards
from .action_encoder import committed_before_action


@dataclass(frozen=True)
class StateKey:
    """Small discrete public state used by the statistical model."""

    street: str
    position: str
    active_players: str
    facing_bet: str
    raise_count: str
    spr: str
    board_texture: str

    def as_tuple(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.street,
            self.position,
            self.active_players,
            self.facing_bet,
            self.raise_count,
            self.spr,
            self.board_texture,
        )

    def as_string(self) -> str:
        """Stable string version suitable for dictionary keys and CSV output."""

        return "|".join(self.as_tuple())


def active_player_count(state) -> int:
    return sum(1 for _player, is_active in state.players_in_hand if is_active)


def active_count_bucket(count: int) -> str:
    if count <= 2:
        return "heads_up"
    if count == 3:
        return "three_way"
    return "multiway"


def position_bucket(player: str, num_players: int) -> str:
    """Return a coarse position bucket from the parser's seat labels.

    The legacy parser assumes p1 = small blind and p2 = big blind.  For six-max
    hands, p6 is treated as button/late position.  The bucket is intentionally
    coarse because sparse state keys will make EM unstable.
    """

    idx = int(player[1:])
    if idx == 1:
        return "sb"
    if idx == 2:
        return "bb"
    if num_players <= 3:
        return "button_or_late"
    if idx == num_players:
        return "button_or_late"
    if idx >= max(3, num_players - 1):
        return "late"
    if idx == 3:
        return "early"
    return "middle"


def raise_count_bucket(state) -> str:
    """Bucket the number of bet/raise actions already made this street."""

    max_level = 0
    for _player, action_info, _amount in state.betting_history:
        max_level = max(max_level, int(action_info[1]))

    if max_level <= 0:
        return "raises_0"
    if max_level == 1:
        return "raises_1"
    return "raises_2plus"


def stack_to_pot_bucket(state, player: str) -> str:
    """Bucket stack-to-pot ratio before the action."""

    pot = float(state.pot_size)
    stack = float(state.current_stacks.get(player, 0.0))
    if pot <= 0:
        return "spr_high"

    spr = stack / pot
    if spr <= 2:
        return "spr_low"
    if spr <= 6:
        return "spr_mid"
    return "spr_high"


def _values_and_suits(board: str) -> tuple[list[int], list[str]]:
    cards = split_cards(board)
    values = [RANK_TO_VALUE[card[0]] for card in cards]
    suits = [card[1] for card in cards]
    return values, suits


def board_texture_bucket(board: str) -> str:
    """Return a small board texture bucket.

    This is deliberately less detailed than ``utils/hand_map.py``.  The goal is
    a stable state key for action likelihoods, not perfect poker semantics.
    """

    cards = split_cards(board)
    if len(cards) < 3:
        return "no_board"

    values, suits = _values_and_suits(board)
    suit_counts = Counter(suits)
    rank_counts = Counter(values)

    paired = max(rank_counts.values(), default=0) >= 2
    monotone = max(suit_counts.values(), default=0) >= 3
    two_tone = max(suit_counts.values(), default=0) == 2

    unique_values = sorted(set(values))
    close_gaps = 0
    for i in range(len(unique_values) - 1):
        if unique_values[i + 1] - unique_values[i] <= 2:
            close_gaps += 1
    connected = close_gaps >= 2

    if paired:
        return "paired"
    if monotone and connected:
        return "monotone_connected"
    if monotone:
        return "monotone"
    if two_tone and connected:
        return "two_tone_connected"
    if two_tone:
        return "two_tone"
    if connected:
        return "connected"
    return "dry"


def current_bet_to_call(state, hand, street: str) -> float:
    """Reconstruct the current street bet before the action.

    The legacy parser stores each history entry's total ``amount_to`` on the
    street.  The amount to call is therefore the maximum prior amount committed
    by any player.  Pre-flop blinds are not listed in betting history, so before
    voluntary action starts the big blind is the current bet.
    """

    current = 0.0
    for _player, _action_info, amount_to in state.betting_history:
        current = max(current, float(amount_to))

    if street == "pre-flop":
        current = max(current, float(getattr(hand, "_bb", 0.0)))
    return current


def facing_bet_bucket(state, hand, street: str, player: str) -> str:
    """Return whether the acting player faces a bet before acting."""

    current_bet = current_bet_to_call(state, hand, street)
    committed = committed_before_action(state, hand, street, player)
    return "facing_bet" if current_bet > committed else "not_facing_bet"


def encode_state(state, hand, street: str, player: str, action_encoding=None) -> StateKey:
    """Encode a parser ``State`` into the compact model state key."""

    num_players = len(hand.player_names)
    return StateKey(
        street=street,
        position=position_bucket(player, num_players),
        active_players=active_count_bucket(active_player_count(state)),
        facing_bet=facing_bet_bucket(state, hand, street, player),
        raise_count=raise_count_bucket(state),
        spr=stack_to_pot_bucket(state, player),
        board_texture=board_texture_bucket(state.community_cards),
    )
