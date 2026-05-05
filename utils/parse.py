import os
import numpy as np

from pathlib import Path
from pokerkit import HandHistory

from utils.hand_map import Card, all_169_classes, get_equivalence_class, poker_hand_mapper
from utils.action_map import classify

class State:
    def __init__(
        self,
        player_order,
        community_cards,
        betting_history,
        player_to_act,
        players_in_hand,
        current_stacks,
        pot_size,
        hand_strength_map,
    ):
        self.player_order = player_order
        self.community_cards = community_cards
        self.betting_history = betting_history
        self.player_to_act = player_to_act
        self.players_in_hand = players_in_hand
        self.current_stacks = current_stacks
        self.pot_size = pot_size
        self.hand_strength_map = hand_strength_map

    def __repr__(self):
        return (
            f"State(player_order={self.player_order}, \n"
            f"State(community_cards={self.community_cards}, \n"
            f"betting_history={self.betting_history}, \n"
            f"player_to_act={self.player_to_act}, \n"
            f"players_in_hand={self.players_in_hand}, \n"
            f"current_stacks={self.current_stacks}, \n"
            f"pot_size={self.pot_size}, \n"
            f"hand_strength_map={self.hand_strength_map})\n"
        )

class Hand:
    STREETS = ['pre-flop', 'flop', 'turn', 'river']

    def __init__(self, hand_history):
        self.hand_history = hand_history

        # poker parameters
        self._bb = hand_history.blinds_or_straddles[1]
        self._sb = hand_history.blinds_or_straddles[0]

        # players
        self._num_players = len(hand_history.players)
        self.player_names = list(hand_history.players)
        self.player_order = list(self.player_names)
        self.seat_to_player = {
            f'p{i + 1}': self.player_names[i] for i in range(self._num_players)
        }
        self.player_to_seat = {
            player: seat for seat, player in self.seat_to_player.items()
        }

        # hand strength maps and hole cards by player
        self.hand_strength_map = {
            player: {} for player in self.player_names   # player --> {'bucket', 'score', 'hand_type', 'outs', 'board_texture'}
        }
        self.hole_cards = {
            player: '' for player in self.player_names   # player --> 'AhKh' format
        }
        self.hole_cards_class = {
            player: '' for player in self.player_names   # player --> 'AKs', 'AQo', etc. format
        }

        # probability vectors
        self.hand_range = {                                     # player1 --> player2 --> 169-dim vector of hand frequencies on player2 from player1 perspective
            player_i: {
                player_j: np.zeros(169) for player_j in self.player_names if player_j != player_i
            } for player_i in self.player_names
        }
        self.hand_strength = {                                  # player1 --> player2 --> 7-dim vector of hand strength scores on player2 from player1 perspective
            player_i: {                                      # buckets of nuts/near-nuts, strong made, medium made, weak made, strong draw, weak draw, air
                player_j: np.zeros(7) for player_j in self.player_names if player_j != player_i
            } for player_i in self.player_names
        }

        # more params
        self.player_positions = {                               # player positions at the table 
            player: self._position_name(player) for player in self.player_names
        }

        self.starting_stacks = {                                # player starting stacks
            self.player_names[i]: hand_history.starting_stacks[i]
            for i in range(self._num_players)
        }

        self.stacks = self.starting_stacks.copy()               # player current stacks, updated after each action

        self.states = {
            'pre-flop': [],
            'flop': [],
            'turn': [],
            'river': [],
        }

        self.actions = {
            'pre-flop': {},
            'flop': {},
            'turn': {},
            'river': {},
        }

        self._players_in_hand = {
            player: True for player in self.player_names
        }

        self._street = 'pre-flop'
        self._community_cards = ''
        self._pot = 0

        self._current_bet = 0              # current amount to call on this street
        self._money_in_round = {
            player: 0 for player in self.player_names
        }
        self._betting_history_this_street = []

        self._street_action_index = {
            'pre-flop': 0,
            'flop': 0,
            'turn': 0,
            'river': 0,
        }

        self._street_action_level = 0

        self._initialized = False
    
    def __repr__(self):
        return (
            f"Hand(player_positions={self.player_positions}, \n"
            f"hole_cards={self.hole_cards}, \n"
            f"starting_stacks={self.starting_stacks}, \n"
            f"states={self.states}, \n"
            f"actions={self.actions})\n"
        )

    @classmethod
    def from_hand_history(cls, hand_history):
        hand = cls(hand_history)
        hand.parse()
        return hand

    @classmethod
    def from_string(cls, hand_history_text: str):
        hand_history = HandHistory.loads(hand_history_text)
        return cls.from_hand_history(hand_history)

    @classmethod
    def from_file(cls, file_path):
        with Path(file_path).expanduser().resolve().open("rb") as file_obj:
            hand_history = HandHistory.load(file_obj)
        return cls.from_hand_history(hand_history)


    def _player_ids(self):
        return list(self.player_names)

    def _player_name_from_token(self, player_token):
        return self.seat_to_player.get(player_token, player_token)

    def _position_name(self, player):
        idx = self.player_names.index(player)
        if idx == 0:
            return 'sb'
        if idx == 1:
            return 'bb'
        position_map = {
            3: 'button',
            4: ['utg', 'button'],
            5: ['utg', 'co', 'button'],
            6: ['utg', 'hj', 'co', 'button'],
        }
        remaining = position_map.get(self._num_players, [])
        return remaining[idx - 2] if idx - 2 < len(remaining) else f'seat_{idx + 1}'

    def _players_in_hand_list(self):
        return [(p, self._players_in_hand[p]) for p in self._player_ids()]

    def set_hand_range_vector(self, observer: str, target: str, class_distribution: dict[str, float]) -> np.ndarray:
        if observer == target:
            raise ValueError("Observer and target must be different players.")
        if observer not in self.hand_range:
            raise KeyError(f"Unknown observer {observer!r}.")
        if target not in self.hand_range[observer]:
            raise KeyError(f"Unknown target {target!r} for observer {observer!r}.")

        classes = all_169_classes()
        vector = np.array([class_distribution.get(hand_class, 0.0) for hand_class in classes], dtype=float)
        total = vector.sum()
        if total > 0:
            vector = vector / total
        self.hand_range[observer][target] = vector
        return vector

    def next_player(self, curr_player):
        idx = self.player_names.index(curr_player)
        for i in range(1, self._num_players + 1):
            next_idx = (idx + i) % self._num_players
            p = self.player_names[next_idx]
            if self._players_in_hand[p]:
                return p
        return None

    def _first_to_act_preflop(self):
        # Assumes seat order is SB, BB, then clockwise.
        if self._num_players == 2:
            return self.player_names[0]
        return self.player_names[2]

    def _first_to_act_postflop(self):
        for p in self._player_ids():
            if self._players_in_hand[p]:
                return p
        return None

    def _active_hand_strength_map(self):
        """
        Return current hand strength only for players still in the hand.
        Preflop or missing hole cards -> None.
        """
        active_strengths = {}

        for p in self._player_ids():
            if not self._players_in_hand[p]:
                continue

            hole = self.hole_cards[p]
            if not hole or not self._community_cards:
                active_strengths[p] = None
                continue

            active_strengths[p] = poker_hand_mapper(hole, self._community_cards)

        return active_strengths

    def _append_state(self, player_to_act):
        self.states[self._street].append(State(
            player_order=self.player_order.copy(),
            community_cards=self._community_cards,
            betting_history=self._betting_history_this_street.copy(),
            player_to_act=player_to_act,
            players_in_hand=self._players_in_hand_list(),
            current_stacks=self.stacks.copy(),
            pot_size=self._pot,
            hand_strength_map=self._active_hand_strength_map(),
        ))

    def _reset_round_state_for_new_street(self):
        self._current_bet = 0
        self._money_in_round = {
            player: 0 for player in self.player_names
        }
        self._betting_history_this_street = []
        self._street_action_level = 0

    def _post_blinds(self):
        # Only once, at hand initialization
        if self._num_players >= 1:
            sb_player = self.player_names[0]
            self.stacks[sb_player] -= self._sb
            self._money_in_round[sb_player] = self._sb
            self._pot += self._sb

        if self._num_players >= 2:
            bb_player = self.player_names[1]
            self.stacks[bb_player] -= self._bb
            self._money_in_round[bb_player] = self._bb
            self._pot += self._bb

        self._current_bet = self._bb

    def _validate_nonnegative_stack(self, player):
        if self.stacks[player] < 0:
            raise ValueError(
                f"Negative stack for {player}: {self.stacks[player]}"
            )

    def _validate_stack_consistency(self):
        total_start = sum(self.starting_stacks.values())
        total_now = sum(self.stacks.values()) + self._pot
        if total_now != total_start:
            raise ValueError(
                f"Chip conservation violated: "
                f"start={total_start}, now_stacks_plus_pot={total_now}"
            )

    def start_hand(self):
        if self._initialized:
            raise ValueError("Hand already initialized")

        self._initialized = True
        self._street = 'pre-flop'
        self._community_cards = ''
        self._pot = 0
        self._players_in_hand = {
            player: True for player in self.player_names
        }

        self._reset_round_state_for_new_street()
        self._post_blinds()

        self._append_state(player_to_act=self._first_to_act_preflop())

    def apply_action(self, raw_action):
        if not self._initialized:
            self.start_hand()

        parts = raw_action.split()

        if parts[0] == 'd' and parts[1] == 'dh':    # hole cards
            player = self._player_name_from_token(parts[2])
            cards = parts[3]
            self.hole_cards[player] = cards
            return

        if parts[0] == 'd' and parts[1] == 'db':    # board cards
            new_cards = parts[2]

            curr_idx = self.STREETS.index(self._street)
            if curr_idx + 1 >= len(self.STREETS):
                raise ValueError("Too many board deals")

            self._street = self.STREETS[curr_idx + 1]
            self._community_cards += new_cards
            self._reset_round_state_for_new_street()

            self.hand_strength_map = self._active_hand_strength_map()
            self._append_state(player_to_act=self._first_to_act_postflop())

            return

        player = self._player_name_from_token(parts[0])   # player actions
        action_type = parts[1]

        if not self._players_in_hand[player]:
            raise ValueError(f"{player} acted after folding")

        amount = 0

        if action_type == 'f':
            self._players_in_hand[player] = False

        elif action_type == 'cc':
            amount = self._current_bet
            contribution = amount - self._money_in_round[player]
            if contribution < 0:
                raise ValueError(
                    f"Negative call contribution for {player}: "
                    f"{contribution}"
                )
            self.stacks[player] -= contribution
            self._money_in_round[player] = amount
            self._pot += contribution
            self._validate_nonnegative_stack(player)

        elif action_type == 'cbr':
            amount = int(parts[2])
            contribution = amount - self._money_in_round[player]
            if contribution <= 0:
                raise ValueError(
                    f"Non-positive raise/bet contribution for {player}: "
                    f"{contribution}"
                )
            self.stacks[player] -= contribution
            self._money_in_round[player] = amount
            self._current_bet = amount
            self._pot += contribution
            self._validate_nonnegative_stack(player)

            self._street_action_level += 1

        elif action_type == 'sm':
            return

        else:
            raise ValueError(f"Unknown action type: {action_type}")
        

        action_bucket = classify(action_type, self._pot, bet=amount)

        self._betting_history_this_street.append((player, (action_bucket, self._street_action_level), amount))

        t = self._street_action_index[self._street]
        self.actions[self._street][t] = (player, (action_bucket, self._street_action_level), amount)
        self._street_action_index[self._street] += 1

        next_to_act = self.next_player(player)

        self._validate_stack_consistency()
        self._append_state(player_to_act=next_to_act)


    def parse(self):
        self.start_hand()
        for raw_action in self.hand_history.actions:
            self.apply_action(raw_action)


class Session:
    def __init__(self, file_path):
        self.file_path = Path(file_path).expanduser().resolve()
        self.hands = []

        if not self.file_path.exists():
            raise FileNotFoundError(f"Session path does not exist: {self.file_path}")
        if not self.file_path.is_dir():
            raise NotADirectoryError(f"Session path is not a directory: {self.file_path}")

    @staticmethod
    def _extract_number(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError as e:
            raise ValueError(
                f"Expected numeric filename stem, got: {path.name}"
            ) from e

    def parse(self) -> list[Hand]:
        self.hands.clear()

        files = sorted(
            (p for p in self.file_path.iterdir() if p.is_file() and p.suffix == '.phh'),
            key=self._extract_number
        )

        for path in files:
            self.hands.append(Hand.from_file(path))

        return self.hands


def parse_single_hand(hand_history_text: str) -> Hand:
    return Hand.from_string(hand_history_text)
