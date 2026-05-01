from pathlib import Path
import os

from pokerkit import HandHistory

from hand_map import poker_hand_mapper
from action_map import classify

class State:
    def __init__(
        self,
        community_cards,
        betting_history,
        player_to_act,
        players_in_hand,
        current_stacks,
        pot_size,
        hand_strength_map,
    ):
        self.community_cards = community_cards
        self.betting_history = betting_history
        self.player_to_act = player_to_act
        self.players_in_hand = players_in_hand
        self.current_stacks = current_stacks
        self.pot_size = pot_size
        self.hand_strength_map = hand_strength_map

    def __repr__(self):
        return (
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

        self._bb = hand_history.blinds_or_straddles[1]
        self._sb = hand_history.blinds_or_straddles[0]

        self._num_players = len(hand_history.players)
        self.player_names = hand_history.players
        self.hand_strength_map = {
            f'p{i + 1}': {} for i in range(self._num_players)
        }

        self.hole_cards = {
            f'p{i + 1}': '' for i in range(self._num_players)
        }

        self.player_positions = {
            f'p{i + 1}': self.player_names[i] for i in range(self._num_players)
        }

        self.starting_stacks = {
            f'p{i + 1}': hand_history.starting_stacks[i]
            for i in range(self._num_players)
        }

        self.stacks = self.starting_stacks.copy()

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
            f'p{i + 1}': True for i in range(self._num_players)
        }

        self._street = 'pre-flop'
        self._community_cards = ''
        self._pot = 0

        self._current_bet = 0              # current amount to call on this street
        self._money_in_round = {
            f'p{i + 1}': 0 for i in range(self._num_players)
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


    def _player_ids(self):
        return [f'p{i + 1}' for i in range(self._num_players)]

    def _players_in_hand_list(self):
        return [(p, self._players_in_hand[p]) for p in self._player_ids()]

    def next_player(self, curr_player):
        idx = int(curr_player[1:]) - 1
        for i in range(1, self._num_players + 1):
            next_idx = (idx + i) % self._num_players
            p = f'p{next_idx + 1}'
            if self._players_in_hand[p]:
                return p
        return None

    def _first_to_act_preflop(self):
        # Assumes p1=SB, p2=BB, action starts left of BB
        if self._num_players == 2:
            return 'p1'
        return 'p3'

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
            f'p{i + 1}': 0 for i in range(self._num_players)
        }
        self._betting_history_this_street = []
        self._street_action_level = 0

    def _post_blinds(self):
        # Only once, at hand initialization
        if self._num_players >= 1:
            self.stacks['p1'] -= self._sb
            self._money_in_round['p1'] = self._sb
            self._pot += self._sb

        if self._num_players >= 2:
            self.stacks['p2'] -= self._bb
            self._money_in_round['p2'] = self._bb
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
            f'p{i + 1}': True for i in range(self._num_players)
        }

        self._reset_round_state_for_new_street()
        self._post_blinds()

        self._append_state(player_to_act=self._first_to_act_preflop())

    def apply_action(self, raw_action):
        if not self._initialized:
            self.start_hand()

        parts = raw_action.split()

        if parts[0] == 'd' and parts[1] == 'dh':    # hole cards
            player = parts[2]
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

        player = parts[0]   # player actions
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
            with path.open('rb') as f:
                hh = HandHistory.load(f)
                hand = Hand(hh)
                hand.parse()
                self.hands.append(hand)

        return self.hands