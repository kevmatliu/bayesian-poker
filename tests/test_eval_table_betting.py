"""Tests for betting-line helpers on parsed :class:`Hand` objects."""

import unittest
from pathlib import Path

from utils.eval.table import (
    betting_history_on_street,
    betting_history_up_to_street_end,
    format_betting_action,
    format_betting_history_serial,
    player_alive_at_street_end,
)
from utils.parse import Hand


class TestBettingFormat(unittest.TestCase):
    def test_format_betting_action(self) -> None:
        self.assertIn("fold", format_betting_action("A", (0, 0), 0))
        self.assertIn("call/check", format_betting_action("B", (1, 0), 100))
        self.assertIn("raise to", format_betting_action("C", (2, 1), 300))

    def test_format_betting_history_serial(self) -> None:
        s = format_betting_history_serial([("X", (0, 0), 0), ("Y", (2, 1), 400)])
        self.assertIn("→", s)
        self.assertIn("X", s)
        self.assertIn("Y", s)


class TestBettingFromPhh(unittest.TestCase):
    def test_flop_has_preflop_in_prior(self) -> None:
        hand = Hand.from_file(Path("pluribus/99/1.phh"))
        prior = betting_history_up_to_street_end(hand, "flop")
        this = betting_history_on_street(hand, "flop")
        self.assertTrue(len(prior) > 0 or len(this) > 0)

    def test_player_alive_at_street_end_unknown_player(self) -> None:
        hand = Hand.from_file(Path("pluribus/99/1.phh"))
        self.assertFalse(player_alive_at_street_end(hand, "pre-flop", "NonexistentPlayer"))

    def test_player_alive_at_street_end_reaches_final_snapshot(self) -> None:
        hand = Hand.from_file(Path("pluribus/99/1.phh"))
        for street in ("pre-flop", "flop", "turn", "river"):
            if not hand.states.get(street):
                continue
            last = hand.states[street][-1].players_in_hand or []
            expected_map = {name: bool(ok) for name, ok in last}
            for name in hand.player_names:
                alive = player_alive_at_street_end(hand, street, name)
                self.assertEqual(alive, bool(expected_map.get(name, False)))


if __name__ == "__main__":
    unittest.main()
