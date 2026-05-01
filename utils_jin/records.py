"""Decision-record extraction from parsed PHH hands.

The filter and EM code should not operate directly on the parser's nested
``Hand.states`` and ``Hand.actions`` dictionaries.  This module converts them
into flat records: one row per player decision.

Each record contains:

* the public state before the action;
* the compact model state key;
* the re-bucketed action;
* true hole-card labels when available.

Those true labels are for supervised baselines and evaluation only.  The filter
itself only uses ``state_key`` and ``action_bucket``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .action_encoder import ActionEncoding, encode_action
from .cards import combo_to_class, split_cards
from .compat import load_legacy_session_class
from .state_encoder import StateKey, encode_state


@dataclass(frozen=True)
class DecisionRecord:
    """One model-ready player decision."""

    session_id: str
    hand_id: str
    hand_index: int
    player_id: str
    street: str
    decision_index: int
    board: str
    pot_before: float
    stack_before: float
    active_player_count: int
    state_key: StateKey
    state_key_str: str
    action_bucket: int
    action_label: str
    amount_to: float
    contribution: float
    pot_fraction: float | None
    legacy_action_bucket: int
    hole_cards: str | None
    hand_class: str | None
    strength_bucket: str | None

    def to_dict(self) -> dict:
        """Return a CSV/DataFrame-friendly dictionary."""

        data = asdict(self)
        data["state_key"] = self.state_key.as_tuple()
        return data


def _active_count(state) -> int:
    return sum(1 for _player, active in state.players_in_hand if active)


def _safe_hand_class(hole_cards: str | None) -> str | None:
    """Return 169 class for known hole cards, else ``None``."""

    if not hole_cards:
        return None
    if "?" in hole_cards:
        return None
    try:
        return combo_to_class(split_cards(hole_cards))
    except ValueError:
        return None


def _strength_bucket_from_state(state, player: str) -> str | None:
    """Read the true post-flop strength bucket stored by the legacy parser."""

    value = state.hand_strength_map.get(player)
    if not value:
        return None
    return value.get("bucket")


def _hand_id(hand, fallback_index: int) -> str:
    """Extract a stable hand id from the legacy ``HandHistory`` object."""

    raw = getattr(hand.hand_history, "hand", None)
    if raw is None:
        return str(fallback_index)
    return str(raw)


def iter_hand_decision_records(
    hand,
    session_id: str,
    hand_index: int,
    include_postflop: bool = True,
) -> Iterable[DecisionRecord]:
    """Yield decision records for one parsed legacy ``Hand`` object."""

    hand_id = _hand_id(hand, hand_index)

    for street in hand.STREETS:
        if not include_postflop and street != "pre-flop":
            continue

        actions = hand.actions[street]
        states = hand.states[street]

        for decision_index in sorted(actions):
            if decision_index >= len(states):
                raise IndexError(
                    f"Missing pre-action state for {session_id=} {hand_id=} "
                    f"{street=} {decision_index=}"
                )

            state = states[decision_index]
            legacy_action = actions[decision_index]
            player = legacy_action[0]

            action: ActionEncoding = encode_action(state, hand, street, legacy_action)
            state_key = encode_state(state, hand, street, player, action_encoding=action)

            hole_cards = hand.hole_cards.get(player) or None
            hand_class = _safe_hand_class(hole_cards)
            strength_bucket = _strength_bucket_from_state(state, player)

            yield DecisionRecord(
                session_id=session_id,
                hand_id=hand_id,
                hand_index=hand_index,
                player_id=player,
                street=street,
                decision_index=decision_index,
                board=state.community_cards,
                pot_before=float(state.pot_size),
                stack_before=float(state.current_stacks.get(player, 0.0)),
                active_player_count=_active_count(state),
                state_key=state_key,
                state_key_str=state_key.as_string(),
                action_bucket=action.bucket,
                action_label=action.label,
                amount_to=action.amount_to,
                contribution=action.contribution,
                pot_fraction=action.pot_fraction,
                legacy_action_bucket=action.legacy_bucket,
                hole_cards=hole_cards,
                hand_class=hand_class,
                strength_bucket=strength_bucket,
            )


def extract_session_records(
    session_path: str | Path,
    include_postflop: bool = True,
    max_hands: int | None = None,
) -> list[DecisionRecord]:
    """Parse one session directory and return flat decision records."""

    session_path = Path(session_path)
    Session = load_legacy_session_class()
    hands = Session(session_path).parse()
    if max_hands is not None:
        hands = hands[:max_hands]

    records: list[DecisionRecord] = []
    for hand_index, hand in enumerate(hands):
        records.extend(
            iter_hand_decision_records(
                hand,
                session_id=session_path.name,
                hand_index=hand_index,
                include_postflop=include_postflop,
            )
        )
    return records


def extract_many_session_records(
    root_path: str | Path,
    include_postflop: bool = True,
    max_sessions: int | None = None,
    max_hands_per_session: int | None = None,
) -> list[DecisionRecord]:
    """Extract records from every session directory under ``root_path``."""

    root = Path(root_path)
    session_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
    if max_sessions is not None:
        session_dirs = session_dirs[:max_sessions]

    records: list[DecisionRecord] = []
    for session_dir in session_dirs:
        records.extend(
            extract_session_records(
                session_dir,
                include_postflop=include_postflop,
                max_hands=max_hands_per_session,
            )
        )
    return records


def group_records_by_player_hand(records: Iterable[DecisionRecord]) -> dict[tuple[str, str, str], list[DecisionRecord]]:
    """Group records by ``(session_id, hand_id, player_id)``."""

    grouped: dict[tuple[str, str, str], list[DecisionRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.session_id, record.hand_id, record.player_id)].append(record)
    return dict(grouped)


def preflop_only(records: Iterable[DecisionRecord]) -> list[DecisionRecord]:
    """Return only pre-flop records."""

    return [record for record in records if record.street == "pre-flop"]

