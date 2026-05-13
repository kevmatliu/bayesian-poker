"""Helpers for tabular evaluation rows (session / hand indexing, board strings, seats)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from utils.parse import Hand


def format_betting_action(
    player: str,
    bucket_and_level: Tuple[int, int],
    amount: int | float,
) -> str:
    """Human-readable token for one ``State.betting_history`` entry (see :class:`Hand`)."""
    b = int(bucket_and_level[0])       # coarse action bucket (fold/call/raise)
    amt = int(amount)                  # chip amount associated with the line
    if b == 0:                         # fold code path
        return f"{player} fold"
    if b == 1:                         # call / check bucket
        return f"{player} call/check (to {amt})"
    return f"{player} raise to {amt}"  # aggressive action with target to-size


def format_betting_history_serial(
    history: Sequence[Tuple[str, Tuple[int, int], int | float]],
    *,
    sep: str = " → ",
) -> str:
    """Join betting tuples (player, (bucket, level), amount) into one line."""
    if not history:                                                    # nothing to stringify
        return ""
    parts = [format_betting_action(p, bl, a) for p, bl, a in history]  # map each tuple to text
    return sep.join(parts)                                             # left-to-right narrative of the street


def betting_history_up_to_street_end(hand: Hand, street: str) -> str:
    """Completed betting on all streets strictly before ``street`` (one line per prior street)."""
    streets = list(Hand.STREETS)                                          # canonical ordering preflop→river
    if street not in streets:                                             # unknown label → no history
        return ""
    i = streets.index(street)                                             # cut index for prior streets only
    lines: list[str] = []                                                 # accumulate one formatted line per completed street
    for s in streets[:i]:                                                 # strictly earlier streets
        sts = hand.states.get(s) or []                                    # timeline of snapshots on that street
        if not sts:                                                       # hand never reached that street
            continue
        hist = sts[-1].betting_history or []                              # final betting line once action closed
        if not hist:                                                      # degenerate all-in walk or missing data
            continue
        label = s.replace("-", " ")                                       # nicer display token (``pre flop`` style)
        lines.append(f"[{label}] {format_betting_history_serial(hist)}")  # bracketed street prefix
    return "\n".join(lines)                                               # multi-line block for CSV / notebook cells


def betting_history_on_street(hand: Hand, street: str) -> str:
    """Betting line for ``street`` using the final snapshot on that street (full street runout)."""
    sts = hand.states.get(street) or []                                  # all decision snapshots on this street
    if not sts:                                                          # street not dealt yet
        return ""
    return format_betting_history_serial(sts[-1].betting_history or [])  # completed line only


def board_at_street_end(hand: Hand, street: str) -> str:
    sts = hand.states.get(street, [])     # snapshots carrying ``community_cards`` string
    if not sts:                           # no board by end of this street
        return ""
    return sts[-1].community_cards or ""  # concatenated two-char cards (e.g. ``AhKdQc``)


def session_phh_files(session_dir: Path) -> List[Path]:
    """Same ordering as ``Session.parse()``."""

    def stem_key(p: Path) -> int:
        try:
            return int(p.stem)                                                       # numeric ``123.phh`` ordering
        except ValueError:
            raise ValueError(f"Expected numeric .phh stem, got {p.name}") from None  # strict contract for sessions

    return sorted(
        (p for p in session_dir.iterdir() if p.is_file() and p.suffix.lower() == ".phh"),  # only hand files
        key=stem_key,                                                                      # ascending numeric order
    )


def seat_columns(hand: Hand) -> Dict[str, str]:
    """Six columns ``p1``..``p6`` with player names (empty if seat unused)."""
    return {f"p{i}": hand.seat_to_player.get(f"p{i}", "") for i in range(1, 7)}  # fixed-width schema for spreadsheets


def hand_number_from_path(phh_path: Path) -> Any:
    stem = phh_path.stem                          # filename without extension
    return int(stem) if stem.isdigit() else stem  # int when possible for sorting, else raw string id
