"""Helpers for tabular evaluation rows (session / hand indexing, board strings, seats)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.parse import Hand


def board_at_street_end(hand: Hand, street: str) -> str:
    sts = hand.states.get(street, [])
    if not sts:
        return ""
    return sts[-1].community_cards or ""


def session_phh_files(session_dir: Path) -> List[Path]:
    """Same ordering as ``Session.parse()``."""

    def stem_key(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            raise ValueError(f"Expected numeric .phh stem, got {p.name}") from None

    return sorted(
        (p for p in session_dir.iterdir() if p.is_file() and p.suffix.lower() == ".phh"),
        key=stem_key,
    )


def seat_columns(hand: Hand) -> Dict[str, str]:
    """Six columns ``p1``..``p6`` with player names (empty if seat unused)."""
    return {f"p{i}": hand.seat_to_player.get(f"p{i}", "") for i in range(1, 7)}


def hand_number_from_path(phh_path: Path) -> Any:
    stem = phh_path.stem
    return int(stem) if stem.isdigit() else stem
