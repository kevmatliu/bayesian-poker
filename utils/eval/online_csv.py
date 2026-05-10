"""Load ``online_range_history.csv``-style range exports and enrich with hand metadata from Pluribus."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

import pandas as pd

from utils.eval.brier import brier_postflop1326, brier_preflop_from_combo1326
from utils.eval.logutil import eval_log
from utils.eval.strength import actual_made_and_draw, expected_made_and_draw_mc
from utils.eval.table import board_at_street_end, seat_columns
from utils.filter import all_combo_keys, combo_key
from utils.parse import Hand
from utils.strength.common import parse_card
from utils.strength.preflop import get_equivalence_class

META_COLUMNS: Tuple[str, ...] = ("session", "hand_number", "street", "observer", "target")


def meta_columns_present(df: pd.DataFrame) -> None:
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"online range CSV missing columns: {missing}")


def combo_probability_columns(df: pd.DataFrame) -> list[str]:
    """All columns aside from fixed metadata (assumes no extra columns yet)."""
    meta = set(META_COLUMNS)
    return [c for c in df.columns if c not in meta]


def load_hand_pluribus(
    pluribus_root: Path,
    session: str,
    hand_number,
) -> Optional[Hand]:
    """Resolve ``pluribus_root / session / {hand_number}.phh`` and parse."""
    stem = str(int(hand_number)) if str(hand_number).isdigit() else str(hand_number)
    path = pluribus_root / str(session) / f"{stem}.phh"
    if not path.is_file():
        return None
    return Hand.from_file(path)


def enrich_online_range_dataframe(
    df: pd.DataFrame,
    pluribus_root: Path,
    *,
    verbose: bool = False,
    progress_every: int = 100,
) -> pd.DataFrame:
    """
    Add ``community_cards``, ``target_hole_cards``, and ``p1``..``p6`` by joining
    each ``(session, hand_number)`` to the corresponding ``.phh`` under ``pluribus_root``.

    ``community_cards`` is cumulative board text for the row's ``street`` (empty on ``pre-flop``).

    With ``verbose=True``, prints progress every ``progress_every`` rows (minimum 1).
    """
    meta_columns_present(df)
    n = len(df)
    eval_log(
        verbose,
        f"enrich_online_range_dataframe: {n} rows | pluribus_root={pluribus_root}",
    )
    step = max(1, int(progress_every))
    out = df.copy()
    cache: dict[tuple[str, int | str], Optional[Hand]] = {}

    def _hn_key(hn) -> int | str:
        if isinstance(hn, int):
            return hn
        s = str(hn)
        return int(s) if s.isdigit() else hn

    def get_hand(sess: str, hn) -> Optional[Hand]:
        key = (str(sess), _hn_key(hn))
        if key not in cache:
            cache[key] = load_hand_pluribus(pluribus_root, sess, hn)
        return cache[key]

    comm: list[str] = []
    holes: list[str] = []
    seats: list[list[str]] = []

    for i, (_, row) in enumerate(out.iterrows(), start=1):
        if verbose and (i == 1 or i == n or i % step == 0):
            eval_log(verbose, f"enrich … row {i}/{n}")
        hand = get_hand(row["session"], row["hand_number"])
        tgt = str(row["target"])
        st = str(row["street"])
        if hand is None:
            comm.append("")
            holes.append("")
            seats.append([""] * 6)
            continue
        if st == "pre-flop":
            comm.append("")
        else:
            comm.append(board_at_street_end(hand, st))
        holes.append(str(hand.hole_cards.get(tgt, "") or ""))
        sc = seat_columns(hand)
        seats.append([sc.get(f"p{i}", "") for i in range(1, 7)])

    out["community_cards"] = comm
    out["target_hole_cards"] = holes
    for i in range(1, 7):
        out[f"p{i}"] = [s[i - 1] for s in seats]
    eval_log(
        verbose,
        f"enrich_online_range_dataframe: done ({len(cache)} unique session/hand hands cached)",
    )
    return out


def _true_preflop_class(hole4: str) -> str:
    if len(hole4) != 4:
        return ""
    return get_equivalence_class([parse_card(hole4[0:2]), parse_card(hole4[2:4])])


def _true_combo_key(hole4: str) -> str:
    if len(hole4) != 4:
        return ""
    return combo_key(parse_card(hole4[0:2]), parse_card(hole4[2:4]))


def add_calibration_columns(
    df: pd.DataFrame,
    combo_cols: Iterable[str],
    *,
    strength_mc_samples: int = 96,
    strength_rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
    progress_every: int = 50,
) -> pd.DataFrame:
    """
    Append ``brier``, ``expected_made_pct``, ``expected_draw``, ``actual_made_pct``, ``actual_draw``.

    Preflop rows: Brier via 1,326 → 169 collapse. Postflop: full 1,326 Brier.

    Expected strength uses :func:`expected_made_and_draw_mc` (see ``strength_mc_samples``).

    With ``verbose=True``, prints progress every ``progress_every`` rows. Per-row strength/Brier
    internals stay quiet unless you call those functions directly with ``verbose=True``.
    """
    meta_columns_present(df)
    n = len(df)
    eval_log(
        verbose,
        f"add_calibration_columns: {n} rows | strength_mc_samples={strength_mc_samples}",
    )
    step = max(1, int(progress_every))
    combo_list = list(combo_cols)
    order = list(all_combo_keys())
    if len(combo_list) != len(order) or set(combo_list) != set(order):
        raise ValueError(
            "Combo columns must match ``all_combo_keys()`` exactly (same 1,326 keys as export)."
        )
    # Use canonical key order for the Brier / probability vector
    out = df.copy()
    brs = []
    emade = []
    edraw = []
    amade = []
    adraw = []
    for ri, (_, row) in enumerate(out.iterrows(), start=1):
        if verbose and (ri == 1 or ri == n or ri % step == 0):
            eval_log(verbose, f"calibration … row {ri}/{n}")
        dist = {c: float(row[c]) for c in order}
        hole = str(row.get("target_hole_cards", "") or "")
        street = str(row["street"])
        board = str(row.get("community_cards", "") or "")

        if len(hole) == 4:
            if street == "pre-flop":
                brs.append(
                    brier_preflop_from_combo1326(dist, _true_preflop_class(hole), verbose=False)
                )
            else:
                tc = _true_combo_key(hole)
                brs.append(
                    brier_postflop1326(dist, order, tc, verbose=False) if tc else float("nan")
                )
        else:
            brs.append(float("nan"))

        if len(board) >= 6:
            e1, e2 = expected_made_and_draw_mc(
                dist,
                board,
                n_samples=strength_mc_samples,
                rng=strength_rng,
                verbose=False,
            )
            a1, a2 = actual_made_and_draw(hole, board, verbose=False)
        else:
            e1 = e2 = a1 = a2 = float("nan")
        emade.append(e1)
        edraw.append(e2)
        amade.append(a1)
        adraw.append(a2)

    out["brier"] = brs
    out["expected_made_pct"] = emade
    out["expected_draw"] = edraw
    out["actual_made_pct"] = amade
    out["actual_draw"] = adraw
    eval_log(verbose, "add_calibration_columns: done")
    return out
