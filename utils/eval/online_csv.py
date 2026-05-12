"""Load ``online_range_history.csv``-style range exports and enrich with hand metadata from Pluribus."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

import pandas as pd

from utils.eval.brier import brier_postflop1326, brier_preflop_from_combo1326
from utils.eval.logutil import eval_log
from utils.eval.strength import (
    expected_made_and_draw_mc,
    made_percentile_calibration_stats,
    made_percentile_vector_1326,
)
from utils.eval.table import board_at_street_end, seat_columns
from utils.filter import all_combo_keys, combo_key
from utils.parse import Hand
from utils.parse import parse_card, parse_cards
from utils.strength.postflop import draw_strength_from_hand
from utils.strength.preflop import get_equivalence_class

META_COLUMNS: Tuple[str, ...] = ("session", "hand_number", "street", "observer", "target")

# Prefix for per-combo made-strength percentile columns in :func:`add_calibration_columns`.
# (Distinct from raw combo-key probability column names, which are exactly ``combo_key`` strings.)
MADE_STRENGTH_PCT_COLUMN_PREFIX = "made_strength_pct__"


def meta_columns_present(df: pd.DataFrame) -> None:
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"online range CSV missing columns: {missing}")


def combo_probability_columns(df: pd.DataFrame) -> list[str]:
    """Canonical 1,326 combo **probability** columns present in ``df`` (``all_combo_keys()`` order)."""
    keys = list(all_combo_keys())
    return [c for c in keys if c in df.columns]


def combo_made_percentile_column_names(
    combo_order: Optional[Iterable[str]] = None,
) -> List[str]:
    """Column names for per-combo made percentiles (see :func:`add_calibration_columns`)."""
    keys = list(combo_order) if combo_order is not None else list(all_combo_keys())
    pfx = MADE_STRENGTH_PCT_COLUMN_PREFIX
    return [f"{pfx}{k}" for k in keys]


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


def _board_cards_list(board_str: str) -> list:
    s = (board_str or "").strip()
    if len(s) < 6:
        return []
    return parse_cards([s[i : i + 2] for i in range(0, len(s), 2)])


def _hole_cards_list(hole4: str) -> list:
    h = (hole4 or "").strip()
    if len(h) != 4:
        return []
    return parse_cards([h[0:2], h[2:4]])


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
    Append calibration columns including Brier, strength summaries, and made-percentile
    scores under the row's combo distribution.

    ``actual_made_pct`` uses :func:`made_percentile_vector_1326` (cached fast path) and the
    static 1,326 row order, not a per-call legacy percentile loop.

    Also appends **1,326** columns ``made_strength_pct__<combo_key>`` — the exact made
    strength percentile for every combo in the static table on the row's board (NaN
    preflop or when the board is too short). Together with the probability columns this
    yields **≥ 2,652** combo-related columns plus metadata and other calibration fields.

    Postflop-only extras: ``made_dist_mu``, ``made_dist_sigma``, ``made_pct_z``,
    ``made_pct_abs_z``, ``combo_nll``, ``made_pct_midrank``, ``made_pct_cdf_le`` (see
    :func:`made_percentile_calibration_stats`).
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
    made_mu = []
    made_sig = []
    made_z = []
    made_abs_z = []
    combo_nll = []
    made_midrank = []
    made_cdf_le = []
    combo_to_ix = {k: i for i, k in enumerate(order)}
    made_per_combo = np.full((n, 1326), np.nan, dtype=np.float64)
    perc_by_board: dict[str, np.ndarray] = {}
    for i, (_, row) in enumerate(out.iterrows()):
        ri = i + 1
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
            hole_cards = _hole_cards_list(hole)
            b_cards = _board_cards_list(board)
            if len(hole_cards) == 2 and len(b_cards) >= 3:
                a2 = float(draw_strength_from_hand(hole_cards, b_cards))
            else:
                a2 = float("nan")

            if board not in perc_by_board:
                perc_by_board[board] = made_percentile_vector_1326(board)
            perc_v = perc_by_board[board]
            made_per_combo[i, :] = perc_v
            p_v = np.array([float(row[c]) for c in order], dtype=np.float64)
            tc = _true_combo_key(hole) if len(hole) == 4 else ""
            j = combo_to_ix.get(tc, -1) if tc else -1
            if j >= 0 and np.isfinite(perc_v[j]):
                a1 = float(perc_v[j])
                mu, sig, z, az, nll, mid, cle = made_percentile_calibration_stats(
                    p_v, perc_v, j
                )
                made_mu.append(mu)
                made_sig.append(sig)
                made_z.append(z)
                made_abs_z.append(az)
                combo_nll.append(nll if np.isfinite(nll) else float("nan"))
                made_midrank.append(mid)
                made_cdf_le.append(cle)
            else:
                a1 = float("nan")
                nan_m = float("nan")
                made_mu.append(nan_m)
                made_sig.append(nan_m)
                made_z.append(nan_m)
                made_abs_z.append(nan_m)
                combo_nll.append(nan_m)
                made_midrank.append(nan_m)
                made_cdf_le.append(nan_m)
        else:
            e1 = e2 = a1 = a2 = float("nan")
            made_mu.append(float("nan"))
            made_sig.append(float("nan"))
            made_z.append(float("nan"))
            made_abs_z.append(float("nan"))
            combo_nll.append(float("nan"))
            made_midrank.append(float("nan"))
            made_cdf_le.append(float("nan"))

        emade.append(e1)
        edraw.append(e2)
        amade.append(a1)
        adraw.append(a2)

    out["brier"] = brs
    out["expected_made_pct"] = emade
    out["expected_draw"] = edraw
    out["actual_made_pct"] = amade
    out["actual_draw"] = adraw
    out["made_dist_mu"] = made_mu
    out["made_dist_sigma"] = made_sig
    out["made_pct_z"] = made_z
    out["made_pct_abs_z"] = made_abs_z
    out["combo_nll"] = combo_nll
    out["made_pct_midrank"] = made_midrank
    out["made_pct_cdf_le"] = made_cdf_le
    perc_cols = combo_made_percentile_column_names(order)
    out = pd.concat(
        [out, pd.DataFrame(made_per_combo, columns=perc_cols, index=out.index)],
        axis=1,
    )
    eval_log(verbose, "add_calibration_columns: done")
    return out
