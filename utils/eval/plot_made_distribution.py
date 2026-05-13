"""Weighted made/draw distribution plot for one calibrated range-history row."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from utils.eval.online_csv import load_hand_pluribus
from utils.eval.strength import made_percentile_vector_1326
from utils.eval.table import betting_history_on_street, betting_history_up_to_street_end
from utils.filter.postflop import parse_combo_key
from utils.parse import Hand, parse_cards
from utils.strength.fast_eval import all_combo_keys_fast
from utils.strength.postflop import draw_strength_from_hand


def _subset_row(
    df: pd.DataFrame,
    *,
    session: str,
    hand_number: Any,
    observer: str,
    target: str,
    street: str,
) -> pd.Series:
    m = (
        (df["session"].astype(str) == str(session))                         # match session id loosely typed
        & (df["hand_number"].astype(str) == str(hand_number))               # normalize numeric/string hand ids
        & (df["observer"].astype(str) == str(observer))                     # observer seat filter
        & (df["target"].astype(str) == str(target))                         # target seat filter
        & (df["street"].astype(str) == str(street))                         # street label filter
    )
    sub = df.loc[m]                                                         # boolean mask row subset
    if sub.empty:                                                           # no matching calibration row
        raise ValueError(
            f"No row for session={session!r} hand_number={hand_number!r} "  # include key fields for debugging
            f"observer={observer!r} target={target!r} street={street!r}"
        )
    if len(sub) > 1:                                                        # ambiguous key should not happen in clean exports
        raise ValueError(f"Expected one row, got {len(sub)}")               # guard duplicate keys
    return sub.iloc[0]                                                      # single series for plotting


def _resolve_betting_texts(
    row: pd.Series,
    hand: Optional[Hand],
    pluribus_root: Optional[Path],
    session: str,
    hand_number: Any,
    street: str,
) -> Tuple[str, str]:
    prior_col = str(row.get("betting_prior_streets", "") or "")                 # prefer precomputed prior-street text
    this_col = str(row.get("betting_this_street", "") or "")                    # prefer precomputed in-street text
    if prior_col or this_col:                                                   # enriched dataframe path
        return prior_col, this_col                                              # trust csv/joined columns
    h = hand                                                                    # optional caller-supplied parsed hand
    if h is None and pluribus_root is not None:                                 # fall back to disk parse when possible
        h = load_hand_pluribus(Path(pluribus_root), str(session), hand_number)  # load phh if missing
    if h is None:                                                               # still unavailable
        return "", ""                                                           # degrade gracefully: no betting caption
    return (
        betting_history_up_to_street_end(h, str(street)),                       # completed prior action string
        betting_history_on_street(h, str(street)),                              # in-street action string
    )


def plot_made_distribution(
    df: pd.DataFrame,
    *,
    session: str,
    hand_number: Any,
    observer: str,
    target: str,
    street: str,
    combo_cols: Optional[list[str]] = None,
    pluribus_root: Optional[Path] = None,
    hand: Optional[Hand] = None,
    n_bins_made: int = 40,
    n_bins_draw: int = 40,
    figsize: Tuple[float, float] = (11.0, 7.5),
    text_width: int = 130,
):
    """
    For one ``(session, hand_number, observer, target, street)`` row in a calibrated
    range dataframe, plot weighted made- and draw-strength histograms and vertical
    lines at the realized ``actual_made_pct`` / ``actual_draw``.

    **Betting context** (shown under the axes): if columns ``betting_prior_streets`` and
    ``betting_this_street`` exist (e.g. from :func:`enrich_online_range_dataframe`), those
    strings are rendered. Otherwise, pass ``pluribus_root`` (and optionally ``hand``) to
    format action from the parsed ``.phh``.
    """
    import matplotlib.pyplot as plt

    row = _subset_row(
        df,                                                                                                       # full enriched history
        session=session,                                                                                          # row key
        hand_number=hand_number,                                                                                  # row key
        observer=observer,                                                                                        # row key
        target=target,                                                                                            # row key
        street=street,                                                                                            # row key
    )
    keys = list(combo_cols) if combo_cols is not None else [c for c in all_combo_keys_fast() if c in df.columns]  # resolve 1326 key order
    if len(keys) != 1326:                                                                                         # strict requirement for dense combo vector
        raise ValueError("Need 1,326 combo probability columns (pass combo_cols explicitly if needed)")           # guide caller

    board = str(row.get("community_cards", "") or "").strip()                                                     # cumulative board text for row
    hole = str(row.get("target_hole_cards", "") or "").strip()                                                    # target hole cards for titles

    p = np.array([float(row[c]) for c in keys], dtype=np.float64)                                                 # combo prob vector in keys order
    s = float(p.sum())                                                                                            # normalization factor
    if s > 0:                                                                                                     # avoid divide-by-zero on empty support
        p = p / s                                                                                                 # enforce simplex for plotting weights

    prior_txt, this_txt = _resolve_betting_texts(
        row, hand, pluribus_root, session, hand_number, street                                                    # caption inputs
    )

    fig, (ax_m, ax_d) = plt.subplots(1, 2, figsize=figsize)                                                       # side-by-side made vs draw panels

    if len(board) >= 6 and street != "pre-flop":                                                                  # postflop with at least flop present
        perc = made_percentile_vector_1326(board)                                                                 # nan-masked percentile field
        board_cards = parse_cards([board[i : i + 2] for i in range(0, len(board), 2)])                            # structured board list
        draws = np.full(1326, np.nan, dtype=np.float64)                                                           # parallel draw scores (nan when invalid)
        for j, k in enumerate(keys):                                                                              # fill draw score per combo column order
            ca, cb = parse_combo_key(k)                                                                           # hole cards for this combo key
            try:
                draws[j] = float(draw_strength_from_hand([ca, cb], board_cards))                                  # heuristic draw vs board
            except ValueError:
                draws[j] = float("nan")                                                                           # illegal card interaction -> ignore in plots

        live_m = np.isfinite(perc) & (p > 0)                                                                      # support mask for made histogram
        if np.any(live_m):                                                                                        # skip plotting if no live positive mass
            w_m = p[live_m]                                                                                       # weights on live made support
            x_m = perc[live_m]                                                                                    # percentile samples matching weights
            w_m = w_m / float(w_m.sum())                                                                          # renorm weights after masking
            counts_m, edges_m = np.histogram(
                x_m, bins=int(n_bins_made), range=(0.0, 1.0), weights=w_m, density=False                          # fixed [0,1] support
            )
            c_m = 0.5 * (edges_m[:-1] + edges_m[1:])                                                              # bin centers
            ax_m.bar(c_m, counts_m, width=np.diff(edges_m), align="center", edgecolor="black", linewidth=0.4)     # weighted made hist
        am = float(row.get("actual_made_pct", float("nan")))                                                      # realized made percentile from calibration
        if np.isfinite(am):                                                                                       # draw vertical truth line when known
            ax_m.axvline(am, color="C3", linestyle="-", linewidth=2.0, label=f"actual made={am:.3f}")             # truth marker

        live_d = np.isfinite(draws) & (p > 0)                                                                     # support mask for draw histogram
        if np.any(live_d):                                                                                        # plot only if something to show
            w_d = p[live_d]                                                                                       # draw-axis weights
            x_d = draws[live_d]                                                                                   # draw strengths
            w_d = w_d / float(w_d.sum())                                                                          # normalize masked weights
            d_min = float(np.nanmin(x_d))                                                                         # empirical min for bin range
            d_max = float(np.nanmax(x_d))                                                                         # empirical max for bin range
            if d_max <= d_min:                                                                                    # degenerate range guard
                d_max = d_min + 1e-6                                                                              # widen trivially to satisfy histogram api
            counts_d, edges_d = np.histogram(
                x_d, bins=int(n_bins_draw), range=(d_min, d_max), weights=w_d, density=False                      # data-driven draw span
            )
            c_d = 0.5 * (edges_d[:-1] + edges_d[1:])                                                              # draw bin centers
            ax_d.bar(c_d, counts_d, width=np.diff(edges_d), align="center", edgecolor="black", linewidth=0.4)     # weighted draw hist
        ad = float(row.get("actual_draw", float("nan")))                                                          # realized draw score
        if np.isfinite(ad):                                                                                       # truth line when defined
            ax_d.axvline(ad, color="C3", linestyle="-", linewidth=2.0, label=f"actual draw={ad:.3f}")             # draw truth marker

        ax_m.set_xlabel("made strength percentile")                                                               # axis label
        ax_m.set_ylabel("probability mass in bin")                                                                # axis label
        ax_m.set_title("Made (weighted)")                                                                         # panel title
        ax_m.legend(loc="upper right", fontsize=8)                                                                # compact legend
        ax_m.grid(True, alpha=0.3)                                                                                # subtle grid

        ax_d.set_xlabel("draw strength (heuristic)")                                                              # axis label
        ax_d.set_ylabel("probability mass in bin")                                                                # axis label
        ax_d.set_title("Draw (weighted)")                                                                         # panel title
        ax_d.legend(loc="upper right", fontsize=8)                                                                # compact legend
        ax_d.grid(True, alpha=0.3)                                                                                # subtle grid
    else:
        ax_m.text(0.5, 0.5, "preflop or no board", ha="center", va="center", transform=ax_m.transAxes)            # placeholder message
        ax_d.text(0.5, 0.5, "preflop or no board", ha="center", va="center", transform=ax_d.transAxes)            # placeholder message

    supt = (
        f"{session} #{hand_number} | {observer}→{target} | {street} | board={board!r} hole={hole!r}"              # one-line context title
    )
    fig.suptitle(supt, fontsize=11, y=0.98)                                                                       # place suptitle near top

    block_a = "Prior streets (completed before this row's street):\n" + (
        textwrap.fill(prior_txt, width=text_width) if prior_txt else "(none)"                                     # wrap or show empty sentinel
    )
    block_b = "This street:\n" + (textwrap.fill(this_txt, width=text_width) if this_txt else "(none)")            # same for in-street text
    caption = block_a + "\n\n" + block_b                                                                          # concatenate blocks for figure text
    fig.text(0.02, 0.02, caption, fontsize=8, family="monospace", va="bottom", ha="left")                         # render betting appendix

    plt.subplots_adjust(bottom=0.28, top=0.88, wspace=0.25)                                                       # leave room for caption and title
    return fig, (ax_m, ax_d)                                                                                      # return figure and axes tuple for further tweaking/saving
