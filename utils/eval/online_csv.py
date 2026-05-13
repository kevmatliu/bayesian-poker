"""Load ``online_range_history.csv``-style range exports and enrich with hand metadata from Pluribus."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

import pandas as pd

from utils.eval.brier import brier_postflop1326, brier_preflop_from_combo1326
from utils.eval.common import eval_log
from utils.eval.strength import (
    expected_made_and_draw_mc,
    made_percentile_calibration_stats,
    made_percentile_vector_1326,
)
from utils.eval.table import (
    betting_history_on_street,
    betting_history_up_to_street_end,
    board_at_street_end,
    player_alive_at_street_end,
    seat_columns,
)
from utils.filter import combo_key
from utils.parse import Hand
from utils.parse import parse_card, parse_cards
from utils.strength.fast_eval import all_combo_keys_fast, combo_key_to_row
from utils.strength.postflop import draw_strength_from_hand
from utils.strength.preflop import get_equivalence_class

META_COLUMNS: Tuple[str, ...] = ("session", "hand_number", "street", "observer", "target")  # required id columns for joins

# Prefix for per-combo made-strength percentile columns in :func:`add_calibration_columns`.
# (Distinct from raw combo-key probability column names, which are exactly ``combo_key`` strings.)
MADE_STRENGTH_PCT_COLUMN_PREFIX = "made_strength_pct__"  # namespace wide percentile columns away from probs

# Prior street for postflop labels (require target was still in before this street).
_STREET_PRIOR_FOR_LABEL: dict[str, str] = {"flop": "pre-flop", "turn": "flop", "river": "turn"}


def meta_columns_present(df: pd.DataFrame) -> None:
    missing = [c for c in META_COLUMNS if c not in df.columns]            # detect absent required fields
    if missing:                                                           # fail before partial enrichment corrupts assumptions
        raise ValueError(f"online range CSV missing columns: {missing}")  # actionable error for callers


def combo_probability_columns(df: pd.DataFrame) -> list[str]:
    """Canonical 1,326 combo **probability** columns present in ``df`` (``all_combo_keys_fast()`` order)."""
    keys = list(all_combo_keys_fast())           # fixed global ordering
    return [c for c in keys if c in df.columns]  # subset to columns actually exported in this csv


def combo_made_percentile_column_names(
    combo_order: Optional[Iterable[str]] = None,
) -> List[str]:
    """Column names for per-combo made percentiles (see :func:`add_calibration_columns`)."""
    keys = list(combo_order) if combo_order is not None else list(all_combo_keys_fast())  # explicit or default order
    pfx = MADE_STRENGTH_PCT_COLUMN_PREFIX                                                 # local alias for readability
    return [f"{pfx}{k}" for k in keys]                                                    # stable wide-column naming scheme


def load_hand_pluribus(
    pluribus_root: Path,
    session: str,
    hand_number,
) -> Optional[Hand]:
    """Resolve ``pluribus_root / session / {hand_number}.phh`` and parse."""
    stem = str(int(hand_number)) if str(hand_number).isdigit() else str(hand_number)  # normalize filenames like 0007.phh
    path = pluribus_root / str(session) / f"{stem}.phh"                               # concrete phh path
    if not path.is_file():                                                            # missing hand file
        return None                                                                   # signal join miss without raising
    return Hand.from_file(path)                                                       # parse full hand object


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

    Also adds ``betting_prior_streets`` (completed action before the row's street) and
    ``betting_this_street`` (action on the row's street, from the final in-street snapshot),
    both human-readable strings for plotting / notebooks.

    Adds ``target_still_in_hand`` — whether ``target`` may be scored on this row:

    * ``pre-flop``: alive at the end of pre-flop (survived the preflop betting round).
    * ``flop`` / ``turn`` / ``river``: alive at the **end of the prior street** (entered this
      street still contesting) **and** alive at the **end of this street**. This drops carry-forward
      CSV rows where the board advanced but ``target`` had already folded.

    ``False`` when the ``.phh`` is missing or the relevant street was not reached.

    With ``verbose=True``, prints progress every ``progress_every`` rows (minimum 1).
    """
    meta_columns_present(df)                                                                     # validate schema early
    n = len(df)                                                                                  # row count for progress logs
    eval_log(
        verbose,                                                                                 # no-op unless verbose
        f"enrich_online_range_dataframe: {n} rows | pluribus_root={pluribus_root}",              # startup banner
    )
    step = max(1, int(progress_every))                                                           # avoid modulo-by-zero and spammy logs
    out = df.copy()                                                                              # never mutate caller's frame in place
    cache: dict[tuple[str, int | str], Optional[Hand]] = {}                                      # memoize parsed hands by (session, hand)

    def _hn_key(hn) -> int | str:
        if isinstance(hn, int):               # already hashable as int cache key
            return hn                         # keep ints stable
        s = str(hn)                           # stringify for digit test
        return int(s) if s.isdigit() else hn  # normalize numeric strings to int keys

    def get_hand(sess: str, hn) -> Optional[Hand]:
        key = (str(sess), _hn_key(hn))                                # canonical cache key
        if key not in cache:                                          # lazy parse on first touch
            cache[key] = load_hand_pluribus(pluribus_root, sess, hn)  # store None on miss
        return cache[key]                                             # return cached hand or None

    comm: list[str] = []                                                                         # parallel column buffer: board text
    holes: list[str] = []                                                                        # parallel column buffer: target holes
    seats: list[list[str]] = []                                                                  # parallel column buffer: six seat holes
    bet_prior: list[str] = []                                                                    # parallel column buffer: prior betting text
    bet_this: list[str] = []                                                                     # parallel column buffer: in-street betting text
    still_in: list[bool] = []                                                                    # target eligible for row metrics (for eval filtering)

    def _target_still_in_for_row(h: Hand, street_label: str, target_name: str) -> bool:
        if street_label == "pre-flop":
            return player_alive_at_street_end(h, "pre-flop", target_name)
        prior = _STREET_PRIOR_FOR_LABEL.get(street_label)
        if prior is None:
            return player_alive_at_street_end(h, street_label, target_name)
        return player_alive_at_street_end(h, prior, target_name) and player_alive_at_street_end(
            h, street_label, target_name
        )

    for i, (_, row) in enumerate(out.iterrows(), start=1):                                       # iterate rows with 1-based human index
        if verbose and (i == 1 or i == n or i % step == 0):                                      # periodic progress
            eval_log(verbose, f"enrich … row {i}/{n}")                                           # heartbeat
        hand = get_hand(row["session"], row["hand_number"])                                      # fetch parsed hand if available
        tgt = str(row["target"])                                                                 # seat whose hole cards we surface
        st = str(row["street"])                                                                  # street label drives board + action windows
        if hand is None:                                                                         # cannot join phh
            comm.append("")                                                                      # empty placeholders keep list lengths aligned
            holes.append("")
            seats.append([""] * 6)
            bet_prior.append("")
            bet_this.append("")
            still_in.append(False)                                                               # unknown hand → exclude from in-hand eval
            continue                                                                             # skip expensive parsing for missing files
        if st == "pre-flop":                                                                     # no public board yet
            comm.append("")                                                                      # convention: empty board string preflop
        else:
            comm.append(board_at_street_end(hand, st))                                           # cumulative board through street end
        holes.append(str(hand.hole_cards.get(tgt, "") or ""))                                    # target hole cards as compact string
        sc = seat_columns(hand)                                                                  # seat→hole mapping for table display
        seats.append([sc.get(f"p{i}", "") for i in range(1, 7)])                                 # fixed p1..p6 order
        bet_prior.append(betting_history_up_to_street_end(hand, st))                             # completed prior action narrative
        bet_this.append(betting_history_on_street(hand, st))                                     # in-street action narrative
        still_in.append(_target_still_in_for_row(hand, st, tgt))                                 # fold filtering for range metrics

    out["community_cards"] = comm                                                                # attach new column
    out["target_hole_cards"] = holes                                                             # attach new column
    for i in range(1, 7):                                                                        # explode seat lists into scalar columns
        out[f"p{i}"] = [s[i - 1] for s in seats]                                                 # i-th seat across rows
    out["betting_prior_streets"] = bet_prior                                                     # attach prior betting text
    out["betting_this_street"] = bet_this                                                        # attach in-street betting text
    out["target_still_in_hand"] = still_in                                                       # attach in-hand flag for downstream metrics
    eval_log(
        verbose,
        f"enrich_online_range_dataframe: done ({len(cache)} unique session/hand hands cached)",  # cache stats
    )
    return out                                                                                   # enriched copy


def _true_preflop_class(hole4: str) -> str:
    if len(hole4) != 4:                                                             # invalid hole token
        return ""                                                                   # sentinel for downstream brier skips
    return get_equivalence_class([parse_card(hole4[0:2]), parse_card(hole4[2:4])])  # 169 class for preflop brier


def _true_combo_key(hole4: str) -> str:
    if len(hole4) != 4:                                               # cannot form two cards
        return ""                                                     # skip postflop brier truth
    return combo_key(parse_card(hole4[0:2]), parse_card(hole4[2:4]))  # canonical 1326 key for postflop brier


def _board_cards_list(board_str: str) -> list:
    s = (board_str or "").strip()                                    # normalize whitespace
    if len(s) < 6:                                                   # fewer than three cards
        return []                                                    # too short for flop logic
    return parse_cards([s[i : i + 2] for i in range(0, len(s), 2)])  # split contiguous board into cards


def _hole_cards_list(hole4: str) -> list:
    h = (hole4 or "").strip()             # normalize hole string
    if len(h) != 4:                       # not two two-char cards
        return []                         # cannot parse holes
    return parse_cards([h[0:2], h[2:4]])  # two-card list for strength evaluators


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
    meta_columns_present(df)                                                                      # require meta columns for row semantics
    n = len(df)                                                                                   # number of rows to calibrate
    eval_log(
        verbose,
        f"add_calibration_columns: {n} rows | strength_mc_samples={strength_mc_samples}",         # config echo
    )
    step = max(1, int(progress_every))                                                            # throttle verbose logs
    combo_list = list(combo_cols)                                                                 # materialize iterable once for set/size checks
    order = list(all_combo_keys_fast())                                                           # canonical fast-eval combo order
    if len(combo_list) != len(order) or set(combo_list) != set(order):                            # must cover exact 1326 support
        raise ValueError(
            "Combo columns must be the same 1,326 ``combo_key`` strings as ``all_combo_keys_fast()`` "
            "(order may differ in the CSV; calibration reindexes to the fast-eval row order)."
        )                                                                                         # prevent silent misalignment between csv columns and fast tables
    # Use canonical key order for the Brier / probability vector
    out = df.copy()                                                                               # avoid mutating input during wide concat
    brs = []                                                                                      # per-row brier scores
    emade = []                                                                                    # expected made percentile (mc)
    edraw = []                                                                                    # expected draw (mc)
    amade = []                                                                                    # actual made percentile at true combo when defined
    adraw = []                                                                                    # actual draw heuristic at true hole when defined
    made_mu = []                                                                                  # conditional mean of made percentile under range (live subset)
    made_sig = []                                                                                 # conditional std dev of made percentile under range
    made_z = []                                                                                   # z-score of truth vs conditional law
    made_abs_z = []                                                                               # absolute z for magnitude summaries
    combo_nll = []                                                                                # negative log prob of true combo under full simplex
    made_midrank = []                                                                             # tie-aware midrank of truth percentile under range
    made_cdf_le = []                                                                              # cdf at truth including ties
    made_per_combo = np.full((n, 1326), np.nan, dtype=np.float64)                                 # cache full percentile rows for export
    perc_by_board: dict[str, np.ndarray] = {}                                                     # memoize percentile vectors keyed by board string
    for i, (_, row) in enumerate(out.iterrows()):                                                 # row-major calibration pass
        ri = i + 1                                                                                # 1-based row index for logging
        if verbose and (ri == 1 or ri == n or ri % step == 0):                                    # periodic progress
            eval_log(verbose, f"calibration … row {ri}/{n}")                                      # heartbeat
        dist = {c: float(row[c]) for c in order}                                                  # dict view of combo distribution in canonical order
        hole = str(row.get("target_hole_cards", "") or "")                                        # realized hole cards if enriched
        street = str(row["street"])                                                               # street determines pre vs post scoring
        board = str(row.get("community_cards", "") or "")                                         # board text if enriched

        if len(hole) == 4:                                                                        # only score brier when holes known
            if street == "pre-flop":                                                              # use 169 abstraction target
                brs.append(
                    brier_preflop_from_combo1326(dist, _true_preflop_class(hole), verbose=False)  # multiclass brier vs 169 truth
                )
            else:                                                                                 # postflop uses exact combo truth
                tc = _true_combo_key(hole)                                                        # 1326 key for realized holding
                brs.append(
                    brier_postflop1326(dist, order, tc, verbose=False) if tc else float("nan")    # brier vs combo truth
                )
        else:
            brs.append(float("nan"))                                                              # cannot define brier without holes

        if len(board) >= 6:                                                                       # at least flop present for postflop strength stats
            e1, e2 = expected_made_and_draw_mc(
                dist,                                                                             # range distribution as dict
                board,                                                                            # board string for evaluators
                n_samples=strength_mc_samples,                                                    # mc accuracy knob
                rng=strength_rng,                                                                 # optional deterministic rng
                verbose=False,                                                                    # keep inner logs quiet during bulk eval
            )
            hole_cards = _hole_cards_list(hole)                                                   # parsed target holes
            b_cards = _board_cards_list(board)                                                    # parsed board list
            if len(hole_cards) == 2 and len(b_cards) >= 3:                                        # valid inputs for draw heuristic
                a2 = float(draw_strength_from_hand(hole_cards, b_cards))                          # realized draw score
            else:
                a2 = float("nan")                                                                 # missing pieces => undefined draw

            if board not in perc_by_board:                                                        # compute percentile field once per unique board
                perc_by_board[board] = made_percentile_vector_1326(board)                         # expensive static table fill
            perc_v = perc_by_board[board]                                                         # reuse cached vector
            made_per_combo[i, :] = perc_v                                                         # stash full row for wide export
            p_v = np.array([float(row[c]) for c in order], dtype=np.float64)                      # dense prob vector aligned to perc_v
            tc = _true_combo_key(hole) if len(hole) == 4 else ""                                  # truth combo key when known
            try:
                j = combo_key_to_row(tc) if tc else -1                                            # map combo key to static row index
            except KeyError:
                j = -1                                                                            # unknown key treated as missing
            if j >= 0 and np.isfinite(perc_v[j]):                                                 # truth row live vs board
                a1 = float(perc_v[j])                                                             # realized made percentile
                mu, sig, z, az, nll, mid, cle = made_percentile_calibration_stats(
                    p_v, perc_v, j                                                                # summarize how truth sits under predicted range law
                )
                made_mu.append(mu)                                                                # record distribution mean
                made_sig.append(sig)                                                              # record spread
                made_z.append(z)                                                                  # record standardized surprise
                made_abs_z.append(az)                                                             # record magnitude
                combo_nll.append(nll if np.isfinite(nll) else float("nan"))                       # cap inf for parquet/csv safety
                made_midrank.append(mid)                                                          # record tie-aware rank statistic
                made_cdf_le.append(cle)                                                           # record cdf at truth
            else:
                a1 = float("nan")                                                                 # undefined actual made when dead/unknown
                nan_m = float("nan")                                                              # shared nan constant for batch append readability
                made_mu.append(nan_m)                                                             # pad summary columns with nan
                made_sig.append(nan_m)
                made_z.append(nan_m)
                made_abs_z.append(nan_m)
                combo_nll.append(nan_m)
                made_midrank.append(nan_m)
                made_cdf_le.append(nan_m)
        else:
            e1 = e2 = a1 = a2 = float("nan")                                                      # short board => no postflop strength summaries
            made_mu.append(float("nan"))                                                          # pad all postflop-only diagnostics
            made_sig.append(float("nan"))
            made_z.append(float("nan"))
            made_abs_z.append(float("nan"))
            combo_nll.append(float("nan"))
            made_midrank.append(float("nan"))
            made_cdf_le.append(float("nan"))

        emade.append(e1)                                                                          # push mc made expectation for this row
        edraw.append(e2)                                                                          # push mc draw expectation
        amade.append(a1)                                                                          # push actual made (or nan)
        adraw.append(a2)                                                                          # push actual draw (or nan)

    out["brier"] = brs                                                                            # attach brier column
    out["expected_made_pct"] = emade                                                              # attach expected made
    out["expected_draw"] = edraw                                                                  # attach expected draw
    out["actual_made_pct"] = amade                                                                # attach actual made
    out["actual_draw"] = adraw                                                                    # attach actual draw
    out["made_dist_mu"] = made_mu                                                                 # attach conditional mean diagnostic
    out["made_dist_sigma"] = made_sig                                                             # attach conditional std diagnostic
    out["made_pct_z"] = made_z                                                                    # attach z diagnostic
    out["made_pct_abs_z"] = made_abs_z                                                            # attach abs z diagnostic
    out["combo_nll"] = combo_nll                                                                  # attach combo nll diagnostic
    out["made_pct_midrank"] = made_midrank                                                        # attach midrank diagnostic
    out["made_pct_cdf_le"] = made_cdf_le                                                          # attach cdf diagnostic
    perc_cols = combo_made_percentile_column_names(order)                                         # wide column names for percentile matrix
    out = pd.concat(
        [out, pd.DataFrame(made_per_combo, columns=perc_cols, index=out.index)],                  # horizontally stack 1326 percentile cols
        axis=1,                                                                                   # concat along columns not rows
    )                                                                                             # build final wide frame
    eval_log(verbose, "add_calibration_columns: done")                                            # completion banner
    return out                                                                                    # calibrated copy
