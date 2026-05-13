"""Batch filtering over session folders (``python -m runners.filter_sessions``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import load_json, preflop_postflop_priors_for_target, read_session_names_file
from utils.filter import ComboRangeFilter
from utils.parse import Session
from utils.postflop_runner_bridge import POSTFLOP_STREETS

from .execute import LOG, _run_preflop_filter_for_hand, all_combo_keys


def _board_at_street_end(hand, street: str) -> str:
    sts = hand.states.get(street, [])
    if not sts:
        return ""
    return sts[-1].community_cards or ""


def _range_history_rows_for_hand(
    hand,
    session_name: str,
    hand_number: Any,
    observer: str,
    target: str,
    preflop_range: Dict[str, float],
    street_snaps: List[Tuple[str, Dict[str, float]]],
    combo_cols: Sequence[str],
) -> List[Dict[str, Any]]:
    """Same row layout as ``online_range_history.csv`` (combo-range history schema)."""
    observer_hole = hand.hole_cards.get(observer, "") or ""
    rows: List[Dict[str, Any]] = []

    def base_row(dist: Dict[str, float], street_label: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "session": session_name,
            "hand_number": hand_number,
            "street": street_label,
            "observer": observer,
            "target": target,
        }
        for c in combo_cols:
            out[c] = float(dist.get(c, 0.0))
        return out

    try:
        pre_dist = ComboRangeFilter.explode_preflop_to_combos(preflop_range, observer_hole, "")
    except ValueError:
        return []
    rows.append(base_row(pre_dist, "pre-flop"))

    snap_map = {s: dict(d) for s, d in street_snaps}

    for street in POSTFLOP_STREETS:
        board = _board_at_street_end(hand, street)
        if len(board) < 6:
            continue
        if street in snap_map:
            dist = snap_map[street]
        else:
            try:
                dist = ComboRangeFilter.explode_preflop_to_combos(preflop_range, observer_hole, board)
            except ValueError:
                continue
        rows.append(base_row(dist, street))

    return rows


def _configure_stdout_and_file_logging(level: int, process_log: Path) -> None:
    """Emit logs to stdout and to ``process_log`` (mirrors visible process in a ``.log`` file)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    root.addHandler(sh)
    process_log.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(process_log, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    root.addHandler(fh)


def filter_sessions_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "For each session in a list file, run preflop range + postflop combo filtering using "
            "population beta from global_priors.json and per-target theta from player_thetas.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pluribus_root",
        type=Path,
        help="Pluribus root directory containing session subfolders (e.g. pluribus/).",
    )
    parser.add_argument(
        "--sessions-file",
        type=Path,
        required=True,
        help="Text file: one session folder name per line (e.g. 30).",
    )
    parser.add_argument(
        "--global-priors",
        type=Path,
        default=Path("artifacts/global_priors.json"),
        help="JSON from train.py (population beta_preflop / beta_facing / beta_no_bet).",
    )
    parser.add_argument(
        "--player-thetas",
        type=Path,
        default=Path("artifacts/player_thetas.json"),
        help="JSON from find_theta.py (per-player theta_pre / theta_post).",
    )
    parser.add_argument(
        "--players",
        nargs="+",
        metavar="NAME",
        default=("MrBlue", "Bill", "Pluribus"),
        help=(
            "Roster of distinct player names (2–6). For two players, a hand is used only when both "
            "are seated. For more than two, each unordered roster pair (n choose 2) is run in both "
            "directions whenever both players in that pair are seated."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.set_defaults(filter_verbose=True)
    parser.add_argument(
        "--filter-verbose",
        action="store_true",
        dest="filter_verbose",
        help="Per-update INFO logs during filtering (default: on).",
    )
    parser.add_argument(
        "--no-filter-verbose",
        action="store_false",
        dest="filter_verbose",
        help="Disable per-update filter INFO logs.",
    )
    parser.add_argument(
        "--range-csv-out",
        type=Path,
        default=Path("artifacts/filter_sessions_range_history.csv"),
        help=(
            "Write combo-range history CSV (metadata + 1326 combo columns per row); "
            "same column layout as ``online_range_history.csv``."
        ),
    )
    parser.add_argument(
        "--process-log-out",
        type=Path,
        default=Path("artifacts/filter_sessions_process.log"),
        help="Process log: same INFO/DEBUG lines as stdout, appended with the full range CSV at end.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write one JSON document with all hand-level filter results.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    args = parser.parse_args(argv)
    try:
        import pandas as pd
    except ImportError:
        print("filter-sessions requires pandas (pip install pandas).", file=sys.stderr)
        return 1

    log_level = getattr(logging, args.log_level)
    process_log_path = args.process_log_out.expanduser().resolve()
    _configure_stdout_and_file_logging(log_level, process_log_path)
    LOG.info("Process log (and stdout): %s", process_log_path)

    pluribus_root = args.pluribus_root.expanduser().resolve()
    if not pluribus_root.is_dir():
        LOG.error("Not a directory: %s", pluribus_root)
        return 2

    gp_path = args.global_priors.expanduser().resolve()
    if not gp_path.is_file():
        LOG.error("Global priors JSON not found: %s", gp_path)
        return 2

    pt_path = args.player_thetas.expanduser().resolve()
    if not pt_path.is_file():
        LOG.error("Player thetas JSON not found: %s", pt_path)
        return 2

    raw_doc = load_json(pt_path)
    if not isinstance(raw_doc, dict):
        LOG.error("Player thetas JSON must be an object: %s", pt_path)
        return 2
    players_block = raw_doc.get("players")
    if not isinstance(players_block, dict) or not players_block:
        LOG.error("Player thetas JSON missing non-empty 'players' object: %s", pt_path)
        return 2

    session_names = read_session_names_file(args.sessions_file)
    if not session_names:
        LOG.error("No sessions in %s", args.sessions_file)
        return 2

    for name in session_names:
        if not (pluribus_root / name).is_dir():
            LOG.error("Session folder missing: %s", pluribus_root / name)
            return 2

    roster = list(dict.fromkeys(args.players))
    if len(roster) != len(args.players):
        LOG.error("--players must list distinct names (got duplicates).")
        return 2
    if len(roster) < 2:
        LOG.error("--players requires at least two distinct names.")
        return 2
    if len(roster) > 6:
        LOG.error("--players accepts at most six names (got %d).", len(roster))
        return 2

    priors_by_target: Dict[str, Tuple[Any, Any]] = {}
    for pname in roster:
        try:
            priors_by_target[pname] = preflop_postflop_priors_for_target(
                pname,
                gp_path,
                players_block,
            )
        except ValueError as exc:
            LOG.error("%s", exc)
            return 2

    if len(roster) > 2:
        perspectives = tuple(
            pair for a, b in combinations(roster, 2) for pair in ((a, b), (b, a))
        )
    else:
        perspectives = tuple((obs, tgt) for obs in roster for tgt in roster if obs != tgt)
    roster_set = frozenset(roster)
    json_sessions: List[Dict[str, Any]] = []
    total_results = 0
    combo_cols = all_combo_keys()
    csv_rows: List[Dict[str, Any]] = []

    for session_name in session_names:
        session_path = pluribus_root / session_name
        LOG.info("Filter sessions | loading %s", session_path)
        session = Session(str(session_path))
        session.parse()
        session_hand_results: List[Dict[str, Any]] = []

        phh_files = sorted(
            (p for p in session_path.iterdir() if p.is_file() and p.suffix.lower() == ".phh"),
            key=lambda p: int(p.stem),
        )

        for hi, hand in enumerate(session.hands):
            stem = phh_files[hi].stem if hi < len(phh_files) else str(hi)
            hand_number = int(stem) if str(stem).isdigit() else stem
            if len(roster) <= 2:
                if not roster_set.issubset(hand.player_names):
                    continue

            for observer, target in perspectives:
                if len(roster) > 2:
                    if observer not in hand.player_names or target not in hand.player_names:
                        continue

                learned_pre, learned_post = priors_by_target[target]
                tag = f"sess={session_name} hand={hand_number} idx={hi} {observer}→{target}"
                LOG.info("Filter sessions | start perspective | %s", tag)
                street_snaps: List[Tuple[str, Dict[str, float]]] = []
                res = _run_preflop_filter_for_hand(
                    hand=hand,
                    observer=observer,
                    target=target,
                    hand_index=hi,
                    phi=0.0,
                    top_k=args.top_k,
                    learned_preflop_model=learned_pre,
                    learned_postflop_model=learned_post,
                    street_end_snapshots=street_snaps,
                    filter_verbose=args.filter_verbose,
                    filter_tag=tag,
                )
                if res is None:
                    LOG.info("Filter sessions | skip (no result) | %s", tag)
                    continue
                row = asdict(res)
                row["session"] = session_name
                row["hand_number"] = hand_number
                session_hand_results.append(row)
                total_results += 1
                history_rows = _range_history_rows_for_hand(
                    hand,
                    session_name,
                    hand_number,
                    observer,
                    target,
                    res.final_range,
                    street_snaps,
                    combo_cols,
                )
                csv_rows.extend(history_rows)
                if history_rows:
                    rh_df = pd.DataFrame(history_rows)
                    LOG.info(
                        "Filter sessions | range history dataframe | %s | shape=%s | n_cols=%d",
                        tag,
                        rh_df.shape,
                        len(rh_df.columns),
                    )
                    LOG.info("Filter sessions | range history CSV chunk:\n%s", rh_df.to_csv(index=False))

        json_sessions.append({"session": session_name, "hand_results": session_hand_results})
        LOG.info(
            "Filter sessions | done %s | hand-level results=%d",
            session_name,
            len(session_hand_results),
        )

    LOG.info("Filter sessions | complete | sessions=%d | total hand results=%d", len(session_names), total_results)

    if not csv_rows:
        LOG.warning("No range-history CSV rows produced (check target preflop actions / seated pairs).")
    else:
        df = pd.DataFrame(csv_rows)
        meta_cols = ["session", "hand_number", "street", "observer", "target"]
        combo_only = [c for c in combo_cols if c in df.columns]
        df = df[meta_cols + combo_only]
        out_csv = args.range_csv_out.expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        LOG.info(
            "Wrote range history CSV (%d rows × %d cols) → %s",
            len(df),
            len(df.columns),
            out_csv,
        )
        LOG.info(
            "Range history dataframe (metadata columns preview):\n%s",
            df[meta_cols].head(5).to_string(index=False),
        )
        with open(process_log_path, "a", encoding="utf-8") as lf:
            lf.write("\n# --- full online_range_history-style dataframe (same as --range-csv-out) ---\n")
            df.to_csv(lf, index=False)
        LOG.info("Appended full range-history CSV (%d cols) to process log: %s", len(df.columns), process_log_path)

    if args.json_out is not None:
        payload = {
            "schema": "bayesian_poker.filter_sessions.v1",
            "pluribus_root": str(pluribus_root),
            "global_priors": str(gp_path),
            "player_thetas": str(pt_path),
            "players": list(roster),
            "sessions": json_sessions,
        }
        out = args.json_out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        LOG.info("Wrote %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(filter_sessions_main())
