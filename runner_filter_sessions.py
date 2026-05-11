"""Batch filtering over session folders (``runner.py filter-sessions``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_common import load_json, read_session_names_file
from priors_artifacts import preflop_postflop_priors_for_target
from runner_execute import LOG, _run_preflop_filter_for_hand
from utils.parse import Session


def filter_sessions_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "For each session in a list file, run preflop range + postflop combo filtering using "
            "population beta from global_priors.json and per-target theta from player_thetas.json "
            "(same prior construction as the online phase of ``runner.py session-split``)."
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
        nargs=2,
        metavar=("P1", "P2"),
        default=("Gogo", "Pluribus"),
        help="Exactly two player names; runs both observer→target directions when both are seated.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--filter-verbose",
        action="store_true",
        help="Per-update INFO logs (can be very large).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write one JSON document with all hand-level filter results.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

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

    p_a, p_b = args.players[0], args.players[1]
    if p_a == p_b:
        LOG.error("--players must name two distinct players.")
        return 2

    priors_by_target: Dict[str, Tuple[Any, Any]] = {}
    for pname in (p_a, p_b):
        try:
            priors_by_target[pname] = preflop_postflop_priors_for_target(
                pname,
                gp_path,
                players_block,
            )
        except ValueError as exc:
            LOG.error("%s", exc)
            return 2

    perspectives: Tuple[Tuple[str, str], ...] = ((p_a, p_b), (p_b, p_a))
    json_sessions: List[Dict[str, Any]] = []
    total_results = 0

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
            if p_a not in hand.player_names or p_b not in hand.player_names:
                continue

            for observer, target in perspectives:
                learned_pre, learned_post = priors_by_target[target]
                tag = f"sess={session_name} hand={hand_number} idx={hi} {observer}→{target}"
                res = _run_preflop_filter_for_hand(
                    hand=hand,
                    observer=observer,
                    target=target,
                    hand_index=hi,
                    phi=0.0,
                    top_k=args.top_k,
                    learned_prior=learned_pre,
                    learned_postflop_prior=learned_post,
                    filter_verbose=args.filter_verbose,
                    filter_tag=tag,
                )
                if res is None:
                    LOG.debug("Skip | %s", tag)
                    continue
                row = asdict(res)
                row["session"] = session_name
                row["hand_number"] = hand_number
                session_hand_results.append(row)
                total_results += 1

        json_sessions.append({"session": session_name, "hand_results": session_hand_results})
        LOG.info(
            "Filter sessions | done %s | hand-level results=%d",
            session_name,
            len(session_hand_results),
        )

    LOG.info("Filter sessions | complete | sessions=%d | total hand results=%d", len(session_names), total_results)

    if args.json_out is not None:
        payload = {
            "schema": "bayesian_poker.filter_sessions.v1",
            "pluribus_root": str(pluribus_root),
            "global_priors": str(gp_path),
            "player_thetas": str(pt_path),
            "players": [p_a, p_b],
            "sessions": json_sessions,
        }
        out = args.json_out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        LOG.info("Wrote %s", out)

    return 0
