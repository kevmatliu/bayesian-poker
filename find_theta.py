#!/usr/bin/env python3
"""Learn per-player theta vectors (preflop + postflop) given frozen global baselines."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from pipeline_common import (
    HandRef,
    collect_grouped_em_bundles_refs,
    dump_json,
    flatten_hands,
    load_json,
    pool_bundles_for_target,
)
from utils.em import run_postflop_theta_em, run_preflop_em
from utils.postflop_runner_bridge import collect_postflop_observations_known_hole_cards

LOG = logging.getLogger("find_theta")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run EM to infer theta_pre and theta_post per player using JSON global priors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Session folder(s) (each folder = one session; files = hands under pluribus/) "
            "and/or individual .phh files."
        ),
    )
    p.add_argument(
        "--global-priors",
        type=Path,
        required=True,
        help="JSON from train.py (artifacts/global_priors.json).",
    )
    p.add_argument(
        "--players",
        nargs="*",
        default=None,
        help="Players to estimate (default: union of all players seen in loaded hands).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/player_thetas.json"),
        help="Output JSON path.",
    )
    p.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Optional previous player_thetas.json to initialize EM (persists tendencies across runs).",
    )
    p.add_argument("--preflop-floor", type=float, default=0.01)
    p.add_argument("--preflop-em-iters", type=int, default=5)
    p.add_argument("--preflop-m-steps", type=int, default=100)
    p.add_argument("--preflop-m-lr", type=float, default=0.05)
    p.add_argument("--preflop-m-l2", type=float, default=0.25)
    p.add_argument("--postflop-floor", type=float, default=1e-6)
    p.add_argument("--postflop-em-iters", type=int, default=10)
    p.add_argument("--postflop-m-steps", type=int, default=200)
    p.add_argument("--postflop-m-lr", type=float, default=0.05)
    p.add_argument("--postflop-m-l2", type=float, default=0.25)
    p.add_argument("--postflop-clip", type=float, default=3.0)
    p.add_argument(
        "--session",
        action="append",
        dest="sessions",
        metavar="NAME",
        help="With a Pluribus root input, restrict to these session folder names (repeatable).",
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Max session subfolders per Pluribus root (after numeric sort).",
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def load_global_priors(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError("global priors JSON must be an object")
    pre = raw.get("preflop") or {}
    post = raw.get("postflop") or {}
    beta_preflop = np.asarray(pre["beta_preflop"], dtype=float)
    beta_facing = np.asarray(post["beta_facing"], dtype=float)
    beta_no_bet = np.asarray(post["beta_no_bet"], dtype=float)
    return beta_preflop, beta_facing, beta_no_bet


def discover_players(refs: Sequence) -> List[str]:
    names = set()
    for ref in refs:
        for p in ref.hand.player_names:
            names.add(p)
    return sorted(names)


def learn_player_thetas(
    inputs: Sequence[str | Path] | None = None,
    global_priors_path: Path | None = None,
    *,
    refs: Optional[Sequence[HandRef]] = None,
    session_names: Optional[Sequence[str]] = None,
    max_sessions: Optional[int] = None,
    players: Optional[Sequence[str]] = None,
    warm_start_path: Optional[Path] = None,
    preflop_floor: float = 0.01,
    preflop_em_iters: int = 5,
    preflop_m_steps: int = 100,
    preflop_m_lr: float = 0.05,
    preflop_m_l2: float = 0.25,
    postflop_floor: float = 1e-6,
    postflop_em_iters: int = 10,
    postflop_m_steps: int = 200,
    postflop_m_lr: float = 0.05,
    postflop_m_l2: float = 0.25,
    postflop_clip: float = 3.0,
) -> Dict[str, Any]:
    if global_priors_path is None:
        raise ValueError("global_priors_path is required")
    beta_preflop, beta_facing, beta_no_bet = load_global_priors(global_priors_path)
    if refs is None:
        if not inputs:
            raise ValueError("learn_player_thetas needs inputs or refs")
        refs = flatten_hands(
            inputs,
            session_names=session_names,
            max_sessions=max_sessions,
        )
    resolved = list(players) if players else discover_players(refs)
    if not resolved:
        raise ValueError("No players to learn (empty hands or empty --players).")

    warm: MutableMapping[str, Mapping[str, Any]] = {}
    if warm_start_path is not None:
        ws = load_json(warm_start_path)
        if isinstance(ws, dict) and isinstance(ws.get("players"), dict):
            warm = dict(ws["players"])  # type: ignore[assignment]

    grouped = collect_grouped_em_bundles_refs(refs, observers=resolved, targets=resolved)

    players_out: Dict[str, Any] = {}
    for player in resolved:
        # One independent EM run per ``player`` string; only hands where this name is in
        # ``hand.player_names`` contribute postflop; only (observer, target) pairs with
        # target == player contribute preflop bundles. Other player names are untouched.
        appears_in_hands = sum(1 for ref in refs if player in ref.hand.player_names)
        preflop_observers = sorted({obs for (obs, tgt) in grouped if tgt == player})

        bundles = pool_bundles_for_target(grouped, player)
        theta_pre_init = None
        theta_post_init = None
        if player in warm:
            entry = warm[player]
            theta_pre_init = entry.get("theta_pre")
            theta_post_init = entry.get("theta_post")

        # Preflop EM
        if bundles:
            theta_pre, _ = run_preflop_em(
                bundles,
                prior_floor=preflop_floor,
                beta_preflop=beta_preflop,
                theta_init=theta_pre_init,
                num_em_iters=preflop_em_iters,
                m_l2=preflop_m_l2,
                m_lr=preflop_m_lr,
                m_steps=preflop_m_steps,
            )
            theta_pre_list = [float(x) for x in theta_pre]
        else:
            theta_pre_list = [0.0, 0.0, 0.0]
            LOG.warning("Player %s: no preflop EM bundles; theta_pre set to zeros.", player)

        # Postflop EM — one observation list per hand (unique global index)
        observations_by_hand: List[List] = []
        for ref in refs:
            if player not in ref.hand.player_names:
                continue
            obs = collect_postflop_observations_known_hole_cards(ref.hand, player, ref.global_index)
            if obs is not None:
                observations_by_hand.append([obs])

        if observations_by_hand:
            theta_post, _ = run_postflop_theta_em(
                observations_by_hand,
                prior_floor=postflop_floor,
                theta_init=theta_post_init,
                beta_facing=beta_facing,
                beta_no_bet=beta_no_bet,
                num_em_iters=postflop_em_iters,
                m_lr=postflop_m_lr,
                m_steps=postflop_m_steps,
                m_l2=postflop_m_l2,
                clip=postflop_clip,
            )
            theta_post_list = [float(x) for x in theta_post]
        else:
            theta_post_list = [0.0, 0.0, 0.0]
            LOG.warning("Player %s: no postflop EM observations; theta_post set to zeros.", player)

        players_out[player] = {
            "theta_pre": theta_pre_list,
            "theta_post": theta_post_list,
            "preflop_theta_labels": ["fold_tilt", "call_tilt", "raise_tilt"],
            "postflop_theta_labels": ["fold_tilt", "passive_tilt", "aggression_tilt"],
            "preflop_bundles": len(bundles),
            "postflop_hands": len(observations_by_hand),
            "loaded_hands_with_this_player_name": appears_in_hands,
            "preflop_observer_names_merged_for_this_target": preflop_observers,
        }

    return {
        "schema": "bayesian_poker.player_thetas.v1",
        "global_priors_path": str(Path(global_priors_path).resolve()),
        "hands_used": len(refs),
        "players": players_out,
        "player_identity": {
            "theta_key": "Exact player name string from parsed .phh files (case-sensitive).",
            "cross_session": (
                "The same string in different session folders denotes the same θ; "
                "a roster name that only appears in session A never receives updates from session B."
            ),
            "isolation": (
                "Each entry in ``players`` is fit in a separate EM loop. Updating one name "
                "does not change tensors or JSON fields for any other name."
            ),
        },
        "notes": {
            "preflop_em": "theta_pre tilts population logits from train.py beta_preflop.",
            "postflop_em": "theta_post tilts population logits from train.py beta_facing / beta_no_bet.",
        },
    }


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    payload = learn_player_thetas(
        args.inputs,
        args.global_priors,
        session_names=args.sessions,
        max_sessions=args.max_sessions,
        players=args.players,
        warm_start_path=args.warm_start,
        preflop_floor=args.preflop_floor,
        preflop_em_iters=args.preflop_em_iters,
        preflop_m_steps=args.preflop_m_steps,
        preflop_m_lr=args.preflop_m_lr,
        preflop_m_l2=args.preflop_m_l2,
        postflop_floor=args.postflop_floor,
        postflop_em_iters=args.postflop_em_iters,
        postflop_m_steps=args.postflop_m_steps,
        postflop_m_lr=args.postflop_m_lr,
        postflop_m_l2=args.postflop_m_l2,
        postflop_clip=args.postflop_clip,
    )

    dump_json(args.out, payload)
    LOG.info("Wrote %s", Path(args.out).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
