#!/usr/bin/env python3
"""Evaluate global priors + per-player theta on held-out hands (log-likelihood)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

from runners.common import (
    HandRef,
    dump_json,
    flatten_hands,
    hole_cards_to_hand_class,
    load_json,
    preflop_decisions_for_hand,
)
from utils.postflop_runner_bridge import collect_postflop_observations_known_hole_cards
from utils.action.postflop import PostflopActionModel
from utils.action.preflop import PreflopActionModel, canonical_preflop_action
from utils.prior.postflop import PostflopPrior
from utils.prior.preflop import PreflopPrior

LOG = logging.getLogger("eval")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score actions under frozen global priors and optional player thetas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Session directories (one folder per session, one .phh per hand) or single .phh files.",
    )
    p.add_argument(
        "--global-priors",
        type=Path,
        required=True,
        help="JSON from ``python -m runners.train`` (global priors).",
    )
    p.add_argument(
        "--player-thetas",
        type=Path,
        default=None,
        help="Optional JSON from ``python -m runners.find_theta`` (uses zeros when omitted).",
    )
    p.add_argument(
        "--players",
        nargs="*",
        default=None,
        help="Restrict scoring to these players (default: all with theta entries or seen in data).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON.",
    )
    p.add_argument("--preflop-floor", type=float, default=0.01)
    p.add_argument("--postflop-floor", type=float, default=1e-6)
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


def load_models(
    global_path: Path,
    player_path: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Tuple[List[float], List[float]]]]:
    raw = load_json(global_path)
    assert isinstance(raw, dict)
    beta_preflop = np.asarray(raw["preflop"]["beta_preflop"], dtype=float)
    beta_facing = np.asarray(raw["postflop"]["beta_facing"], dtype=float)
    beta_no_bet = np.asarray(raw["postflop"]["beta_no_bet"], dtype=float)

    thetas: Dict[str, Tuple[List[float], List[float]]] = {}
    if player_path is not None:
        pt = load_json(player_path)
        if isinstance(pt, dict) and isinstance(pt.get("players"), dict):
            for name, row in pt["players"].items():
                if not isinstance(row, dict):
                    continue
                tp = row.get("theta_pre") or [0.0, 0.0, 0.0]
                ts = row.get("theta_post") or [0.0, 0.0, 0.0]
                thetas[str(name)] = (
                    [float(x) for x in tp],
                    [float(x) for x in ts],
                )

    return beta_preflop, beta_facing, beta_no_bet, thetas


def evaluate_split(
    inputs: Sequence[str | Path] | None = None,
    global_priors_path: Path | None = None,
    *,
    refs: Optional[Sequence[HandRef]] = None,
    session_names: Optional[Sequence[str]] = None,
    max_sessions: Optional[int] = None,
    player_thetas_path: Optional[Path] = None,
    players_filter: Optional[Sequence[str]] = None,
    preflop_floor: float = 0.01,
    postflop_floor: float = 1e-6,
) -> Dict[str, Any]:
    if global_priors_path is None:
        raise ValueError("global_priors_path is required")
    beta_preflop, beta_facing, beta_no_bet, thetas_map = load_models(global_priors_path, player_thetas_path)
    if refs is None:
        if not inputs:
            raise ValueError("evaluate_split needs inputs or refs")
        refs = flatten_hands(
            inputs,
            session_names=session_names,
            max_sessions=max_sessions,
        )

    allowed = set(players_filter) if players_filter else None

    agg: MutableMapping[str, MutableMapping[str, float]] = {}

    def bump(player: str, phase: str, logp: float) -> None:
        if allowed is not None and player not in allowed:
            return
        slot = agg.setdefault(player, {"preflop_loglik": 0.0, "postflop_loglik": 0.0, "preflop_n": 0, "postflop_n": 0})
        slot[phase + "_loglik"] += logp
        slot[phase + "_n"] += 1

    for ref in refs:
        hand = ref.hand
        for player in hand.player_names:
            if allowed is not None and player not in allowed:
                continue
            tp, ts = thetas_map.get(player, ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
            pre_prior = PreflopActionModel(
                PreflopPrior(floor=preflop_floor, beta_preflop=beta_preflop),
                tuple(float(x) for x in tp),
            )

            post_prior = PostflopActionModel(
                PostflopPrior(
                    floor=postflop_floor,
                    beta_facing=beta_facing,
                    beta_no_bet=beta_no_bet,
                ),
                tuple(float(x) for x in ts),
            )

            hc = hole_cards_to_hand_class(hand.hole_cards.get(player, "") or "")
            if hc is not None:
                for dec in preflop_decisions_for_hand(hand, player, ref.global_index):
                    probs = pre_prior.action_probs(hc, dec.state_key)
                    a = canonical_preflop_action(dec.action_bucket)
                    p = max(float(probs.get(a, 0.0)), 1e-300)
                    bump(player, "preflop", math.log(p))

            obs = collect_postflop_observations_known_hole_cards(hand, player, ref.global_index)
            if obs is None:
                continue
            for feat, action in obs.decisions:
                probs = post_prior.action_probs(feat)
                p = max(float(probs.get(action, 0.0)), 1e-300)
                bump(player, "postflop", math.log(p))

    summary_players: Dict[str, Any] = {}
    for player, m in sorted(agg.items()):
        pn = int(m["preflop_n"])
        pfn = int(m["postflop_n"])
        summary_players[player] = {
            "preflop_avg_log_prob": (m["preflop_loglik"] / pn) if pn else None,
            "preflop_decisions": pn,
            "postflop_avg_log_prob": (m["postflop_loglik"] / pfn) if pfn else None,
            "postflop_decisions": pfn,
        }

    total_pre_n = sum(int(v["preflop_n"]) for v in agg.values())
    total_post_n = sum(int(v["postflop_n"]) for v in agg.values())
    total_pre_ll = sum(float(v["preflop_loglik"]) for v in agg.values())
    total_post_ll = sum(float(v["postflop_loglik"]) for v in agg.values())

    return {
        "schema": "bayesian_poker.eval_metrics.v1",
        "hands": len(refs),
        "global_priors_path": str(Path(global_priors_path).resolve()),
        "player_thetas_path": str(Path(player_thetas_path).resolve()) if player_thetas_path else None,
        "aggregate": {
            "preflop_avg_log_prob": (total_pre_ll / total_pre_n) if total_pre_n else None,
            "preflop_decisions": total_pre_n,
            "postflop_avg_log_prob": (total_post_ll / total_post_n) if total_post_n else None,
            "postflop_decisions": total_post_n,
        },
        "by_player": summary_players,
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

    metrics = evaluate_split(
        args.inputs,
        args.global_priors,
        session_names=args.sessions,
        max_sessions=args.max_sessions,
        player_thetas_path=args.player_thetas,
        players_filter=args.players,
        preflop_floor=args.preflop_floor,
        postflop_floor=args.postflop_floor,
    )

    LOG.info(
        "Evaluated %d hands | preflop n=%s avg_log_p=%s | postflop n=%s avg_log_p=%s",
        metrics["hands"],
        metrics["aggregate"]["preflop_decisions"],
        metrics["aggregate"]["preflop_avg_log_prob"],
        metrics["aggregate"]["postflop_decisions"],
        metrics["aggregate"]["postflop_avg_log_prob"],
    )

    if args.out:
        dump_json(args.out, metrics)
        LOG.info("Wrote %s", Path(args.out).resolve())

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
