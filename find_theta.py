#!/usr/bin/env python3
"""Learn per-player theta vectors (preflop + postflop) given frozen global baselines."""

from __future__ import annotations

import argparse
import logging
import sys
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from pipeline_common import (
    HandRef,
    dump_json,
    flatten_hands,
    gather_preflop_bundles_for_target_player,
    load_json,
)
from utils.em import run_postflop_theta_em, run_preflop_em
from utils.postflop_runner_bridge import collect_postflop_observations_known_hole_cards

LOG = logging.getLogger("find_theta")


def _log_preflop_bundle_composition(target: str, metas: List[Dict[str, Any]]) -> None:
    if not metas:
        LOG.info(
            "Preflop EM bundles | target=%s | count=0 (no hands with target preflop actions under observer filter).",
            target,
        )
        return
    by_session: Dict[str, List[Any]] = defaultdict(list)
    for m in metas:
        by_session[str(m["session"])].append(m["hand_number"])
    obs_set = sorted({str(m["observer"]) for m in metas})
    n_dec = sum(int(m.get("n_target_preflop_decisions", 0)) for m in metas)
    LOG.info(
        "Preflop EM bundles | target=%s | bundles=%d | target_preflop_decisions_total=%d | observers=%s | sessions=%d",
        target,
        len(metas),
        n_dec,
        obs_set,
        len(by_session),
    )
    for sess in sorted(by_session.keys(), key=lambda x: (len(x), x)):
        hands = sorted(by_session[sess], key=lambda h: (isinstance(h, str), h))
        preview = hands[:24]
        tail = f" …(+{len(hands) - len(preview)} more)" if len(hands) > len(preview) else ""
        LOG.info("  session %r | hands=%d | hand_numbers=%s%s", sess, len(hands), preview, tail)
    for m in metas:
        LOG.debug(
            "    bundle | session=%s hand=%s observer=%s → target=%s | n_decisions=%s",
            m.get("session"),
            m.get("hand_number"),
            m.get("observer"),
            m.get("target"),
            m.get("n_target_preflop_decisions"),
        )


def _log_postflop_em_hand_composition(target: str, metas: List[Dict[str, Any]]) -> None:
    if not metas:
        LOG.info(
            "Postflop EM hands | target=%s | count=0 (no known-hole postflop rows under filter).",
            target,
        )
        return
    by_session: Dict[str, List[Any]] = defaultdict(list)
    for m in metas:
        by_session[str(m["session"])].append(m["hand_number"])
    LOG.info(
        "Postflop EM hands | target=%s | hands=%d | sessions=%d",
        target,
        len(metas),
        len(by_session),
    )
    for sess in sorted(by_session.keys(), key=lambda x: (len(x), x)):
        hands = sorted(by_session[sess], key=lambda h: (isinstance(h, str), h))
        preview = hands[:24]
        tail = f" …(+{len(hands) - len(preview)} more)" if len(hands) > len(preview) else ""
        LOG.info("  session %r | hands=%d | hand_numbers=%s%s", sess, len(hands), preview, tail)
    for m in metas:
        LOG.debug(
            "    postflop row | session=%s hand=%s | phh=%s",
            m.get("session"),
            m.get("hand_number"),
            m.get("phh_path"),
        )


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
        "--em-pair",
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help=(
            "Two distinct player names: learn θ for both, each using only the other as preflop observer "
            "and postflop table filter (bilateral pair EM). Implies --players A B unless you also set --players "
            "to the same two names."
        ),
    )
    p.add_argument(
        "--preflop-em-observer",
        action="append",
        dest="preflop_em_observers",
        metavar="NAME",
        help=(
            "Restrict preflop EM bundles to these observer names (repeatable). "
            "Ignored when --em-pair is set."
        ),
    )
    p.add_argument(
        "--postflop-require-observer",
        default=None,
        metavar="NAME",
        help=(
            "Only use hands where this player is at the table for postflop EM. Ignored when --em-pair is set."
        ),
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
    em_history_dir: Optional[Path] = None,
    restrict_preflop_observers: Optional[Sequence[str]] = None,
    postflop_require_observer: Optional[str] = None,
    em_pair: Optional[Tuple[str, str]] = None,
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
    if em_pair is not None:
        a, b = em_pair[0], em_pair[1]
        if a == b:
            raise ValueError("em_pair must name two distinct players.")
        if players is not None and set(players) != {a, b}:
            raise ValueError(f"--players must match em_pair exactly; got {players!r} vs pair {(a, b)!r}.")
        resolved = list(players) if players is not None else [a, b]
        LOG.info(
            "Bilateral EM pair %r ↔ %r: each player gets θ_pre/θ_post using only the counterpart as "
            "preflop observer and postflop co-seat filter.",
            a,
            b,
        )
    else:
        resolved = list(players) if players else discover_players(refs)
    if not resolved:
        raise ValueError("No players to learn (empty hands or empty --players).")

    em_hist_root: Optional[Path] = (
        Path(em_history_dir).expanduser().resolve() if em_history_dir is not None else None
    )

    warm: MutableMapping[str, Mapping[str, Any]] = {}
    if warm_start_path is not None:
        ws = load_json(warm_start_path)
        if isinstance(ws, dict) and isinstance(ws.get("players"), dict):
            warm = dict(ws["players"])  # type: ignore[assignment]

    global_restrict_obs = (
        list(dict.fromkeys(restrict_preflop_observers)) if restrict_preflop_observers is not None else None
    )
    if em_pair is None and (global_restrict_obs is not None or postflop_require_observer is not None):
        LOG.info(
            "EM scope | theta_players=%s | restrict_preflop_observers=%s | postflop_require_observer=%s",
            resolved,
            global_restrict_obs,
            postflop_require_observer,
        )

    players_out: Dict[str, Any] = {}
    for player in resolved:
        # One independent EM run per ``player`` string; only hands where this name is in
        # ``hand.player_names`` contribute postflop; only (observer, target) pairs with
        # target == player contribute preflop bundles. Other player names are untouched.
        if em_pair is not None:
            p0, p1 = em_pair
            if player not in (p0, p1):
                raise ValueError(f"Player {player!r} not in em_pair {em_pair!r}.")
            counterpart = p1 if player == p0 else p0
            cur_restrict: Optional[List[str]] = [counterpart]
            cur_postflop_req: Optional[str] = counterpart
        else:
            cur_restrict = global_restrict_obs
            cur_postflop_req = postflop_require_observer

        appears_in_hands = sum(1 for ref in refs if player in ref.hand.player_names)
        bundles, bundle_metas = gather_preflop_bundles_for_target_player(
            refs,
            resolved,
            player,
            restrict_observers=cur_restrict,
        )
        preflop_observers = sorted({m["observer"] for m in bundle_metas})
        _log_preflop_bundle_composition(player, bundle_metas)

        em_log_path: Optional[Path] = None
        em_log_f = None
        if em_hist_root is not None:
            em_hist_root.mkdir(parents=True, exist_ok=True)
            if em_pair is not None:
                em_log_path = em_hist_root / f"{counterpart}_{player}_em.jsonl"
                pair_file = True
            elif cur_restrict is not None and len(resolved) == 1 and len(cur_restrict) == 1:
                em_log_path = em_hist_root / f"{cur_restrict[0]}_{player}_em.jsonl"
                pair_file = True
            else:
                em_log_path = em_hist_root / f"{player}_em.jsonl"
                pair_file = False
            em_log_f = em_log_path.open("w", encoding="utf-8")
            LOG.info(
                "EM history jsonl | path=%s | bilateral_pair=%s",
                em_log_path.resolve(),
                pair_file,
            )

        def _em_write(rec: Dict[str, Any]) -> None:
            if em_log_f is None:
                return
            rec = {**rec, "player": player, "theta_target": player}
            if cur_restrict is not None and len(cur_restrict) == 1:
                rec["preflop_observer_scope"] = cur_restrict[0]
            if em_pair is not None:
                rec["em_pair"] = [em_pair[0], em_pair[1]]
            em_log_f.write(json.dumps(rec, default=str) + "\n")

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
                history_hook=_em_write if em_log_f is not None else None,
                bundle_meta=bundle_metas,
            )
            theta_pre_list = [float(x) for x in theta_pre]
        else:
            theta_pre_list = [0.0, 0.0, 0.0]
            LOG.warning("Player %s: no preflop EM bundles; theta_pre set to zeros.", player)

        # Postflop EM — one observation list per hand (unique global index)
        observations_by_hand: List[List] = []
        postflop_hand_metas: List[Dict[str, Any]] = []
        for ref in refs:
            if player not in ref.hand.player_names:
                continue
            if cur_postflop_req is not None and cur_postflop_req not in ref.hand.player_names:
                continue
            obs = collect_postflop_observations_known_hole_cards(ref.hand, player, ref.global_index)
            if obs is not None:
                observations_by_hand.append([obs])
                _path = Path(ref.source)
                postflop_hand_metas.append(
                    {
                        "session": _path.parent.name,
                        "hand_number": int(_path.stem) if _path.stem.isdigit() else _path.stem,
                        "phh_path": ref.source,
                        "target": player,
                    }
                )

        _log_postflop_em_hand_composition(player, postflop_hand_metas)

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
                history_hook=_em_write if em_log_f is not None else None,
                hand_meta=postflop_hand_metas,
            )
            theta_post_list = [float(x) for x in theta_post]
        else:
            theta_post_list = [0.0, 0.0, 0.0]
            LOG.warning("Player %s: no postflop EM observations; theta_post set to zeros.", player)

        pair_label = None
        if cur_restrict is not None and len(cur_restrict) == 1:
            pair_label = f"{cur_restrict[0]}|{player}"

        players_out[player] = {
            "theta_pre": theta_pre_list,
            "theta_post": theta_post_list,
            "preflop_theta_labels": ["fold_tilt", "call_tilt", "raise_tilt"],
            "postflop_theta_labels": ["fold_tilt", "passive_tilt", "aggression_tilt"],
            "preflop_bundles": len(bundles),
            "postflop_hands": len(observations_by_hand),
            "loaded_hands_with_this_player_name": appears_in_hands,
            "preflop_observer_names_merged_for_this_target": preflop_observers,
            "preflop_em_observers_filter": list(cur_restrict) if cur_restrict is not None else None,
            "postflop_require_observer": cur_postflop_req,
            "observer_target_pair_label": pair_label,
            "em_history_jsonl": str(em_log_path.resolve()) if em_log_path is not None else None,
        }

        if em_log_f is not None:
            em_log_f.close()

    out: Dict[str, Any] = {
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
    if em_pair is not None:
        out["em_scope"] = {
            "mode": "bilateral_pair",
            "em_pair": [em_pair[0], em_pair[1]],
            "note": "Each entry in players uses the counterpart as sole preflop observer and postflop co-seat filter.",
        }
    elif global_restrict_obs is not None or postflop_require_observer is not None:
        out["em_scope"] = {
            "restrict_preflop_observers": list(global_restrict_obs) if global_restrict_obs is not None else None,
            "postflop_require_observer": postflop_require_observer,
        }
    return out


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
        restrict_preflop_observers=args.preflop_em_observers,
        postflop_require_observer=args.postflop_require_observer,
        em_pair=tuple(args.em_pair) if args.em_pair is not None else None,
    )

    dump_json(args.out, payload)
    LOG.info("Wrote %s", Path(args.out).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
