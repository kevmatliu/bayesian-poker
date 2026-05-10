"""Session-list split pipeline (``runner.py session-split``)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipeline_common import dump_json, flatten_hands, read_session_names_file, split_session_names
from utils.filter import ComboRangeFilter
from utils.parse import Session
from utils.postflop_runner_bridge import POSTFLOP_STREETS
from utils.prior.postflop import PostflopPrior
from utils.prior.preflop import PreflopPrior

from find_theta import learn_player_thetas, load_global_priors
from train import train_global_priors

from runner_execute import LOG, PREFLOP_PRIOR_FLOOR, _run_preflop_filter_for_hand, all_combo_keys


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


def _priors_for_online_target(
    target: str,
    global_priors_path: Path,
    players_block: Dict[str, Any],
) -> Tuple[PreflopPrior, PostflopPrior]:
    beta_preflop, beta_facing, beta_no_bet = load_global_priors(global_priors_path)
    if target not in players_block:
        raise ValueError(
            f"Target {target!r} missing from player θ JSON (not seen during EM?). "
            f"Keys: {sorted(players_block.keys())}"
        )
    entry = players_block[target]
    tp = entry["theta_pre"]
    ts = entry["theta_post"]
    return (
        PreflopPrior(
            theta_pre=tuple(float(x) for x in tp),
            floor=PREFLOP_PRIOR_FLOOR,
            beta_preflop=beta_preflop,
        ),
        PostflopPrior(
            theta_post=tuple(float(x) for x in ts),
            floor=1e-6,
            beta_facing=beta_facing,
            beta_no_bet=beta_no_bet,
        ),
    )


def session_split_main(argv: Optional[List[str]] = None) -> int:
    try:
        import pandas as pd
    except ImportError:
        LOG.error("session-split requires pandas (pip install pandas).")
        return 1

    parser = argparse.ArgumentParser(
        description=(
            "Split sessions from a list file: 50% train global priors, 30% bilateral EM (1 outer iter) "
            "for two players, 20% online filtering from both perspectives + combo-range CSV."
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
        "--players",
        nargs=2,
        metavar=("P1", "P2"),
        required=True,
        help="Exactly two player names. EM learns θ for both (each from the other's perspective); "
        "online filtering runs P1→P2 and P2→P1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-session-frac", type=float, default=0.5)
    parser.add_argument("--em-session-frac", type=float, default=0.3)
    parser.add_argument("--online-session-frac", type=float, default=0.2)
    parser.add_argument("--global-priors-out", type=Path, default=Path("artifacts/global_priors.json"))
    parser.add_argument("--player-thetas-out", type=Path, default=Path("artifacts/player_thetas.json"))
    parser.add_argument("--range-csv-out", type=Path, default=Path("artifacts/online_range_history.csv"))
    parser.add_argument("--thetas-dir", type=Path, default=Path("thetas"))
    parser.add_argument("--parse-workers", type=int, default=0)
    parser.add_argument("--preflop-train-epochs", type=int, default=50)
    parser.add_argument("--postflop-train-epochs", type=int, default=50)
    parser.add_argument("--preflop-m-steps", type=int, default=100)
    parser.add_argument("--postflop-m-steps", type=int, default=200)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    p_a, p_b = args.players[0], args.players[1]
    if p_a == p_b:
        LOG.error("--players must name two distinct players.")
        return 2

    tw = args.train_session_frac + args.em_session_frac + args.online_session_frac
    if abs(tw - 1.0) > 1e-6:
        LOG.error("Train/EM/online session fractions must sum to 1.0 (got %s).", tw)
        return 2

    pluribus_root = args.pluribus_root.expanduser().resolve()
    if not pluribus_root.is_dir():
        LOG.error("Not a directory: %s", pluribus_root)
        return 2

    session_names = read_session_names_file(args.sessions_file)
    if not session_names:
        LOG.error("No sessions in %s", args.sessions_file)
        return 2

    for name in session_names:
        if not (pluribus_root / name).is_dir():
            LOG.error("Session folder missing: %s", pluribus_root / name)
            return 2

    train_s, em_s, online_s = split_session_names(
        session_names,
        train_frac=args.train_session_frac,
        em_frac=args.em_session_frac,
        online_frac=args.online_session_frac,
        seed=args.seed,
    )
    LOG.info("Session split (seed=%s): train=%s | EM=%s | online=%s", args.seed, train_s, em_s, online_s)

    if not train_s or not em_s or not online_s:
        LOG.error("Each of train / EM / online session splits must be non-empty.")
        return 2

    train_inputs = [pluribus_root / s for s in train_s]
    em_inputs = [pluribus_root / s for s in em_s]
    online_inputs = [pluribus_root / s for s in online_s]

    LOG.info("Loading train hands from %d sessions…", len(train_inputs))
    train_refs = flatten_hands(train_inputs, parse_workers=args.parse_workers)
    if not train_refs:
        LOG.error("No hands parsed for train split.")
        return 2

    LOG.info("Training global priors on %d hands…", len(train_refs))
    gp = train_global_priors(
        refs=train_refs,
        preflop_epochs=args.preflop_train_epochs,
        postflop_epochs=args.postflop_train_epochs,
    )
    dump_json(args.global_priors_out, gp)
    LOG.info("Wrote global priors → %s", args.global_priors_out.resolve())

    LOG.info("Loading EM hands from %d sessions…", len(em_inputs))
    em_refs = flatten_hands(em_inputs, parse_workers=args.parse_workers)
    if not em_refs:
        LOG.error("No hands parsed for EM split.")
        return 2

    players_em = sorted({p for ref in em_refs for p in ref.hand.player_names})
    for name in (p_a, p_b):
        if name not in players_em:
            LOG.error(
                "Player %r does not appear in any EM-split hand (required for bilateral EM). Present: %s",
                name,
                players_em,
            )
            return 2

    em_pair = (p_a, p_b)
    LOG.info(
        "Bilateral EM + online filtering | pair=%r ↔ %r | JSON will contain both players' θ; "
        "online phase runs both filter directions.",
        p_a,
        p_b,
    )

    thetas_payload = learn_player_thetas(
        refs=em_refs,
        global_priors_path=args.global_priors_out,
        players=[p_a, p_b],
        em_pair=em_pair,
        preflop_em_iters=1,
        postflop_em_iters=1,
        preflop_m_steps=args.preflop_m_steps,
        postflop_m_steps=args.postflop_m_steps,
        em_history_dir=args.thetas_dir,
    )
    dump_json(args.player_thetas_out, thetas_payload)
    LOG.info("Wrote player θ (both players) → %s", args.player_thetas_out.resolve())

    players_block = thetas_payload.get("players") or {}
    priors_by_target: Dict[str, Tuple[PreflopPrior, PostflopPrior]] = {}
    for pname in (p_a, p_b):
        try:
            priors_by_target[pname] = _priors_for_online_target(
                pname,
                args.global_priors_out,
                players_block,
            )
        except ValueError as exc:
            LOG.error("%s", exc)
            return 2

    perspectives: Tuple[Tuple[str, str], ...] = ((p_a, p_b), (p_b, p_a))
    combo_cols = all_combo_keys()
    csv_rows: List[Dict[str, Any]] = []

    for session_name in online_s:
        session_path = pluribus_root / session_name
        session = Session(session_path)
        session.parse()
        phh_files = sorted(
            (p for p in session_path.iterdir() if p.is_file() and p.suffix.lower() == ".phh"),
            key=lambda p: int(p.stem),
        )
        for hi, hand in enumerate(session.hands):
            stem = phh_files[hi].stem if hi < len(phh_files) else str(hi)
            hand_number = int(stem) if stem.isdigit() else stem
            if p_a not in hand.player_names or p_b not in hand.player_names:
                continue

            for observer, target in perspectives:
                learned_pre, learned_post = priors_by_target[target]
                tag = f"online sess={session_name} hand={hand_number} idx={hi} {observer}→{target}"
                LOG.info("Online filter | start perspective | %s", tag)
                street_snaps: List[Tuple[str, Dict[str, float]]] = []
                res = _run_preflop_filter_for_hand(
                    hand=hand,
                    observer=observer,
                    target=target,
                    hand_index=hi,
                    phi=0.0,
                    top_k=10,
                    learned_prior=learned_pre,
                    learned_postflop_prior=learned_post,
                    street_end_snapshots=street_snaps,
                    filter_verbose=True,
                    filter_tag=tag,
                )
                if res is None:
                    LOG.info("Online filter | skip (no result) | %s", tag)
                    continue
                csv_rows.extend(
                    _range_history_rows_for_hand(
                        hand,
                        session_name,
                        hand_number,
                        observer,
                        target,
                        res.final_range,
                        street_snaps,
                        combo_cols,
                    )
                )

    if not csv_rows:
        LOG.warning("No online range rows produced (check both players act preflop in online sessions).")
    else:
        df = pd.DataFrame(csv_rows)
        meta_cols = ["session", "hand_number", "street", "observer", "target"]
        combo_only = [c for c in combo_cols if c in df.columns]
        df = df[meta_cols + combo_only]
        args.range_csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.range_csv_out, index=False)
        LOG.info("Wrote range history CSV (%d rows) → %s", len(df), args.range_csv_out.resolve())

    return 0
