#!/usr/bin/env python3
"""Train population-level baseline weights (global priors) from Pluribus ``.phh`` hands."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pipeline_common import (
    HandRef,
    collect_postflop_supervised_rows,
    collect_preflop_supervised_rows,
    dump_json,
    flatten_hands,
    postflop_phi_column_labels,
    preflop_phi_column_labels,
)
from utils.prior.postflop import (
    CALL,
    FOLD as PF_FOLD_POST,
    PHI_DIM,
    PostflopPrior,
    RAISE as PF_RAISE_POST,
    train_baseline_facing_bet,
    train_baseline_no_bet,
)
from utils.prior.preflop import (
    CHECK_CALL,
    FOLD,
    HEURISTIC_BETA_PREFLOP,
    PREFLOP_PHI_DIM,
    RAISE,
    train_baseline_preflop,
)

LOG = logging.getLogger("train")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit global preflop + postflop multinomial baselines from labeled hole-card actions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Pluribus-style inputs: each path is either a session folder (one numbered .phh per hand) "
            "or a single .phh file. Parent folders like pluribus/ collect many sessions."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/global_priors.json"),
        help="Output JSON path.",
    )
    p.add_argument("--preflop-lr", type=float, default=0.15)
    p.add_argument("--preflop-epochs", type=int, default=50)
    p.add_argument("--preflop-l2", type=float, default=0.0)
    p.add_argument("--postflop-lr", type=float, default=0.15)
    p.add_argument("--postflop-epochs", type=int, default=50)
    p.add_argument("--postflop-l2", type=float, default=0.0)
    p.add_argument(
        "--session",
        action="append",
        dest="sessions",
        metavar="NAME",
        help=(
            "When loading a Pluribus root (folder of session subfolders), only include these "
            "session directory names, e.g. --session 30 --session 99. Repeatable."
        ),
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Cap how many session subfolders to load from each Pluribus root (after numeric sort).",
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def train_global_priors(
    inputs: Sequence[str | Path] | None = None,
    *,
    refs: Sequence[HandRef] | None = None,
    session_names: Optional[Sequence[str]] = None,
    max_sessions: Optional[int] = None,
    preflop_lr: float = 0.15,
    preflop_epochs: int = 50,
    preflop_l2: float = 0.0,
    postflop_lr: float = 0.15,
    postflop_epochs: int = 50,
    postflop_l2: float = 0.0,
) -> Dict[str, Any]:
    if refs is None:
        if not inputs:
            raise ValueError("train_global_priors needs inputs or refs")
        refs = flatten_hands(
            inputs,
            session_names=session_names,
            max_sessions=max_sessions,
        )
        LOG.info("Loaded %d hands from %d path(s)", len(refs), len(tuple(inputs)))
    else:
        LOG.info("Using %d pre-loaded hand refs", len(refs))

    X_pre, y_pre = collect_preflop_supervised_rows(refs)
    Xf, yf, Xn, yn = collect_postflop_supervised_rows(refs)

    payload: Dict[str, Any] = {
        "schema": "bayesian_poker.global_priors.v1",
        "hands_used": len(refs),
    }
    if session_names is not None:
        payload["session_filter"] = list(session_names)
    if max_sessions is not None:
        payload["max_sessions"] = max_sessions

    # Preflop beta: rows correspond to FOLD, CHECK_CALL, RAISE
    if X_pre.shape[0] > 0:
        beta_preflop = train_baseline_preflop(
            X_pre,
            y_pre,
            learning_rate=preflop_lr,
            max_epochs=preflop_epochs,
            l2=preflop_l2,
        )
        LOG.info("Preflop baseline: %d samples, beta shape %s", X_pre.shape[0], beta_preflop.shape)
    else:
        beta_preflop = HEURISTIC_BETA_PREFLOP.copy()
        LOG.warning("No supervised preflop rows (need hole cards + preflop actions); using heuristic beta_preflop.")

    payload["preflop"] = {
        "beta_preflop": beta_preflop.tolist(),
        "shape": list(beta_preflop.shape),
        "feature_dim": PREFLOP_PHI_DIM,
        "phi_column_labels": preflop_phi_column_labels(),
        "action_labels": ["fold", "call_or_check", "raise"],
        "action_indices": {"fold": FOLD, "call_or_check": CHECK_CALL, "raise": RAISE},
        "training_samples": int(X_pre.shape[0]),
    }

    # Postflop: two heads — facing bet (3-way) and no bet (call vs raise)
    if Xf.shape[0] > 0:
        beta_facing = train_baseline_facing_bet(
            Xf,
            yf,
            learning_rate=postflop_lr,
            max_epochs=postflop_epochs,
            l2=postflop_l2,
        )
    else:
        beta_facing = PostflopPrior().beta_facing_matrix
        LOG.warning("No facing-bet postflop rows; using heuristic beta_facing.")

    if Xn.shape[0] > 0:
        beta_no_bet = train_baseline_no_bet(
            Xn,
            yn,
            learning_rate=postflop_lr,
            max_epochs=postflop_epochs,
            l2=postflop_l2,
        )
    else:
        beta_no_bet = PostflopPrior().beta_no_bet_matrix
        LOG.warning("No no-bet postflop rows; using heuristic beta_no_bet.")

    payload["postflop"] = {
        "beta_facing": np.asarray(beta_facing).tolist(),
        "beta_no_bet": np.asarray(beta_no_bet).tolist(),
        "shape_facing": list(np.asarray(beta_facing).shape),
        "shape_no_bet": list(np.asarray(beta_no_bet).shape),
        "phi_dim": PHI_DIM,
        "phi_column_labels": postflop_phi_column_labels(),
        "facing_action_labels": ["fold", "call", "raise"],
        "facing_indices": {"fold": PF_FOLD_POST, "call": CALL, "raise": PF_RAISE_POST},
        "no_bet_action_labels": ["call", "raise"],
        "no_bet_indices": {"call": CALL, "raise": PF_RAISE_POST},
        "training_samples_facing": int(Xf.shape[0]),
        "training_samples_no_bet": int(Xn.shape[0]),
    }

    return payload


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    payload = train_global_priors(
        args.inputs,
        session_names=args.sessions,
        max_sessions=args.max_sessions,
        preflop_lr=args.preflop_lr,
        preflop_epochs=args.preflop_epochs,
        preflop_l2=args.preflop_l2,
        postflop_lr=args.postflop_lr,
        postflop_epochs=args.postflop_epochs,
        postflop_l2=args.postflop_l2,
    )

    out_path = dump_json(args.out, payload)
    LOG.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
