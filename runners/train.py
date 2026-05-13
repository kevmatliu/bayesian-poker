#!/usr/bin/env python3
"""Train population-level baseline weights (global priors) from Pluribus ``.phh`` hands."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .common import (
    HandRef,
    collect_postflop_supervised_rows,
    collect_preflop_supervised_rows,
    dump_json,
    flatten_hands,
    postflop_phi_column_labels,
    preflop_phi_column_labels,
    read_session_names_file,
)
from utils.action.postflop import (
    CALL,
    FOLD as PF_FOLD_POST,
    PHI_DIM,
    PostflopPrior,
    RAISE as PF_RAISE_POST,
    train_baseline_facing_bet,
    train_baseline_no_bet,
)
from utils.action.preflop import (
    CHECK_CALL,
    FOLD,
    HEURISTIC_BETA_PREFLOP,
    PREFLOP_PHI_DIM,
    RAISE,
    train_baseline_preflop,
)

LOG = logging.getLogger("train")


def _flatten_hands_log_progress(phase: str, detail: Dict[str, Any]) -> None:
    """Hook for ``flatten_hands`` — periodic parse progress (same idea as runner hand N/M logs)."""
    if phase == "flatten_hands_jobs":
        LOG.info("Parse queue: %d .phh file(s)", detail.get("n_files", 0))
    elif phase == "flatten_hands_parse":
        done = int(detail.get("done", 0))
        total = int(detail.get("total", 0))
        workers = detail.get("workers")
        if workers:
            LOG.info("Parsed %d/%d .phh files (workers=%s)", done, total, workers)
        else:
            LOG.info("Parsed %d/%d .phh files", done, total)


def _postflop_rows_log_progress(phase: str, detail: Dict[str, Any]) -> None:
    """Hook for ``collect_postflop_supervised_rows`` — same wording as the phase start, with done/total."""
    if phase == "postflop_rows_progress":
        done = int(detail.get("done", 0))
        total = int(detail.get("total", 0))
        LOG.info("Collecting postflop supervised rows (%d/%d hands)…", done, total)


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
        "--postflop-equity-mc",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Flop rollout equity Monte Carlo samples when collecting postflop supervised rows "
            "(Method E). Lower is much faster on large corpora; use 32 for closer parity with "
            "the interactive runner default."
        ),
    )
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
        "--sessions-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Same as --session but read names from a text file: one session directory per line, "
            "# starts a comment. Merged with any --session flags (file first, then CLI, duplicates dropped)."
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
    postflop_equity_mc: int = 8,
) -> Dict[str, Any]:
    """Fit population-level ``beta`` weights and return a JSON-serializable payload.

    Pipeline:

    1. **Load** — Either use ``refs`` directly or call :func:`runners.common.flatten_hands`
       on ``inputs`` with optional session filters.
    2. **Supervised rows** — :func:`runners.common.collect_preflop_supervised_rows`
       and :func:`runners.common.collect_postflop_supervised_rows` (known hole
       cards only). Postflop equity features use ``postflop_equity_mc`` Monte
       Carlo samples on the flop for speed.
    3. **Fit** — Multinomial logit for preflop (3-way) and two postflop heads
       (facing bet: 3-way; no bet: 2-way). If a partition has zero rows, the
       corresponding ``beta`` falls back to built-in heuristics from
       :class:`utils.prior.preflop.PreflopPrior` / :class:`utils.prior.postflop.PostflopPrior`.

    Args:
        inputs: Paths to ``.phh`` files, session dirs, or pluribus roots (ignored if ``refs`` given).
        refs: Pre-parsed hands; skips disk load when callers already flattened.
        session_names: If set, only session subfolders whose **directory name**
            is in this set are loaded under each pluribus root.
        max_sessions: Cap sessions per root after numeric sort.
        preflop_lr, preflop_epochs, preflop_l2: Baseline SGD hyperparameters for preflop.
        postflop_lr, postflop_epochs, postflop_l2: Same for **both** postflop heads.
        postflop_equity_mc: Flop rollout MC sample count when building postflop rows.

    Returns:
        Dict with ``schema``, ``hands_used``, optional ``session_filter`` /
        ``max_sessions``, ``preflop`` block (``beta_preflop``, labels, sample counts),
        and ``postflop`` block (``beta_facing``, ``beta_no_bet``, ``phi_dim``, etc.).
    """
    if refs is None:
        if not inputs:
            raise ValueError("train_global_priors needs inputs or refs")
        resolved = [str(Path(x).expanduser().resolve()) for x in inputs]
        shown = resolved[:5]
        tail = ""
        if len(resolved) > 5:
            tail = f" … (+{len(resolved) - 5} more)"
        LOG.info(
            "Loading hands | paths (%d): %s%s | session_filter=%s | max_sessions=%s",
            len(resolved),
            ", ".join(shown),
            tail,
            list(session_names) if session_names is not None else None,
            max_sessions,
        )
        t_flat = time.perf_counter()
        refs = flatten_hands(
            inputs,
            session_names=session_names,
            max_sessions=max_sessions,
            progress_hook=_flatten_hands_log_progress,
        )
        LOG.info(
            "Loaded %d hands from %d path(s) in %.1fs",
            len(refs),
            len(tuple(inputs)),
            time.perf_counter() - t_flat,
        )

    t_rows = time.perf_counter()
    LOG.info("Collecting preflop supervised rows (%d hands)…", len(refs))
    X_pre, y_pre = collect_preflop_supervised_rows(refs)
    LOG.info(
        "Preflop supervised rows done in %.1fs | samples=%d",
        time.perf_counter() - t_rows,
        int(X_pre.shape[0]),
    )

    t_pf = time.perf_counter()
    n_hands = len(refs)
    if n_hands == 0:
        LOG.info("Collecting postflop supervised rows (0 hands)…")
        Xf, yf, Xn, yn = collect_postflop_supervised_rows(
            refs, postflop_equity_mc=postflop_equity_mc
        )
    else:
        Xf, yf, Xn, yn = collect_postflop_supervised_rows(
            refs,
            progress_hook=_postflop_rows_log_progress,
            postflop_equity_mc=postflop_equity_mc,
        )
    LOG.info(
        "Postflop supervised rows done in %.1fs | facing=%d | no_bet=%d",
        time.perf_counter() - t_pf,
        int(Xf.shape[0]),
        int(Xn.shape[0]),
    )

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
        LOG.info(
            "Fitting preflop baseline | samples=%d | epochs=%d | lr=%s | l2=%s",
            int(X_pre.shape[0]),
            preflop_epochs,
            preflop_lr,
            preflop_l2,
        )
        t_fit = time.perf_counter()
        beta_preflop = train_baseline_preflop(
            X_pre,
            y_pre,
            learning_rate=preflop_lr,
            max_epochs=preflop_epochs,
            l2=preflop_l2,
        )
        LOG.info(
            "Preflop baseline fit done in %.1fs | beta shape %s",
            time.perf_counter() - t_fit,
            beta_preflop.shape,
        )
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
        LOG.info(
            "Fitting postflop baseline (facing bet) | samples=%d | epochs=%d | lr=%s | l2=%s",
            int(Xf.shape[0]),
            postflop_epochs,
            postflop_lr,
            postflop_l2,
        )
        t_face = time.perf_counter()
        beta_facing = train_baseline_facing_bet(
            Xf,
            yf,
            learning_rate=postflop_lr,
            max_epochs=postflop_epochs,
            l2=postflop_l2,
        )
        LOG.info("Postflop facing-bet fit done in %.1fs", time.perf_counter() - t_face)
    else:
        beta_facing = PostflopPrior().beta_facing_matrix
        LOG.warning("No facing-bet postflop rows; using heuristic beta_facing.")

    if Xn.shape[0] > 0:
        LOG.info(
            "Fitting postflop baseline (no bet) | samples=%d | epochs=%d | lr=%s | l2=%s",
            int(Xn.shape[0]),
            postflop_epochs,
            postflop_lr,
            postflop_l2,
        )
        t_nb = time.perf_counter()
        beta_no_bet = train_baseline_no_bet(
            Xn,
            yn,
            learning_rate=postflop_lr,
            max_epochs=postflop_epochs,
            l2=postflop_l2,
        )
        LOG.info("Postflop no-bet fit done in %.1fs", time.perf_counter() - t_nb)
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
        "equity_mc_samples": int(postflop_equity_mc),
        "facing_action_labels": ["fold", "call", "raise"],
        "facing_indices": {"fold": PF_FOLD_POST, "call": CALL, "raise": PF_RAISE_POST},
        "no_bet_action_labels": ["call", "raise"],
        "no_bet_indices": {"call": CALL, "raise": PF_RAISE_POST},
        "training_samples_facing": int(Xf.shape[0]),
        "training_samples_no_bet": int(Xn.shape[0]),
    }

    return payload


def main(argv: List[str] | None = None) -> int:
    """CLI entry: parse args, configure logging, run :func:`train_global_priors`, write JSON."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    LOG.info(
        "Run start | out=%s | preflop train: epochs=%d lr=%s | postflop train: epochs=%d lr=%s | "
        "postflop row equity MC=%d",
        args.out,
        args.preflop_epochs,
        args.preflop_lr,
        args.postflop_epochs,
        args.postflop_lr,
        args.postflop_equity_mc,
    )

    file_sessions = read_session_names_file(args.sessions_file) if args.sessions_file else []
    cli_sessions = list(args.sessions or [])
    session_names = None
    if file_sessions or cli_sessions:
        seen: set[str] = set()
        merged: List[str] = []
        for name in file_sessions + cli_sessions:
            if name not in seen:
                seen.add(name)
                merged.append(name)
        session_names = merged

    if args.sessions_file:
        LOG.info(
            "Sessions file: %s (%d name(s) before merge with --session)",
            args.sessions_file.expanduser().resolve(),
            len(file_sessions),
        )

    payload = train_global_priors(
        args.inputs,
        session_names=session_names,
        max_sessions=args.max_sessions,
        preflop_lr=args.preflop_lr,
        preflop_epochs=args.preflop_epochs,
        preflop_l2=args.preflop_l2,
        postflop_lr=args.postflop_lr,
        postflop_epochs=args.postflop_epochs,
        postflop_l2=args.postflop_l2,
        postflop_equity_mc=args.postflop_equity_mc,
    )

    out_path = dump_json(args.out, payload)
    LOG.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
