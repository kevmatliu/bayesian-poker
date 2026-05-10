"""Hand-level train / θ / test pipeline CLI (``runner.py pipeline``)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline_common import dump_json, flatten_hands, split_hand_refs

from find_theta import learn_player_thetas
from test import evaluate_split
from train import train_global_priors

from runner_execute import LOG

REPO_ROOT = Path(__file__).resolve().parent


class PipelineJsonLogger:
    """Structured pipeline steps to stdout and ``logs/logs_<UTC timestamp>.json``."""

    def __init__(self) -> None:
        logs_dir = REPO_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.path = logs_dir / f"logs_{ts}.json"
        self.started_iso = datetime.now(timezone.utc).isoformat()
        self.steps: List[Dict[str, Any]] = []

    def record(self, step: str, **detail: Any) -> None:
        rec: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "step": step,
        }
        if detail:
            rec["detail"] = {k: _json_safe(v) for k, v in detail.items()}
        self.steps.append(rec)
        tail = ""
        if detail:
            tail = " | " + json.dumps(rec["detail"], default=str, sort_keys=True)
        print(f"[pipeline] {step}{tail}", flush=True)

    def finalize(
        self,
        *,
        success: bool,
        exit_code: int,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "run_started_utc": self.started_iso,
            "run_finished_utc": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "exit_code": exit_code,
            "log_file": str(self.path.resolve()),
            "steps": self.steps,
        }
        if summary:
            payload["summary"] = _json_safe(summary)
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[pipeline] wrote_json_log | {self.path.resolve()}", flush=True)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def pipeline_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline: split hands from inputs (default 0.5 train / 0.25 θ / 0.25 test), "
            "train global priors, learn per-player θ, run evaluation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["pluribus"],
        help=(
            "Data paths: a Pluribus *root* (e.g. pluribus/) expands to every immediate session subfolder; "
            "a session folder (e.g. pluribus/30/) loads only that session; a .phh file loads one hand. "
            "Default: pluribus (all sessions under ./pluribus). Pass multiple paths to merge corpora."
        ),
    )
    parser.add_argument(
        "--session",
        action="append",
        dest="sessions",
        metavar="NAME",
        help=(
            "When an input is a Pluribus root, only load these session directory names "
            "(e.g. --session 30 --session 31). Repeatable."
        ),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="When an input is a Pluribus root, load at most this many session subfolders (numeric order).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for hand splits.")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--find-theta-frac", type=float, default=0.25)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument(
        "--global-priors-out",
        type=Path,
        default=Path("artifacts/global_priors.json"),
    )
    parser.add_argument(
        "--player-thetas-out",
        type=Path,
        default=Path("artifacts/player_thetas.json"),
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=Path("artifacts/eval_metrics.json"),
    )
    parser.add_argument("--warm-start-theta", type=Path, default=None)
    parser.add_argument(
        "--preflop-train-epochs",
        type=int,
        default=10,
        help="Multinomial baseline training epochs inside train_global_priors.",
    )
    parser.add_argument(
        "--postflop-train-epochs",
        type=int,
        default=10,
        help="Multinomial baseline training epochs for postflop heads.",
    )
    parser.add_argument("--preflop-theta-em-iters", type=int, default=4)
    parser.add_argument("--preflop-theta-m-steps", type=int, default=80)
    parser.add_argument("--postflop-theta-em-iters", type=int, default=4)
    parser.add_argument("--postflop-theta-m-steps", type=int, default=80)
    parser.add_argument(
        "--parse-workers",
        type=int,
        default=0,
        help=(
            "Parallel worker processes to parse .phh files during flatten_hands (0 = auto, "
            "min(8, CPU count); 1 = single-threaded). Parsing is the main startup bottleneck."
        ),
    )

    args = parser.parse_args(argv)

    pj = PipelineJsonLogger()
    pj.record("pipeline_begin", config=_json_safe(vars(args)))

    try:
        pj.record(
            "flatten_hands_start",
            inputs=list(args.inputs),
            sessions=args.sessions,
            max_sessions=args.max_sessions,
            parse_workers=args.parse_workers,
        )

        def _parse_progress(phase: str, detail: Dict[str, Any]) -> None:
            pj.record(phase, **detail)

        all_refs = flatten_hands(
            args.inputs,
            session_names=args.sessions,
            max_sessions=args.max_sessions,
            parse_workers=args.parse_workers,
            progress_hook=_parse_progress,
        )
        pj.record("flatten_hands_done", n_hands=len(all_refs))

        if len(all_refs) < 3:
            LOG.error("Need at least three hands to split into train / θ / test (got %d).", len(all_refs))
            pj.record("abort", reason="need_at_least_three_hands", n_hands=len(all_refs))
            pj.finalize(
                success=False,
                exit_code=2,
                summary={"n_hands": len(all_refs)},
            )
            return 2

        pj.record(
            "split_hands_start",
            seed=args.seed,
            train_frac=args.train_frac,
            find_theta_frac=args.find_theta_frac,
            test_frac=args.test_frac,
        )
        train_refs, theta_refs, test_refs = split_hand_refs(
            all_refs,
            train_frac=args.train_frac,
            find_theta_frac=args.find_theta_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        pj.record(
            "split_hands_done",
            n_train=len(train_refs),
            n_find_theta=len(theta_refs),
            n_test=len(test_refs),
        )

        LOG.info(
            "Loaded %d hands from %s | split (%g/%g/%g): train=%d | find_theta=%d | test=%d",
            len(all_refs),
            ", ".join(args.inputs),
            args.train_frac,
            args.find_theta_frac,
            args.test_frac,
            len(train_refs),
            len(theta_refs),
            len(test_refs),
        )

        pj.record(
            "train_global_priors_start",
            n_train_hands=len(train_refs),
            preflop_train_epochs=args.preflop_train_epochs,
            postflop_train_epochs=args.postflop_train_epochs,
        )
        gp = train_global_priors(
            refs=train_refs,
            preflop_epochs=args.preflop_train_epochs,
            postflop_epochs=args.postflop_train_epochs,
        )
        pre = gp.get("preflop") or {}
        post = gp.get("postflop") or {}
        pj.record(
            "train_global_priors_done",
            hands_used=gp.get("hands_used"),
            preflop_training_samples=pre.get("training_samples"),
            postflop_training_samples_facing=post.get("training_samples_facing"),
            postflop_training_samples_no_bet=post.get("training_samples_no_bet"),
        )

        pj.record("write_artifact_start", kind="global_priors", path=args.global_priors_out)
        dump_json(args.global_priors_out, gp)
        pj.record("write_artifact_done", kind="global_priors", path=args.global_priors_out)
        LOG.info("Wrote global priors → %s", args.global_priors_out.resolve())

        pj.record(
            "learn_player_thetas_start",
            n_theta_hands=len(theta_refs),
            preflop_theta_em_iters=args.preflop_theta_em_iters,
            preflop_theta_m_steps=args.preflop_theta_m_steps,
            postflop_theta_em_iters=args.postflop_theta_em_iters,
            postflop_theta_m_steps=args.postflop_theta_m_steps,
            warm_start_theta=args.warm_start_theta,
        )
        thetas_payload = learn_player_thetas(
            refs=theta_refs,
            global_priors_path=args.global_priors_out,
            warm_start_path=args.warm_start_theta,
            preflop_em_iters=args.preflop_theta_em_iters,
            preflop_m_steps=args.preflop_theta_m_steps,
            postflop_em_iters=args.postflop_theta_em_iters,
            postflop_m_steps=args.postflop_theta_m_steps,
        )
        players_block = thetas_payload.get("players") or {}
        pj.record(
            "learn_player_thetas_done",
            n_players=len(players_block),
            player_names_sorted=sorted(players_block.keys()),
            note="Each name is fit independently; see player_thetas.json player_identity.",
        )

        pj.record("write_artifact_start", kind="player_thetas", path=args.player_thetas_out)
        dump_json(args.player_thetas_out, thetas_payload)
        pj.record("write_artifact_done", kind="player_thetas", path=args.player_thetas_out)
        LOG.info("Wrote player θ → %s", args.player_thetas_out.resolve())

        pj.record("evaluate_split_start", n_test_hands=len(test_refs))
        metrics = evaluate_split(
            refs=test_refs,
            global_priors_path=args.global_priors_out,
            player_thetas_path=args.player_thetas_out,
        )
        pj.record("evaluate_split_done", aggregate=metrics.get("aggregate"))

        pj.record("write_artifact_start", kind="eval_metrics", path=args.eval_out)
        dump_json(args.eval_out, metrics)
        pj.record("write_artifact_done", kind="eval_metrics", path=args.eval_out)
        LOG.info("Wrote eval metrics → %s", args.eval_out.resolve())

        summary = {
            "aggregate": metrics.get("aggregate"),
            "artifacts": {
                "global_priors": str(args.global_priors_out.resolve()),
                "player_thetas": str(args.player_thetas_out.resolve()),
                "eval_metrics": str(args.eval_out.resolve()),
            },
            "players_learned": sorted(players_block.keys()),
            "theta_keying": "One θ vector per exact .phh player name; EM loops are independent per name.",
        }
        pj.finalize(success=True, exit_code=0, summary=summary)
        print(json.dumps(metrics["aggregate"], indent=2))
        return 0
    except Exception as exc:
        LOG.exception("Pipeline failed: %s", exc)
        pj.record(
            "pipeline_exception",
            error=str(exc),
            exc_type=type(exc).__name__,
        )
        pj.finalize(
            success=False,
            exit_code=1,
            summary={"error": str(exc), "exc_type": type(exc).__name__},
        )
        raise
