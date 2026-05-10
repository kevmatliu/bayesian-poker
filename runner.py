"""CLI entrypoint: single-hand / session filtering, session-split, and hand-level pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent
UTILS_DIR = REPO_ROOT / "utils"

# Repo root must precede ``utils/`` so top-level ``test.py`` / ``train.py`` win over ``utils/test.py``.
for path in (str(UTILS_DIR), str(REPO_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from runner_execute import dump_result_json, run_preflop_filter
from runner_models import EMPostflopRunConfig, EMPreflopRunConfig
from runner_pipeline import pipeline_main
from runner_session_split import session_split_main

LOG = logging.getLogger("runner")


def _configure_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a session or hand file, run preflop range filtering, and optionally learn EM priors."
    )
    parser.add_argument("session_path", help="Directory containing numeric .phh files, or a single .phh file.")
    parser.add_argument(
        "--observer",
        action="append",
        dest="observers",
        help="Observer player name such as MrWhite. Repeat to include multiple observers.",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Target player name whose range is being inferred, such as Gogo.",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=0.0,
        help="Preflop temperature when EM is off (phi shim on PreflopPrior). Ignored for likelihood when --em is set.",
    )
    parser.add_argument(
        "--em",
        action="store_true",
        help="Run EM to learn theta_pre [= [theta_fold, theta_call, theta_raise]] per observer|target pair.",
    )
    parser.add_argument("--em-iters", type=int, default=5, help="Outer EM iterations (E / M cycles).")
    parser.add_argument("--em-m-steps", type=int, default=100, help="Gradient steps in each M-step.")
    parser.add_argument("--em-lr", type=float, default=0.05, help="M-step learning rate.")
    parser.add_argument("--em-l2", type=float, default=0.25, help="L2 penalty on theta_pre in the M-step.")
    parser.add_argument(
        "--em-postflop",
        action="store_true",
        help="Learn post-flop theta_post [fold, passive, aggression] per observer|target (needs target hole cards in .phh).",
    )
    parser.add_argument(
        "--postflop-em-iters",
        type=int,
        default=10,
        help="Outer EM iterations for post-flop theta_post.",
    )
    parser.add_argument("--postflop-m-steps", type=int, default=200, help="Gradient steps per post-flop M-step.")
    parser.add_argument("--postflop-em-lr", type=float, default=0.05, help="Learning rate for post-flop M-step.")
    parser.add_argument("--postflop-em-l2", type=float, default=0.25, help="L2 penalty on post-flop theta_post.")
    parser.add_argument(
        "--postflop-prior-floor",
        type=float,
        default=1e-6,
        help="Probability floor for PostflopPrior during EM.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-probability preflop classes to surface per result.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the runner result as JSON.")
    parser.add_argument("--json-out", help="Optional filepath to write the JSON result to.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Progress logs go to stderr (default INFO). Use DEBUG for per-pair filter lines.",
    )
    parser.add_argument("--quiet", action="store_true", help="Shorthand for --log-level WARNING (errors only).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    log_level = logging.WARNING if args.quiet else getattr(logging, args.log_level)
    _configure_logging(log_level)

    em_cfg = EMPreflopRunConfig(
        enabled=args.em,
        outer_iters=args.em_iters,
        m_steps=args.em_m_steps,
        m_lr=args.em_lr,
        m_l2=args.em_l2,
    )
    postflop_em_cfg = EMPostflopRunConfig(
        enabled=args.em_postflop,
        outer_iters=args.postflop_em_iters,
        m_steps=args.postflop_m_steps,
        m_lr=args.postflop_em_lr,
        m_l2=args.postflop_em_l2,
        prior_floor=args.postflop_prior_floor,
    )
    result = run_preflop_filter(
        source_path=args.session_path,
        observers=args.observers,
        targets=args.targets,
        phi=args.phi,
        top_k=args.top_k,
        em_cfg=em_cfg,
        postflop_em_cfg=postflop_em_cfg,
    )
    LOG.info("Run finished: %d hand-level filter results", len(result.hand_results))

    if args.json_out:
        dump_result_json(result, args.json_out)

    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return 0

    print(f"Session: {result.session_path}")
    print(f"Observers: {', '.join(result.observers) if result.observers else '(none)'}")
    print(f"Targets: {', '.join(result.targets) if result.targets else '(none)'}")
    print(f"Fixed phi (ignored when EM enabled): {result.phi}")
    if result.em.enabled and result.em.theta_pre_by_pair:
        print("EM: learned theta_pre [fold, call, raise] per observer|target:")
        for pair_key, theta in sorted(result.em.theta_pre_by_pair.items()):
            tf, tc, tr_ = theta
            print(f"  {pair_key}: [{tf:.6f}, {tc:.6f}, {tr_:.6f}]")
        print(
            f"EM: outer iters={result.em.outer_iterations}, M-step steps={result.em.m_step_steps}, "
            f"lr={result.em.m_learning_rate}, l2={result.em.m_l2}"
        )
    else:
        print(f"EM: {result.em.note or 'disabled (pass --em to learn theta_pre).'}")
    if result.postflop.enabled and result.postflop.theta_post_by_pair:
        print("Post-flop EM: learned theta_post [fold, passive/call, aggression] per observer|target:")
        for pk in sorted(result.postflop.theta_post_by_pair.keys()):
            tf, tc, tr_ = result.postflop.theta_post_by_pair[pk]
            nh = result.postflop.hands_with_target_cards_per_pair.get(pk, 0)
            print(f"  {pk} ({nh} hands): [{tf:.6f}, {tc:.6f}, {tr_:.6f}]")
        print(
            f"Post-flop EM: outer iters={result.postflop.outer_iterations}, "
            f"M-step steps={result.postflop.m_step_steps}, lr={result.postflop.m_learning_rate}, l2={result.postflop.m_l2}"
        )
    else:
        print(
            f"Post-flop EM: {result.postflop.note or 'disabled (pass --em-postflop; requires target hole cards in history).'}"
        )

    for hand_result in result.hand_results:
        print()
        print(f"Hand {hand_result.hand_index} | {hand_result.observer} -> {hand_result.target}")
        print(f"Observer hole cards: {hand_result.observer_hole_cards or '(unknown)'}")
        print(f"Preflop decisions: {len(hand_result.decisions)}")
        print(f"Preflop log-likelihood: {hand_result.log_likelihood:.6f}")
        for entry in hand_result.top_range:
            print(f"  {entry['hand_class']}: {entry['probability']:.6f}")
        if hand_result.postflop_decisions:
            print(f"Post-flop combo decisions: {len(hand_result.postflop_decisions)}")
            print(f"Post-flop combo log-likelihood: {hand_result.postflop_log_likelihood:.6f}")
            for entry in hand_result.top_combos:
                print(f"  {entry['combo']}: {entry['probability']:.6f}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pipeline":
        raise SystemExit(pipeline_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "session-split":
        raise SystemExit(session_split_main(sys.argv[2:]))
    raise SystemExit(main())


# Re-export common symbols for interactive / library use
__all__ = [
    "EMPostflopRunConfig",
    "EMPreflopRunConfig",
    "dump_result_json",
    "main",
    "pipeline_main",
    "run_preflop_filter",
    "session_split_main",
]
