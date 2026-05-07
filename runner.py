from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
UTILS_DIR = REPO_ROOT / "utils"

for path in (str(REPO_ROOT), str(UTILS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.em import (
    PostflopThetaObservation,
    PreflopEMDecision,
    PreflopEMHandBundle,
    run_postflop_theta_em,
    run_preflop_em,
)
from utils.filter import PreflopRangeFilter
from utils.filter.helpers import initial_class_prior, normalize
from utils.prior.preflop import PreflopPrior, state_key_from_parse_state
from utils.parse import Hand, Session
from utils.postflop_runner_bridge import collect_session_postflop_hands_by_pair

PREFLOP_PRIOR_FLOOR = 0.01

LOG = logging.getLogger("runner")


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


def _configure_logging(level: int) -> None:
    """Log progress to stderr so stdout stays clean for ``--json``."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


@dataclass(frozen=True)
class EMPreflopRunConfig:
    """CLI knobs for EM on theta_pre (fold / call / raise tendencies)."""

    enabled: bool = False
    outer_iters: int = 5
    m_steps: int = 100
    m_lr: float = 0.05
    m_l2: float = 0.25


@dataclass(frozen=True)
class EMPreflopResult:
    enabled: bool = False
    theta_pre_by_pair: Dict[str, List[float]] = field(default_factory=dict)
    outer_iterations: int = 0
    m_step_steps: int = 0
    m_learning_rate: float = 0.0
    m_l2: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class EMPostflopRunConfig:
    """CLI knobs for EM on theta_post (fold / passive / aggression)."""

    enabled: bool = False
    outer_iters: int = 10
    m_steps: int = 200
    m_lr: float = 0.05
    m_l2: float = 0.25
    prior_floor: float = 1e-6


@dataclass(frozen=True)
class EMPostflopResult:
    enabled: bool = False
    theta_post_by_pair: Dict[str, List[float]] = field(default_factory=dict)
    outer_iterations: int = 0
    m_step_steps: int = 0
    m_learning_rate: float = 0.0
    m_l2: float = 0.0
    hands_with_target_cards_per_pair: Dict[str, int] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class PreflopDecision:
    hand_index: int
    observer: str
    target: str
    action_index: int
    state_key: str
    action_bucket: int
    amount: int


@dataclass
class HandFilterResult:
    hand_index: int
    observer: str
    target: str
    observer_hole_cards: str
    phi: float
    decisions: List[PreflopDecision] = field(default_factory=list)
    top_range: List[Dict[str, float]] = field(default_factory=list)
    final_range: Dict[str, float] = field(default_factory=dict)
    log_likelihood: float = 0.0


@dataclass
class RunnerResult:
    session_path: str
    observers: List[str]
    targets: List[str]
    phi: float
    hand_results: List[HandFilterResult]
    em: EMPreflopResult
    postflop: EMPostflopResult


def _resolve_requested_players(session: Session, requested_players: Optional[Iterable[str]]) -> List[str]:
    available = list(session.hands[0].player_names) if session.hands else []
    if requested_players:
        resolved = list(dict.fromkeys(requested_players))
        missing = [player for player in resolved if player not in available]
        if missing:
            raise ValueError(f"Unknown player names: {missing}. Available players: {available}")
        return resolved
    if not session.hands:
        return []
    return available


def _preflop_decisions_for_hand(
    hand,
    observer: str,
    target: str,
    hand_index: int,
) -> List[PreflopDecision]:
    decisions: List[PreflopDecision] = []
    preflop_actions = hand.actions.get("pre-flop", {})
    preflop_states = hand.states.get("pre-flop", [])

    for action_index in sorted(preflop_actions):
        actor, (_, _raise_level), amount = preflop_actions[action_index]
        if actor != target:
            continue
        if action_index >= len(preflop_states):
            continue

        state = preflop_states[action_index]
        state_key = state_key_from_parse_state(state, target).as_string()
        action_bucket = preflop_actions[action_index][1][0]
        decisions.append(
            PreflopDecision(
                hand_index=hand_index,
                observer=observer,
                target=target,
                action_index=action_index,
                state_key=state_key,
                action_bucket=action_bucket,
                amount=amount,
            )
        )

    return decisions


def _collect_grouped_em_bundles(
    hands: Sequence[Hand],
    observers: Sequence[str],
    targets: Sequence[str],
) -> Dict[Tuple[str, str], List[PreflopEMHandBundle]]:
    """One bundle per (hand, observer, target) with at least one target pre-flop action."""
    groups: Dict[Tuple[str, str], List[PreflopEMHandBundle]] = defaultdict(list)
    for hand_index, hand in enumerate(hands):
        for observer in observers:
            for target in targets:
                decisions = _preflop_decisions_for_hand(hand, observer, target, hand_index)
                if not decisions:
                    continue
                dead = hand.hole_cards.get(observer, "")
                initial_range = normalize(initial_class_prior(dead_cards=dead))
                bundle = PreflopEMHandBundle(
                    tuple(
                        PreflopEMDecision(d.state_key, d.action_bucket)
                        for d in decisions
                    ),
                    initial_range,
                )
                groups[(observer, target)].append(bundle)
    return groups


def _run_preflop_em_per_pair(
    groups: Dict[Tuple[str, str], List[PreflopEMHandBundle]],
    em_cfg: EMPreflopRunConfig,
) -> Dict[str, List[float]]:
    """Learn one theta_pre vector per (observer, target) pair."""
    out: Dict[str, List[float]] = {}
    pairs_non_empty = [(pair, bs) for pair, bs in groups.items() if bs]
    total_pairs = len(pairs_non_empty)
    for idx, ((observer, target), bundles) in enumerate(pairs_non_empty, start=1):
        pair_label = f"{observer}|{target}"
        n_decisions = sum(len(b.decisions) for b in bundles)
        LOG.info(
            "EM pair %d/%d: %s (%d hands / bundles, %d target preflop decisions)",
            idx,
            total_pairs,
            pair_label,
            len(bundles),
            n_decisions,
        )
        theta, _ = run_preflop_em(
            bundles,
            prior_floor=PREFLOP_PRIOR_FLOOR,
            num_em_iters=em_cfg.outer_iters,
            m_l2=em_cfg.m_l2,
            m_lr=em_cfg.m_lr,
            m_steps=em_cfg.m_steps,
        )
        out[pair_label] = [float(x) for x in theta]
        LOG.info(
            "EM finished %s: theta_pre [fold, call, raise] = [%.6f, %.6f, %.6f]",
            pair_label,
            out[pair_label][0],
            out[pair_label][1],
            out[pair_label][2],
        )
    return out


def _run_postflop_em_per_pair(
    groups: Dict[Tuple[str, str], List[List[PostflopThetaObservation]]],
    pf_cfg: EMPostflopRunConfig,
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    """Learn theta_post per observer|target from hands where target hole cards are known."""
    theta_out: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    nonempty = [(pair, seq) for pair, seq in groups.items() if seq]
    total_pairs = len(nonempty)
    for idx, ((observer, target), observations_by_hand) in enumerate(nonempty, start=1):
        pair_label = f"{observer}|{target}"
        n_hands = len(observations_by_hand)
        LOG.info(
            "Post-flop EM pair %d/%d: %s (%d hands with known target hole cards)",
            idx,
            total_pairs,
            pair_label,
            n_hands,
        )
        theta, _ = run_postflop_theta_em(
            observations_by_hand,
            prior_floor=pf_cfg.prior_floor,
            num_em_iters=pf_cfg.outer_iters,
            m_lr=pf_cfg.m_lr,
            m_steps=pf_cfg.m_steps,
            m_l2=pf_cfg.m_l2,
        )
        theta_out[pair_label] = [float(x) for x in theta]
        counts[pair_label] = n_hands
        LOG.info(
            "Post-flop EM finished %s: theta_post [fold, passive, agg] = [%.6f, %.6f, %.6f]",
            pair_label,
            theta_out[pair_label][0],
            theta_out[pair_label][1],
            theta_out[pair_label][2],
        )
    return theta_out, counts


def _compute_postflop_em_result(
    hands: Sequence[Hand],
    resolved_observers: List[str],
    resolved_targets: List[str],
    pf_cfg: Optional[EMPostflopRunConfig],
) -> EMPostflopResult:
    if not pf_cfg or not pf_cfg.enabled:
        return EMPostflopResult()

    LOG.info("Post-flop EM: collecting hands with known target hole cards")
    groups = collect_session_postflop_hands_by_pair(
        list(hands), resolved_observers, resolved_targets
    )
    n_bundles = sum(len(v) for v in groups.values())
    LOG.info("Post-flop EM: %d hand bundles (pre pair-breakdown)", n_bundles)

    if n_bundles == 0:
        LOG.warning(
            "Post-flop EM: no usable hands (need target hole cards in .phh and post-flop actions)."
        )
        return EMPostflopResult(
            enabled=True,
            note="No post-flop EM data: unknown target cards or no post-flop target actions.",
        )

    theta_pf, counts_pf = _run_postflop_em_per_pair(groups, pf_cfg)
    for obs in resolved_observers:
        for tgt in resolved_targets:
            key = f"{obs}|{tgt}"
            counts_pf.setdefault(key, 0)

    return EMPostflopResult(
        enabled=True,
        theta_post_by_pair=theta_pf,
        outer_iterations=pf_cfg.outer_iters,
        m_step_steps=pf_cfg.m_steps,
        m_learning_rate=pf_cfg.m_lr,
        m_l2=pf_cfg.m_l2,
        hands_with_target_cards_per_pair=counts_pf,
        note="Learned from hands where the target's hole cards appear in the history.",
    )


def _learned_prior_for_pair(
    observer: str,
    target: str,
    theta_by_pair: Dict[str, List[float]],
) -> Optional[PreflopPrior]:
    key = f"{observer}|{target}"
    if key not in theta_by_pair:
        return None
    return PreflopPrior(theta_pre=tuple(theta_by_pair[key]), floor=PREFLOP_PRIOR_FLOOR)


def _run_preflop_filter_for_hand(
    hand,
    observer: str,
    target: str,
    hand_index: int,
    phi: float,
    top_k: int,
    learned_prior: Optional[PreflopPrior] = None,
) -> Optional[HandFilterResult]:
    if observer == target:
        return None

    decisions = _preflop_decisions_for_hand(hand, observer, target, hand_index)
    if not decisions:
        return None

    observer_hole_cards = hand.hole_cards.get(observer, "")
    preflop_filter = PreflopRangeFilter(
        observer_name=observer,
        target_name=target,
        observer_hole_cards=observer_hole_cards,
        prior_model=learned_prior,
    )
    if learned_prior is None:
        preflop_filter.phi = phi

    for decision in decisions:
        preflop_filter.update(decision.state_key, decision.action_bucket)

    hand.set_hand_range_vector(observer, target, preflop_filter.range)

    top_range = [
        {"hand_class": hand_class, "probability": prob}
        for hand_class, prob in preflop_filter.top_k(top_k)
    ]

    return HandFilterResult(
        hand_index=hand_index,
        observer=observer,
        target=target,
        observer_hole_cards=observer_hole_cards,
        phi=phi,
        decisions=decisions,
        top_range=top_range,
        final_range=preflop_filter.range,
        log_likelihood=preflop_filter.log_likelihood(),
    )


def run_hand_preflop_filter(
    hand: Hand,
    observers: Optional[Iterable[str]] = None,
    targets: Optional[Iterable[str]] = None,
    phi: float = 0.0,
    top_k: int = 10,
    hand_index: int = 0,
    source_label: str = "<single-hand>",
    em_cfg: Optional[EMPreflopRunConfig] = None,
    postflop_em_cfg: Optional[EMPostflopRunConfig] = None,
) -> RunnerResult:
    available_players = list(hand.player_names)
    resolved_observers = list(dict.fromkeys(observers)) if observers else available_players
    resolved_targets = list(dict.fromkeys(targets)) if targets else available_players

    missing_observers = [player for player in resolved_observers if player not in available_players]
    missing_targets = [player for player in resolved_targets if player not in available_players]
    if missing_observers or missing_targets:
        raise ValueError(
            f"Unknown players. observers={missing_observers}, targets={missing_targets}, available={available_players}"
        )

    theta_by_pair: Dict[str, List[float]] = {}
    em_result = EMPreflopResult()

    if em_cfg and em_cfg.enabled:
        LOG.info("EM enabled: collecting preflop bundles (single source hand)")
        groups = _collect_grouped_em_bundles([hand], resolved_observers, resolved_targets)
        n_bundles = sum(len(bs) for bs in groups.values())
        LOG.info(
            "EM: %d (observer, target) groups, %d total bundles",
            len([1 for bs in groups.values() if bs]),
            n_bundles,
        )
        if any(groups.values()):
            theta_by_pair = _run_preflop_em_per_pair(groups, em_cfg)
            em_result = EMPreflopResult(
                enabled=True,
                theta_pre_by_pair=theta_by_pair,
                outer_iterations=em_cfg.outer_iters,
                m_step_steps=em_cfg.m_steps,
                m_learning_rate=em_cfg.m_lr,
                m_l2=em_cfg.m_l2,
                note="theta_pre replaces --phi for the action prior when EM is enabled.",
            )
        else:
            LOG.warning("EM requested but no target pre-flop actions were found for this hand.")
            em_result = EMPreflopResult(
                enabled=True,
                note="EM requested but no target pre-flop actions were found for this hand.",
            )

    pf_result = _compute_postflop_em_result(
        [hand], resolved_observers, resolved_targets, postflop_em_cfg
    )

    hand_results: List[HandFilterResult] = []
    pair_jobs = len(resolved_observers) * len(resolved_targets)
    LOG.info(
        "Preflop filter single hand (%s): %d observer×target jobs",
        source_label,
        pair_jobs,
    )
    for observer in resolved_observers:
        for target in resolved_targets:
            learned = _learned_prior_for_pair(observer, target, theta_by_pair)
            result = _run_preflop_filter_for_hand(
                hand=hand,
                observer=observer,
                target=target,
                hand_index=hand_index,
                phi=phi,
                top_k=top_k,
                learned_prior=learned,
            )
            if result is not None:
                hand_results.append(result)
                LOG.debug(
                    "Filtered hand_index=%s %s→%s (%d decisions, log_L=%.4f)",
                    hand_index,
                    observer,
                    target,
                    len(result.decisions),
                    result.log_likelihood,
                )
    LOG.info(
        "Preflop filter done (1 hand): %d results from %d jobs",
        len(hand_results),
        pair_jobs,
    )

    return RunnerResult(
        session_path=source_label,
        observers=resolved_observers,
        targets=resolved_targets,
        phi=phi,
        hand_results=hand_results,
        em=em_result,
        postflop=pf_result,
    )


def run_session_preflop_filter(
    session_path: str | Path,
    observers: Optional[Iterable[str]] = None,
    targets: Optional[Iterable[str]] = None,
    phi: float = 0.0,
    top_k: int = 10,
    em_cfg: Optional[EMPreflopRunConfig] = None,
    postflop_em_cfg: Optional[EMPostflopRunConfig] = None,
) -> RunnerResult:
    path_resolved = str(Path(session_path).expanduser().resolve())
    LOG.info("Loading session: %s", path_resolved)
    session = Session(session_path)
    session.parse()

    resolved_observers = _resolve_requested_players(session, observers)
    resolved_targets = _resolve_requested_players(session, targets)
    LOG.info(
        "Parsed %d hands | observers=%s targets=%s",
        len(session.hands),
        resolved_observers,
        resolved_targets,
    )

    theta_by_pair: Dict[str, List[float]] = {}
    em_result = EMPreflopResult()

    if em_cfg and em_cfg.enabled:
        LOG.info("EM enabled: collecting preflop bundles across session")
        groups = _collect_grouped_em_bundles(session.hands, resolved_observers, resolved_targets)
        n_bundles = sum(len(bs) for bs in groups.values())
        groups_with_data = sum(1 for bs in groups.values() if bs)
        LOG.info(
            "EM: %d (observer, target) groups with data, %d total bundles",
            groups_with_data,
            n_bundles,
        )
        if any(groups.values()):
            theta_by_pair = _run_preflop_em_per_pair(groups, em_cfg)
            em_result = EMPreflopResult(
                enabled=True,
                theta_pre_by_pair=theta_by_pair,
                outer_iterations=em_cfg.outer_iters,
                m_step_steps=em_cfg.m_steps,
                m_learning_rate=em_cfg.m_lr,
                m_l2=em_cfg.m_l2,
                note="theta_pre replaces --phi for the action prior when EM is enabled.",
            )
        else:
            LOG.warning("EM requested but no target pre-flop actions were found in the session.")
            em_result = EMPreflopResult(
                enabled=True,
                note="EM requested but no target pre-flop actions were found in the session.",
            )

    pf_result = _compute_postflop_em_result(
        session.hands, resolved_observers, resolved_targets, postflop_em_cfg
    )

    hand_results: List[HandFilterResult] = []
    total_hands = len(session.hands)
    pair_jobs_per_hand = len(resolved_observers) * len(resolved_targets)
    LOG.info(
        "Preflop filtering %d hands × %d observer×target jobs (max %d filter runs)",
        total_hands,
        pair_jobs_per_hand,
        total_hands * pair_jobs_per_hand,
    )
    for hand_index, hand in enumerate(session.hands):
        LOG.info(
            "Preflop filter hand %d/%d (phh index in session)",
            hand_index + 1,
            total_hands,
        )
        for observer in resolved_observers:
            for target in resolved_targets:
                learned = _learned_prior_for_pair(observer, target, theta_by_pair)
                result = _run_preflop_filter_for_hand(
                    hand=hand,
                    observer=observer,
                    target=target,
                    hand_index=hand_index,
                    phi=phi,
                    top_k=top_k,
                    learned_prior=learned,
                )
                if result is not None:
                    hand_results.append(result)
                    LOG.debug(
                        "Hand %d: %s→%s (%d decisions, log_L=%.4f)",
                        hand_index,
                        observer,
                        target,
                        len(result.decisions),
                        result.log_likelihood,
                    )
    LOG.info("Preflop filtering complete: %d hand results collected", len(hand_results))

    return RunnerResult(
        session_path=path_resolved,
        observers=resolved_observers,
        targets=resolved_targets,
        phi=phi,
        hand_results=hand_results,
        em=em_result,
        postflop=pf_result,
    )


def run_preflop_filter(
    source_path: str | Path,
    observers: Optional[Iterable[str]] = None,
    targets: Optional[Iterable[str]] = None,
    phi: float = 0.0,
    top_k: int = 10,
    em_cfg: Optional[EMPreflopRunConfig] = None,
    postflop_em_cfg: Optional[EMPostflopRunConfig] = None,
) -> RunnerResult:
    source = Path(source_path).expanduser().resolve()
    LOG.info("Run start: %s (%s)", source, "session directory" if source.is_dir() else "single hand file")
    if source.is_dir():
        return run_session_preflop_filter(
            session_path=source,
            observers=observers,
            targets=targets,
            phi=phi,
            top_k=top_k,
            em_cfg=em_cfg,
            postflop_em_cfg=postflop_em_cfg,
        )
    if source.is_file():
        hand = Hand.from_file(source)
        return run_hand_preflop_filter(
            hand=hand,
            observers=observers,
            targets=targets,
            phi=phi,
            top_k=top_k,
            hand_index=0,
            source_label=str(source),
            em_cfg=em_cfg,
            postflop_em_cfg=postflop_em_cfg,
        )
    raise FileNotFoundError(f"Source path does not exist: {source}")


def dump_result_json(result: RunnerResult, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return output


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
    parser.add_argument(
        "--em-iters",
        type=int,
        default=5,
        help="Outer EM iterations (E / M cycles).",
    )
    parser.add_argument(
        "--em-m-steps",
        type=int,
        default=100,
        help="Gradient steps in each M-step.",
    )
    parser.add_argument(
        "--em-lr",
        type=float,
        default=0.05,
        help="M-step learning rate.",
    )
    parser.add_argument(
        "--em-l2",
        type=float,
        default=0.25,
        help="L2 penalty on theta_pre in the M-step.",
    )
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
    parser.add_argument(
        "--postflop-m-steps",
        type=int,
        default=200,
        help="Gradient steps per post-flop M-step.",
    )
    parser.add_argument(
        "--postflop-em-lr",
        type=float,
        default=0.05,
        help="Learning rate for post-flop M-step.",
    )
    parser.add_argument(
        "--postflop-em-l2",
        type=float,
        default=0.25,
        help="L2 penalty on post-flop theta_post.",
    )
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the runner result as JSON.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional filepath to write the JSON result to.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Progress logs go to stderr (default INFO). Use DEBUG for per-pair filter lines.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Shorthand for --log-level WARNING (errors only).",
    )
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
        print(f"EM: outer iters={result.em.outer_iterations}, M-step steps={result.em.m_step_steps}, lr={result.em.m_learning_rate}, l2={result.em.m_l2}")
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
        print(f"Log-likelihood: {hand_result.log_likelihood:.6f}")
        for entry in hand_result.top_range:
            print(f"  {entry['hand_class']}: {entry['probability']:.6f}")

    return 0


def pipeline_main(argv: Optional[List[str]] = None) -> int:
    """Train global priors (50%), fit player θ (25%), evaluate (25%) on disjoint hand sets."""
    import argparse

    from pipeline_common import dump_json, flatten_hands, split_hand_refs

    from find_theta import learn_player_thetas
    from test import evaluate_split
    from train import train_global_priors

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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pipeline":
        raise SystemExit(pipeline_main(sys.argv[2:]))
    raise SystemExit(main())
