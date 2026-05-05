from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
UTILS_DIR = REPO_ROOT / "utils"

for path in (str(REPO_ROOT), str(UTILS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.filter import PreflopRangeFilter
from utils.gto_prior import state_key_from_parse_state
from utils.parse import Hand, Session


@dataclass(frozen=True)
class EMPlaceholderConfig:
    enabled: bool = False
    note: str = "Placeholder only. EM is not wired into this runner yet."


@dataclass(frozen=True)
class PostflopPlaceholderConfig:
    enabled: bool = False
    note: str = "Placeholder only. Post-flop modeling is not wired into this runner yet."
    board_model: Optional[str] = None
    strength_bucket_model: Optional[str] = None
    transition_model: Optional[str] = None


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
    em: EMPlaceholderConfig
    postflop: PostflopPlaceholderConfig


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


def _run_preflop_filter_for_hand(
    hand,
    observer: str,
    target: str,
    hand_index: int,
    phi: float,
    top_k: int,
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
    )
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
    em: Optional[EMPlaceholderConfig] = None,
    postflop: Optional[PostflopPlaceholderConfig] = None,
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

    hand_results: List[HandFilterResult] = []
    for observer in resolved_observers:
        for target in resolved_targets:
            result = _run_preflop_filter_for_hand(
                hand=hand,
                observer=observer,
                target=target,
                hand_index=hand_index,
                phi=phi,
                top_k=top_k,
            )
            if result is not None:
                hand_results.append(result)

    return RunnerResult(
        session_path=source_label,
        observers=resolved_observers,
        targets=resolved_targets,
        phi=phi,
        hand_results=hand_results,
        em=em or EMPlaceholderConfig(),
        postflop=postflop or PostflopPlaceholderConfig(),
    )


def run_session_preflop_filter(
    session_path: str | Path,
    observers: Optional[Iterable[str]] = None,
    targets: Optional[Iterable[str]] = None,
    phi: float = 0.0,
    top_k: int = 10,
    em: Optional[EMPlaceholderConfig] = None,
    postflop: Optional[PostflopPlaceholderConfig] = None,
) -> RunnerResult:
    session = Session(session_path)
    session.parse()

    resolved_observers = _resolve_requested_players(session, observers)
    resolved_targets = _resolve_requested_players(session, targets)
    hand_results: List[HandFilterResult] = []

    for hand_index, hand in enumerate(session.hands):
        for observer in resolved_observers:
            for target in resolved_targets:
                result = _run_preflop_filter_for_hand(
                    hand=hand,
                    observer=observer,
                    target=target,
                    hand_index=hand_index,
                    phi=phi,
                    top_k=top_k,
                )
                if result is not None:
                    hand_results.append(result)

    return RunnerResult(
        session_path=str(Path(session_path).expanduser().resolve()),
        observers=resolved_observers,
        targets=resolved_targets,
        phi=phi,
        hand_results=hand_results,
        em=em or EMPlaceholderConfig(),
        postflop=postflop or PostflopPlaceholderConfig(),
    )


def run_preflop_filter(
    source_path: str | Path,
    observers: Optional[Iterable[str]] = None,
    targets: Optional[Iterable[str]] = None,
    phi: float = 0.0,
    top_k: int = 10,
    em: Optional[EMPlaceholderConfig] = None,
    postflop: Optional[PostflopPlaceholderConfig] = None,
) -> RunnerResult:
    source = Path(source_path).expanduser().resolve()
    if source.is_dir():
        return run_session_preflop_filter(
            session_path=source,
            observers=observers,
            targets=targets,
            phi=phi,
            top_k=top_k,
            em=em,
            postflop=postflop,
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
            em=em,
            postflop=postflop,
        )
    raise FileNotFoundError(f"Source path does not exist: {source}")


def dump_result_json(result: RunnerResult, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return output


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a session directory or a single hand file and run preflop range filtering only."
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
        help="Fixed preflop temperature parameter. EM learning is not wired yet.",
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
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = run_preflop_filter(
        source_path=args.session_path,
        observers=args.observers,
        targets=args.targets,
        phi=args.phi,
        top_k=args.top_k,
    )

    if args.json_out:
        dump_result_json(result, args.json_out)

    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return 0

    print(f"Session: {result.session_path}")
    print(f"Observers: {', '.join(result.observers) if result.observers else '(none)'}")
    print(f"Targets: {', '.join(result.targets) if result.targets else '(none)'}")
    print(f"Fixed phi: {result.phi}")
    print(f"EM: {result.em.note}")
    print(f"Post-flop: {result.postflop.note}")

    for hand_result in result.hand_results:
        print()
        print(f"Hand {hand_result.hand_index} | {hand_result.observer} -> {hand_result.target}")
        print(f"Observer hole cards: {hand_result.observer_hole_cards or '(unknown)'}")
        print(f"Preflop decisions: {len(hand_result.decisions)}")
        print(f"Log-likelihood: {hand_result.log_likelihood:.6f}")
        for entry in hand_result.top_range:
            print(f"  {entry['hand_class']}: {entry['probability']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
