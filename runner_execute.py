"""Preflop/postflop filtering and pair-EM orchestration (imported after ``runner.py`` sets ``sys.path``)."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.em import PreflopEMDecision, PreflopEMHandBundle, run_postflop_theta_em, run_preflop_em
from utils.filter import ComboRangeFilter, PreflopRangeFilter, all_combo_keys
from utils.filter.helpers import initial_class_prior, normalize
from utils.prior.postflop import PostflopPrior
from utils.prior.preflop import PreflopPrior, state_key_from_parse_state
from utils.parse import Hand, Session
from utils.postflop_runner_bridge import (
    collect_session_postflop_bundles_by_pair,
    combo_features_for_state,
    postflop_target_decisions_for_hand,
    raw_action_bucket_to_postflop,
)

from runner_models import (
    EMPostflopResult,
    EMPostflopRunConfig,
    EMPreflopResult,
    EMPreflopRunConfig,
    HandFilterResult,
    PostflopDecision,
    PreflopDecision,
    PREFLOP_PRIOR_FLOOR,
    RunnerResult,
)

LOG = logging.getLogger("runner")

__all__ = [
    "LOG",
    "PREFLOP_PRIOR_FLOOR",
    "all_combo_keys",
    "dump_result_json",
    "run_hand_preflop_filter",
    "run_preflop_filter",
    "run_session_preflop_filter",
    "_run_preflop_filter_for_hand",
]


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
    hand: Hand,
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
                    tuple(PreflopEMDecision(d.state_key, d.action_bucket) for d in decisions),
                    initial_range,
                )
                groups[(observer, target)].append(bundle)
    return groups


def _run_preflop_em_per_pair(
    groups: Dict[Tuple[str, str], List[PreflopEMHandBundle]],
    em_cfg: EMPreflopRunConfig,
) -> Dict[str, List[float]]:
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
    groups: Dict[Tuple[str, str], List[Any]],
    pf_cfg: EMPostflopRunConfig,
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    theta_out: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    nonempty = [(pair, seq) for pair, seq in groups.items() if seq]
    total_pairs = len(nonempty)
    for idx, ((observer, target), bundles_by_hand) in enumerate(nonempty, start=1):
        pair_label = f"{observer}|{target}"
        n_hands = len(bundles_by_hand)
        LOG.info(
            "Post-flop EM pair %d/%d: %s (%d hands with target postflop actions)",
            idx,
            total_pairs,
            pair_label,
            n_hands,
        )
        theta, _ = run_postflop_theta_em(
            bundles_by_hand,
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

    LOG.info("Post-flop EM: collecting combo-prior bundles (observer dead cards + board)")
    groups = collect_session_postflop_bundles_by_pair(list(hands), resolved_observers, resolved_targets)
    n_bundles = sum(len(v) for v in groups.values())
    LOG.info("Post-flop EM: %d hand bundles (pre pair-breakdown)", n_bundles)

    if n_bundles == 0:
        LOG.warning(
            "Post-flop EM: no usable hands (need target postflop actions and valid combo prior)."
        )
        return EMPostflopResult(
            enabled=True,
            note="No post-flop EM data: no target postflop actions or combo prior could not be built.",
        )

    theta_pf, counts_pf = _run_postflop_em_per_pair(groups, pf_cfg)
    for obs in resolved_observers:
        for tgt in resolved_targets:
            counts_pf.setdefault(f"{obs}|{tgt}", 0)

    return EMPostflopResult(
        enabled=True,
        theta_post_by_pair=theta_pf,
        outer_iterations=pf_cfg.outer_iters,
        m_step_steps=pf_cfg.m_steps,
        m_learning_rate=pf_cfg.m_lr,
        m_l2=pf_cfg.m_l2,
        hands_with_target_cards_per_pair=counts_pf,
        note="Learned from hands with target postflop actions; latent combo prior matches preflop EM (initial_class_prior → 1,326).",
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


def _learned_postflop_prior_for_pair(
    observer: str,
    target: str,
    theta_by_pair: Dict[str, List[float]],
    floor: float,
) -> Optional[PostflopPrior]:
    key = f"{observer}|{target}"
    if key not in theta_by_pair:
        return None
    return PostflopPrior(theta_post=tuple(theta_by_pair[key]), floor=floor)


def _fmt_filter_tag(filter_tag: str) -> str:
    return f"[{filter_tag}] " if filter_tag else ""


def _run_postflop_combo_filter_for_hand(
    hand: Hand,
    observer: str,
    target: str,
    hand_index: int,
    preflop_range: Dict[str, float],
    learned_postflop_prior: Optional[PostflopPrior],
    postflop_floor: float,
    top_k: int,
    street_end_snapshots: Optional[List[Tuple[str, Dict[str, float]]]] = None,
    *,
    filter_verbose: bool = False,
    filter_tag: str = "",
) -> Tuple[List[PostflopDecision], List[Dict[str, float]], Dict[str, float], float]:
    tag = _fmt_filter_tag(filter_tag)
    target_actions = postflop_target_decisions_for_hand(hand, target)
    if not target_actions:
        if filter_verbose:
            LOG.info("%sFilter postflop skip | %s→%s | no target postflop actions", tag, observer, target)
        return [], [], {}, 0.0

    if filter_verbose:
        LOG.info(
            "%sFilter postflop start | hand_index=%d | %s observes %s | target_action_count=%d",
            tag,
            hand_index,
            observer,
            target,
            len(target_actions),
        )

    observer_hole = hand.hole_cards.get(observer, "") or ""
    prior_model = learned_postflop_prior or PostflopPrior(floor=postflop_floor)
    combo_filter = ComboRangeFilter(
        observer_name=observer,
        target_name=target,
        observer_hole_cards=observer_hole,
        prior_model=prior_model,
    )

    decisions: List[PostflopDecision] = []
    strength_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
    last_board: Optional[str] = None
    initialized = False
    prev_street_after_update: Optional[str] = None

    for street, action_index, raw_action in target_actions:
        if (
            street_end_snapshots is not None
            and initialized
            and combo_filter.combos
            and prev_street_after_update is not None
            and street != prev_street_after_update
        ):
            street_end_snapshots.append((prev_street_after_update, dict(combo_filter.combos)))
        states = hand.states.get(street, [])
        if action_index >= len(states):
            continue
        state = states[action_index]
        board = state.community_cards or ""
        if len(board) < 6:
            continue

        if not initialized or board != last_board:
            try:
                if not initialized:
                    combo_filter.explode_from_preflop(preflop_range, board=board)
                    initialized = True
                else:
                    combo_filter.set_board(board)
            except ValueError as exc:
                LOG.warning(
                    "Combo filter init/board update failed at hand=%d %s→%s street=%s: %s",
                    hand_index,
                    observer,
                    target,
                    street,
                    exc,
                )
                return decisions, [], combo_filter.combos, combo_filter.log_likelihood()
            last_board = board
            strength_cache.clear()
            if filter_verbose:
                LOG.info(
                    "%sPostflop range init/board | street=%s | board_cards=%d | n_combos=%d",
                    tag,
                    street,
                    len(board) // 2,
                    len(combo_filter.combos),
                )

        _, feats = combo_features_for_state(
            state,
            target,
            street,
            combo_filter.combos.keys(),
            strength_cache=strength_cache,
        )
        if not feats:
            continue

        raw_bucket = int(raw_action[1][0])
        post_action = raw_action_bucket_to_postflop(raw_bucket)
        amount = int(raw_action[2])

        try:
            combo_filter.update(post_action, feats, state_key=f"{street}|{action_index}")
        except ValueError as exc:
            LOG.warning(
                "Combo filter zero-evidence at hand=%d %s→%s street=%s action=%s: %s",
                hand_index,
                observer,
                target,
                street,
                post_action,
                exc,
            )
            break

        if combo_filter.steps:
            st = combo_filter.steps[-1]
            msg = (
                f"postflop Bayes update | {street} | action_index={action_index} | "
                f"raw_bucket={raw_bucket} post_action={post_action} amt={amount} | "
                f"evidence={st.evidence:.6f} | top={st.top_class} p={st.top_prob:.4f} | ess={st.ess:.2f}"
            )
            if filter_verbose:
                LOG.info("%s%s", tag, msg)
            else:
                LOG.debug("%s%s", tag, msg)

        decisions.append(
            PostflopDecision(
                hand_index=hand_index,
                observer=observer,
                target=target,
                street=street,
                action_index=action_index,
                raw_action_bucket=raw_bucket,
                postflop_action=post_action,
                amount=amount,
            )
        )
        prev_street_after_update = street

    if (
        street_end_snapshots is not None
        and initialized
        and combo_filter.combos
        and prev_street_after_update is not None
    ):
        street_end_snapshots.append((prev_street_after_update, dict(combo_filter.combos)))

    if combo_filter.combos:
        hand.set_combo_range_vector(observer, target, combo_filter.combos)

    top_combos = [
        {"combo": combo, "probability": prob}
        for combo, prob in combo_filter.top_k(top_k)
    ]
    if filter_verbose and decisions:
        LOG.info(
            "%sFilter postflop done | updates=%d | log_L=%.6f | top_combos_preview=%s",
            tag,
            len(decisions),
            combo_filter.log_likelihood(),
            top_combos[: min(3, len(top_combos))],
        )
    return decisions, top_combos, combo_filter.class_marginal(), combo_filter.log_likelihood()


def _run_preflop_filter_for_hand(
    hand: Hand,
    observer: str,
    target: str,
    hand_index: int,
    phi: float,
    top_k: int,
    learned_prior: Optional[PreflopPrior] = None,
    learned_postflop_prior: Optional[PostflopPrior] = None,
    postflop_floor: float = 1e-6,
    street_end_snapshots: Optional[List[Tuple[str, Dict[str, float]]]] = None,
    *,
    filter_verbose: bool = False,
    filter_tag: str = "",
) -> Optional[HandFilterResult]:
    tag = _fmt_filter_tag(filter_tag)
    if observer == target:
        return None

    decisions = _preflop_decisions_for_hand(hand, observer, target, hand_index)
    if not decisions:
        if filter_verbose:
            LOG.info("%sFilter skip | %s→%s | no target preflop decisions", tag, observer, target)
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

    if filter_verbose:
        LOG.info(
            "%sFilter preflop start | hand_index=%d | %s observes %s | target_preflop_decisions=%d | "
            "observer_holes_known=%s | learned_preflop_prior=%s",
            tag,
            hand_index,
            observer,
            target,
            len(decisions),
            bool(observer_hole_cards),
            learned_prior is not None,
        )

    for i, decision in enumerate(decisions):
        preflop_filter.update(decision.state_key, decision.action_bucket)
        step = preflop_filter.steps[-1]
        msg = (
            f"preflop Bayes update {i + 1}/{len(decisions)} | state={step.state_key} | "
            f"bucket={decision.action_bucket} amt={decision.amount} | evidence={step.evidence:.6f} | "
            f"top_class={step.top_class} p={step.top_prob:.4f} | ess={step.ess:.2f}"
        )
        if filter_verbose:
            LOG.info("%s%s", tag, msg)
        else:
            LOG.debug("%s%s", tag, msg)

    if filter_verbose:
        LOG.info(
            "%sFilter preflop done | log_L=%.6f | top_classes=%s",
            tag,
            preflop_filter.log_likelihood(),
            preflop_filter.top_k(min(5, top_k)),
        )

    hand.set_hand_range_vector(observer, target, preflop_filter.range)

    top_range = [
        {"hand_class": hand_class, "probability": prob}
        for hand_class, prob in preflop_filter.top_k(top_k)
    ]

    postflop_decisions, top_combos, combo_marginal, postflop_log_likelihood = (
        _run_postflop_combo_filter_for_hand(
            hand=hand,
            observer=observer,
            target=target,
            hand_index=hand_index,
            preflop_range=preflop_filter.range,
            learned_postflop_prior=learned_postflop_prior,
            postflop_floor=postflop_floor,
            top_k=top_k,
            street_end_snapshots=street_end_snapshots,
            filter_verbose=filter_verbose,
            filter_tag=filter_tag,
        )
    )

    if filter_verbose:
        LOG.info(
            "%sFilter hand complete | %s→%s | preflop_log_L=%.6f postflop_log_L=%.6f | "
            "postflop_decisions=%d",
            tag,
            observer,
            target,
            preflop_filter.log_likelihood(),
            postflop_log_likelihood,
            len(postflop_decisions),
        )

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
        postflop_decisions=postflop_decisions,
        top_combos=top_combos,
        final_combo_marginal=combo_marginal,
        postflop_log_likelihood=postflop_log_likelihood,
    )


def _maybe_preflop_em(
    hands: Sequence[Hand],
    resolved_observers: List[str],
    resolved_targets: List[str],
    em_cfg: Optional[EMPreflopRunConfig],
) -> Tuple[Dict[str, List[float]], EMPreflopResult]:
    theta_by_pair: Dict[str, List[float]] = {}
    em_result = EMPreflopResult()
    if not (em_cfg and em_cfg.enabled):
        return theta_by_pair, em_result

    LOG.info("EM: collecting preflop bundles from %d hand(s)", len(hands))
    groups = _collect_grouped_em_bundles(hands, resolved_observers, resolved_targets)
    n_bundles = sum(len(bs) for bs in groups.values())
    groups_with_data = sum(1 for bs in groups.values() if bs)
    LOG.info("EM: %d (observer, target) groups with data, %d total bundles", groups_with_data, n_bundles)

    if not any(groups.values()):
        note = (
            "EM requested but no target pre-flop actions were found for this hand."
            if len(hands) == 1
            else "EM requested but no target pre-flop actions were found in the session."
        )
        LOG.warning(note)
        return theta_by_pair, EMPreflopResult(enabled=True, note=note)

    theta_by_pair = _run_preflop_em_per_pair(groups, em_cfg)
    return theta_by_pair, EMPreflopResult(
        enabled=True,
        theta_pre_by_pair=theta_by_pair,
        outer_iterations=em_cfg.outer_iters,
        m_step_steps=em_cfg.m_steps,
        m_learning_rate=em_cfg.m_lr,
        m_l2=em_cfg.m_l2,
        note="theta_pre replaces --phi for the action prior when EM is enabled.",
    )


def _filter_hands_observer_target_grid(
    hands: Sequence[Hand],
    resolved_observers: List[str],
    resolved_targets: List[str],
    *,
    theta_by_pair: Dict[str, List[float]],
    pf_result: EMPostflopResult,
    postflop_prior_floor: float,
    phi: float,
    top_k: int,
    hand_indices: Optional[Sequence[int]] = None,
    multi_hand_log: bool = True,
) -> List[HandFilterResult]:
    hand_results: List[HandFilterResult] = []
    nh = len(hands)
    for i, hand in enumerate(hands):
        hi = hand_indices[i] if hand_indices is not None else i
        if multi_hand_log and nh > 1:
            LOG.info("Preflop+combo filter hand %d/%d (phh index in session)", i + 1, nh)
        for observer in resolved_observers:
            for target in resolved_targets:
                learned = _learned_prior_for_pair(observer, target, theta_by_pair)
                learned_post = _learned_postflop_prior_for_pair(
                    observer,
                    target,
                    pf_result.theta_post_by_pair,
                    postflop_prior_floor,
                )
                result = _run_preflop_filter_for_hand(
                    hand=hand,
                    observer=observer,
                    target=target,
                    hand_index=hi,
                    phi=phi,
                    top_k=top_k,
                    learned_prior=learned,
                    learned_postflop_prior=learned_post,
                    postflop_floor=postflop_prior_floor,
                )
                if result is not None:
                    hand_results.append(result)
                    LOG.debug(
                        "Hand %s: %s→%s (%d preflop, %d postflop, log_L_pre=%.4f log_L_post=%.4f)",
                        hi,
                        observer,
                        target,
                        len(result.decisions),
                        len(result.postflop_decisions),
                        result.log_likelihood,
                        result.postflop_log_likelihood,
                    )
    return hand_results


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

    theta_by_pair, em_result = _maybe_preflop_em(
        [hand], resolved_observers, resolved_targets, em_cfg
    )
    pf_result = _compute_postflop_em_result([hand], resolved_observers, resolved_targets, postflop_em_cfg)
    postflop_prior_floor = postflop_em_cfg.prior_floor if postflop_em_cfg is not None else 1e-6

    pair_jobs = len(resolved_observers) * len(resolved_targets)
    LOG.info("Preflop filter single hand (%s): %d observer×target jobs", source_label, pair_jobs)

    hand_results = _filter_hands_observer_target_grid(
        [hand],
        resolved_observers,
        resolved_targets,
        theta_by_pair=theta_by_pair,
        pf_result=pf_result,
        postflop_prior_floor=postflop_prior_floor,
        phi=phi,
        top_k=top_k,
        hand_indices=[hand_index],
        multi_hand_log=False,
    )
    LOG.info(
        "Preflop+combo filter done (1 hand): %d results from %d jobs",
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

    theta_by_pair, em_result = _maybe_preflop_em(
        session.hands, resolved_observers, resolved_targets, em_cfg
    )
    pf_result = _compute_postflop_em_result(
        session.hands, resolved_observers, resolved_targets, postflop_em_cfg
    )
    postflop_prior_floor = postflop_em_cfg.prior_floor if postflop_em_cfg is not None else 1e-6

    total_hands = len(session.hands)
    pair_jobs_per_hand = len(resolved_observers) * len(resolved_targets)
    LOG.info(
        "Preflop+combo filtering %d hands × %d observer×target jobs (max %d filter runs)",
        total_hands,
        pair_jobs_per_hand,
        total_hands * pair_jobs_per_hand,
    )

    hand_results = _filter_hands_observer_target_grid(
        session.hands,
        resolved_observers,
        resolved_targets,
        theta_by_pair=theta_by_pair,
        pf_result=pf_result,
        postflop_prior_floor=postflop_prior_floor,
        phi=phi,
        top_k=top_k,
        multi_hand_log=True,
    )
    LOG.info("Preflop+combo filtering complete: %d hand results collected", len(hand_results))

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
    LOG.info(
        "Run start: %s (%s)",
        source,
        "session directory" if source.is_dir() else "single hand file",
    )
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
