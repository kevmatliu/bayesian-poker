"""Preflop hand-class posterior metrics (E-step) for range evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from runners.common import (
    flatten_hands,
    hole_cards_to_hand_class,
    preflop_decisions_for_hand,
    read_session_names_file,
)
from utils.action.preflop import PreflopActionModel, PreflopPrior
from utils.em.preflop import PreflopEMDecision, PreflopEMHandBundle, e_step_hand_class_posterior
from utils.filter.common import initial_class_prior, normalize
from utils.strength.preflop import all_169_classes
from utils.eval.repo_paths import resolve_session_filter_path, resolve_session_theta_path

EPS = 1e-12  # floor class prob inside log for numerical stability


def em_and_online_hand_refs(repo: Path, *, pluribus_root: Path | None = None):
    pluribus_root = pluribus_root or (repo / "pluribus")                   # default data root beside repo
    em_s = read_session_names_file(resolve_session_theta_path(repo))       # sessions backing em thetas
    online_s = read_session_names_file(resolve_session_filter_path(repo))  # online/filter eval sessions
    em_refs = flatten_hands([pluribus_root / s for s in em_s])             # hand refs for em split
    online_refs = flatten_hands([pluribus_root / s for s in online_s])     # hand refs for online split
    return em_refs, online_refs                                            # tuple consumed by posterior evaluators


def build_bundles_with_truth(
    refs,
    target: str,
    observer: str,
) -> list[tuple[PreflopEMHandBundle, str, dict]]:
    rows = []                                                                             # collect (bundle, true_class, meta) tuples
    for ref in refs:                                                                      # scan each hand reference
        names = ref.hand.player_names                                                     # roster for membership checks
        if target not in names or observer not in names:                                  # skip hands missing required seats
            continue                                                                      # cannot define observer-informed range without both
        true_class = hole_cards_to_hand_class(ref.hand.hole_cards.get(target, "") or "")  # ground-truth 169 label
        if true_class is None:                                                            # unknown or unparsable hole cards
            continue                                                                      # cannot score nll without truth
        decisions = preflop_decisions_for_hand(ref.hand, target, ref.global_index)        # observed action codes
        if not decisions:                                                                 # no preflop decisions recorded
            continue                                                                      # nothing to condition the posterior on
        dead = ref.hand.hole_cards.get(observer, "") or ""                                # observer cards block some combos
        initial_range = normalize(initial_class_prior(dead_cards=dead))                   # bayesian starting prior over classes
        bundle = PreflopEMHandBundle(
            tuple(PreflopEMDecision(d.state_key, d.action_bucket) for d in decisions),    # freeze decision sequence
            initial_range,                                                                # attach initial dirichlet/prior point as starting q support
        )
        rows.append(
            (
                bundle,                                                                   # hand-level em input
                true_class,                                                               # string label for evaluation
                {"global_index": ref.global_index, "n_decisions": len(decisions)},        # light debug metadata
            )
        )
    return rows                                                                           # list ready for e-step or prior-only baselines


def run_estep(rows, prior):
    return [
        (true_class, e_step_hand_class_posterior(bundle, prior))  # pair truth with posterior dict after e-step
        for bundle, true_class, _meta in rows                     # ignore meta in pure posterior pass
    ]


def prior_only_results(rows):
    return [(true_class, dict(bundle.initial_range)) for bundle, true_class, _meta in rows]  # baseline without actions


def compute_all_posterior_results(
    players: list[str],
    observer_for_target: dict[str, str],
    *,
    beta_pre: np.ndarray,
    theta_pre: dict[str, tuple],
    em_refs,
    online_refs,
) -> dict[tuple[str, str, str], list[tuple[str, dict]]]:
    """Keys ``(player, split, predictor)`` with split ``em|online``, predictor three-way."""
    data: dict[tuple[str, str], list] = {}                                                  # cache built rows per (player, split)
    for player in players:                                                                  # loop eval subjects
        observer = observer_for_target[player]                                              # seat providing dead cards for filtering
        data[(player, "em")] = build_bundles_with_truth(em_refs, player, observer)          # em-session bundles
        data[(player, "online")] = build_bundles_with_truth(online_refs, player, observer)  # online-session bundles

    results: dict[tuple[str, str, str], list[tuple[str, dict]]] = {}                        # output container keyed by triple
    for player in players:                                                                  # evaluate each player with their models
        prior_pop = PreflopActionModel(
            PreflopPrior(beta_preflop=beta_pre), (0.0, 0.0, 0.0)                            # population logits: global betas only
        )
        prior_pl = PreflopActionModel(
            PreflopPrior(beta_preflop=beta_pre), tuple(theta_pre[player])                   # player-specific offset thetas
        )
        for split in ("em", "online"):                                                      # compare splits with identical machinery
            rows = data[(player, split)]                                                    # fetch prebuilt bundles for this player/split
            results[(player, split, "prior_only")] = prior_only_results(rows)               # ignore actions: initial range only
            results[(player, split, "population")] = run_estep(rows, prior_pop)             # e-step under population prior
            results[(player, split, "player")] = run_estep(rows, prior_pl)                  # e-step under personalized prior
    return results                                                                          # nested posterior collections for metrics layer


def true_hand_nll(results: list[tuple[str, dict]]) -> float:
    if not results:                        # empty input list
        return float("nan")                # undefined average nll
    nlls = []                              # accumulate per-hand nll contributions
    for true_class, q in results:          # iterate posterior dicts with known truth
        p = float(q.get(true_class, 0.0))  # posterior prob assigned to realized class
        nlls.append(-np.log(max(p, EPS)))  # clamp inside log to avoid -inf
    return float(np.mean(nlls))            # report mean nll over hands


def ranks_from_results(
    results: list[tuple[str, dict]], *, all_169: list[str], index_of: dict[str, int]
) -> np.ndarray:
    out = []                                                           # collect integer ranks (1-based)
    for true_class, q in results:                                      # each hand defines a ranking task
        p = np.array([q.get(h, 0.0) for h in all_169], dtype=float)    # dense prob vector in canonical class order
        order = np.argsort(-p, kind="stable")                          # descending sort indices with stable tie break
        rank = int(np.where(order == index_of[true_class])[0][0]) + 1  # convert zero-based position to 1-based rank
        out.append(rank)                                               # store rank for this hand
    return np.asarray(out, dtype=int)                                  # vector for distribution plots / summaries


def default_observer_map(players: list[str]) -> dict[str, str]:
    out = {}                                      # map target player -> observer seat name
    for i, p in enumerate(players):               # fixed rotation over declared player order
        out[p] = players[(i + 1) % len(players)]  # each player observes the next seat cyclically
    return out                                    # simple default pairing for dead-card modeling
