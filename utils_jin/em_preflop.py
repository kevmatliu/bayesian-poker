"""Starter utilities for pre-flop EM.

This is not a full experiment runner.  It provides the core pieces needed once
the project is ready to run EM:

1. E-step: filter each player-hand to compute responsibilities q_m(h).
2. Soft counts: turn those responsibilities into N[h, s, a].
3. M-step helper: convert soft counts plus GTO pseudo-counts into likelihoods.

These functions are deliberately simple and dictionary-based so the math is easy
to inspect before optimizing.
"""

from __future__ import annotations

from collections import defaultdict
from math import log
from typing import Iterable

from .action_encoder import ACTION_LABELS
from .cards import all_169_classes
from ..utils.gto_prior import ACTION_BUCKETS, PreflopGTOPrior
from .preflop_filter import filter_preflop_records
from .records import DecisionRecord, group_records_by_player_hand, preflop_only

GroupKey = tuple[str, str, str]
CountKey = tuple[str, str, int]


def e_step_preflop(
    records: Iterable[DecisionRecord],
    prior_model: PreflopGTOPrior | None = None,
) -> dict[GroupKey, dict[str, float]]:
    """Compute q_m(h) for each player-hand group using pre-flop actions."""

    prior_model = prior_model or PreflopGTOPrior()
    grouped = group_records_by_player_hand(preflop_only(records))

    responsibilities: dict[GroupKey, dict[str, float]] = {}
    for key, group_records in grouped.items():
        filt = filter_preflop_records(group_records, prior_model=prior_model)
        responsibilities[key] = dict(filt.range)
    return responsibilities


def expected_action_counts(
    records: Iterable[DecisionRecord],
    responsibilities: dict[GroupKey, dict[str, float]],
) -> dict[CountKey, float]:
    """Compute soft counts N[h, s, a] from E-step responsibilities.

    For each observed action by player-hand m, every hand class h contributes
    q_m(h) fractional count to the observed (state, action).
    """

    counts: dict[CountKey, float] = defaultdict(float)
    for record in preflop_only(records):
        group_key = (record.session_id, record.hand_id, record.player_id)
        q = responsibilities.get(group_key)
        if q is None:
            continue

        for hand_class, prob in q.items():
            counts[(hand_class, record.state_key_str, record.action_bucket)] += prob
    return dict(counts)


def m_step_likelihoods(
    counts: dict[CountKey, float],
    prior_model: PreflopGTOPrior | None = None,
    kappa: float = 5.0,
    smoothing: float = 0.1,
) -> dict[tuple[str, str], dict[int, float]]:
    """Convert soft counts into phi[h, s][a] probabilities.

    The update is MAP-style:

        phi[h, s, a] proportional to
        soft_count[h, s, a] + smoothing + kappa * p_GTO(a | h, s).

    ``kappa`` controls how strongly the GTO-inspired prior affects the M-step.
    """

    prior_model = prior_model or PreflopGTOPrior()
    states = sorted({state_key for _hand_class, state_key, _action in counts})
    likelihoods: dict[tuple[str, str], dict[int, float]] = {}

    for hand_class in all_169_classes():
        for state_key in states:
            numerators: dict[int, float] = {}
            for action in ACTION_BUCKETS:
                prior_alpha = prior_model.dirichlet_alpha(hand_class, state_key, kappa)[action]
                observed = counts.get((hand_class, state_key, action), 0.0)
                numerators[action] = observed + smoothing + prior_alpha

            denom = sum(numerators.values())
            likelihoods[(hand_class, state_key)] = {
                action: value / denom for action, value in numerators.items()
            }

    return likelihoods


def true_class_log_loss(
    records: Iterable[DecisionRecord],
    responsibilities: dict[GroupKey, dict[str, float]],
    eps: float = 1e-12,
) -> float:
    """Average -log posterior mass on the true pre-flop class.

    This is only available for data like Pluribus where hole cards are known.
    """

    grouped = group_records_by_player_hand(preflop_only(records))
    losses: list[float] = []

    for key, group in grouped.items():
        true_class = next((record.hand_class for record in group if record.hand_class), None)
        if true_class is None:
            continue
        prob = responsibilities.get(key, {}).get(true_class, 0.0)
        losses.append(-log(max(prob, eps)))

    if not losses:
        raise ValueError("No known true hand classes available for log-loss.")
    return sum(losses) / len(losses)


def summarize_counts(counts: dict[CountKey, float], top_n: int = 10) -> list[tuple[CountKey, float]]:
    """Return largest soft counts for quick debugging."""

    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]


def action_label(action_bucket: int) -> str:
    """Readable action label for count tables."""

    return ACTION_LABELS[action_bucket]

