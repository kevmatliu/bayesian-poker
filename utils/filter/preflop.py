from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from utils.filter.helpers import (
    FilterStep,
    STRENGTH_BUCKETS,
    effective_sample_size,
    initial_class_prior,
    normalize,
    sample_combo_for_class,
)
from utils.gto_prior import GTOPrior, StateKey
from utils.hand_map import poker_hand_mapper

class PreflopRangeFilter:
    """
    Bayesian filter over the 169 pre-flop hand classes.

    R_t(h) ∝ R_{t-1}(h) * P(a_t | h, s_t, phi)

    phi is the temperature of the range stored inside the GTOPrior.
    phi = 0  → pure GTO.
    phi > 0  → wider than GTO (loose player).
    phi < 0  → tighter than GTO (nit).
    """

    def __init__(
        self,
        observer_name: str,
        target_name: str,
        observer_hole_cards: str = "",
        prior_model: Optional[GTOPrior] = None,
        initial_range: Optional[Dict[str, float]] = None,
    ):
        self.observer_name = observer_name
        self.target_name = target_name
        self.observer_hole_cards = observer_hole_cards
        self.prior_model = prior_model or GTOPrior()
        self.range: Dict[str, float] = normalize(
            initial_range or initial_class_prior(dead_cards=observer_hole_cards)
        )
        self.steps: List[FilterStep] = []

    @property
    def phi(self) -> float:
        return self.prior_model.phi

    @phi.setter
    def phi(self, value: float) -> None:
        self.prior_model.phi = value

    def update(
        self,
        state_key: StateKey | str,
        action_bucket: int,
    ) -> Dict[str, float]:
        """Apply one Bayesian filtering update R_t ∝ R_{t-1} * likelihood."""
        state_key_str = (
            state_key.as_string() if isinstance(state_key, StateKey) else state_key
        )

        unnorm: Dict[str, float] = {
            h: prob * self.prior_model.action_probability(h, state_key, action_bucket)
            for h, prob in self.range.items()
        }

        evidence = sum(unnorm.values())
        if evidence <= 0:
            raise ValueError(
                f"Filtering produced zero evidence at state={state_key_str}, "
                f"action={action_bucket}.  Check the floor in GTOPrior."
            )

        self.range = {h: v / evidence for h, v in unnorm.items()}
        top_class, top_prob = self.top_k(1)[0]
        self.steps.append(FilterStep(
            state_key=state_key_str,
            action_bucket=action_bucket,
            evidence=evidence,
            ess=effective_sample_size(self.range),
            top_class=top_class,
            top_prob=top_prob,
            layer="preflop",
        ))
        return self.range

    def collapse_to_strength(self, board_cards: str) -> Dict[str, float]:
        """Marginalise R_{T_pre} onto the 7 strength buckets given the board.

        Q_0(w) = sum_{h : f(h, c_flop) = w} R_{T_pre}(h)

        Returns a dict keyed by STRENGTH_BUCKETS strings.
        This is the bridge from pre-flop to post-flop filtering.

        Note: only equivalence classes that have at least one concrete
        hole-card combination compatible with the board are mapped.
        Classes with no valid combos are ignored (their probability mass
        gets redistributed via normalisation).
        """
        q: Dict[str, float] = {w: 0.0 for w in STRENGTH_BUCKETS}
        unresolved = 0.0

        for hand_class, prob in self.range.items():
            if prob <= 0:
                continue
            sample = sample_combo_for_class(hand_class, board_cards)
            if sample is None:
                unresolved += prob
                continue
            try:
                result = poker_hand_mapper(sample, board_cards)
                bucket = result["bucket"]
                if bucket in q:
                    q[bucket] += prob
                else:
                    unresolved += prob
            except Exception:
                unresolved += prob

        total = sum(q.values())
        if total <= 0:
            # Fallback: uniform
            return {w: 1.0 / len(STRENGTH_BUCKETS) for w in STRENGTH_BUCKETS}

        # Distribute unresolved mass proportionally
        if unresolved > 0:
            for w in q:
                q[w] += unresolved * (q[w] / total)

        return normalize(q)

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.range.items(), key=lambda x: x[1], reverse=True)[:k]

    def true_class_probability(self, true_hand_class: str) -> float:
        return self.range.get(true_hand_class, 0.0)

    def log_likelihood(self) -> float:
        """Sum of log-evidences accumulated during filtering (cumulative log-loss proxy)."""
        return sum(math.log(step.evidence) for step in self.steps if step.evidence > 0)
