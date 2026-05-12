"""Preflop Bayesian filter over 169 strategic hand classes."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from utils.filter.common import FilterStep, effective_sample_size, initial_class_prior, normalize
from utils.action.preflop import PreflopActionModel, PreflopPrior, StateKey


class PreflopRangeFilter:
    """
    Bayesian filter over the 169 pre-flop hand classes.

    At each observed action::

        R_t(h) ∝ R_{t-1}(h) * P(a_t | h, s_t)

    ``P`` uses the population baseline from ``prior_model``'s ``beta_preflop``
    (via :meth:`PreflopPrior.baseline_action_probs`), then applies the **explicit**
    ``theta_pre`` passed to this filter (same tilt as :class:`PreflopActionModel`).
    ``prior_model`` should be a :class:`PreflopPrior` (baseline only); tendency
    enters only through ``theta_pre`` here.
    """

    def __init__(
        self,
        observer_name: str,
        target_name: str,
        observer_hole_cards: str = "",
        prior_model: Optional[PreflopPrior] = None,
        *,
        theta_pre: Sequence[float] | None = None,
        initial_range: Optional[Dict[str, float]] = None,
    ):
        self.observer_name = observer_name
        self.target_name = target_name
        self.observer_hole_cards = observer_hole_cards
        self.prior_model = prior_model or PreflopPrior()
        if theta_pre is None:
            self._theta_pre: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        else:
            t = tuple(float(x) for x in theta_pre)
            if len(t) != 3:
                raise ValueError("theta_pre must have length 3 (fold, passive, aggression tilts).")
            self._theta_pre = t
        self.range: Dict[str, float] = normalize(
            initial_range or initial_class_prior(dead_cards=observer_hole_cards)
        )
        self.steps: List[FilterStep] = []
        self._action_model = PreflopActionModel(self.prior_model, self._theta_pre)

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
            h: prob
            * self._action_model.action_probability(
                h, state_key, action_bucket,
            )
            for h, prob in self.range.items()
        }

        evidence = sum(unnorm.values())
        if evidence <= 0:
            raise ValueError(
                f"Filtering produced zero evidence at state={state_key_str}, "
                f"action={action_bucket}.  Check the floor in PreflopPrior."
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

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.range.items(), key=lambda x: x[1], reverse=True)[:k]

    def true_class_probability(self, true_hand_class: str) -> float:
        return self.range.get(true_hand_class, 0.0)

    def log_likelihood(self) -> float:
        """Sum of log-evidences accumulated during filtering (cumulative log-loss proxy)."""
        return sum(math.log(step.evidence) for step in self.steps if step.evidence > 0)
