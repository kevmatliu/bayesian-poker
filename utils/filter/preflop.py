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

        R_t(h) propto R_{t-1}(h) * P(a_t | h, s_t)

    ``P`` uses the population baseline from ``prior_model`` and ``beta_preflop``
    then applies the explicit``theta_pre`` passed to this filter.
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
        self.observer_name = observer_name                                          # who is updating beliefs
        self.target_name = target_name                                              # villain whose range is inferred
        self.observer_hole_cards = observer_hole_cards                              # blockers for initial uniform-over-combos prior
        self.prior_model = prior_model or PreflopPrior()                            # population baseline object
        if theta_pre is None:                                                       # default: no deviation from baseline
            self._theta_pre: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        else:
            t = tuple(float(x) for x in theta_pre)                                  # coerce to floats defensively
            if len(t) != 3:                                                         # must match utility vector dimensionality
                raise ValueError("theta_pre must have length 3 (fold, passive, aggression tilts).")
            self._theta_pre = t
        self.range: Dict[str, float] = normalize(                                   # current belief over 169 classes
            initial_range or initial_class_prior(dead_cards=observer_hole_cards)    # user prior or uniform-with-blockers
        )
        self.steps: List[FilterStep] = []                                           # append-only audit trail of updates
        self._action_model = PreflopActionModel(self.prior_model, self._theta_pre)  # likelihood evaluator with tilt

    def update(
        self,
        state_key: StateKey | str,
        action_bucket: int,
    ) -> Dict[str, float]:
        """Apply one Bayesian filtering update R_t ∝ R_{t-1} * likelihood."""
        state_key_str = (
            state_key.as_string() if isinstance(state_key, StateKey) else state_key  # normalize to string for logging
        )

        unnorm: Dict[str, float] = {
            h: prob
            * self._action_model.action_probability(                                 # P(observed action | class h, spot)
                h, state_key, action_bucket,
            )
            for h, prob in self.range.items()                                        # multiply prior mass by likelihood
        }

        evidence = sum(unnorm.values())                                              # marginal likelihood of the observed action
        if evidence <= 0:                                                            # numerical failure or inconsistent model
            raise ValueError(
                f"Filtering produced zero evidence at state={state_key_str}, "
                f"action={action_bucket}.  Check the floor in PreflopPrior."
            )

        self.range = {h: v / evidence for h, v in unnorm.items()}                    # Bayes posterior update
        top_class, top_prob = self.top_k(1)[0]                                       # track MAP for diagnostics
        self.steps.append(FilterStep(                                                # record scalar summaries for notebooks
            state_key=state_key_str,
            action_bucket=action_bucket,
            evidence=evidence,
            ess=effective_sample_size(self.range),                                   # concentration after update
            top_class=top_class,
            top_prob=top_prob,
            layer="preflop",
        ))
        return self.range                                                            # allow chaining in scripts

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.range.items(), key=lambda x: x[1], reverse=True)[:k]     # partial sort by probability mass

    def true_class_probability(self, true_hand_class: str) -> float:
        return self.range.get(true_hand_class, 0.0)  # marginal belief on the realized class (for Brier / log score)

    def log_likelihood(self) -> float:
        """Sum of log-evidences accumulated during filtering (cumulative log-loss proxy)."""
        return sum(math.log(step.evidence) for step in self.steps if step.evidence > 0)  # skip impossible zeros
