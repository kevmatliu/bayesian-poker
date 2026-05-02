"""Pre-flop Bayesian range filtering.

The filter tracks a distribution over 169 pre-flop hand classes.  Each observed
action updates the range by multiplying by the action likelihood from the
GTO-inspired prior:

    R_t(h) proportional to R_{t-1}(h) * phi[h, s_t, a_t].

This is also the E-step building block for a pre-flop-only EM implementation:
given a sequence of observed actions, the final filtered range is the posterior
responsibility vector q(h).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils_jin.cards import all_169_classes, initial_class_prior, normalize_distribution
from .gto_prior import PreflopGTOPrior
from ..utils_jin.state_encoder import StateKey


@dataclass(frozen=True)
class FilterStep:
    """Debug record for one Bayesian update."""

    state_key: str
    action_bucket: int
    evidence: float
    ess: float
    top_class: str
    top_prob: float


def effective_sample_size(distribution: dict[str, float]) -> float:
    """Return ESS = 1 / sum p_i^2 for a categorical distribution."""

    denom = sum(prob * prob for prob in distribution.values())
    if denom <= 0:
        return 0.0
    return 1.0 / denom


class PreflopRangeFilter:
    """Bayesian filter over the 169 pre-flop hand classes."""

    def __init__(
        self,
        prior_model: PreflopGTOPrior | None = None,
        initial_range: dict[str, float] | None = None,
    ):
        self.prior_model = prior_model or PreflopGTOPrior()
        self.range = normalize_distribution(initial_range or initial_class_prior())
        self.steps: list[FilterStep] = []

    def update(self, state_key: StateKey | str, action_bucket: int) -> dict[str, float]:
        """Apply one Bayesian filtering update and return the new range."""

        state_key_str = state_key.as_string() if isinstance(state_key, StateKey) else state_key
        unnormalized: dict[str, float] = {}

        for hand_class, old_prob in self.range.items():
            likelihood = self.prior_model.action_probability(hand_class, state_key, action_bucket)
            unnormalized[hand_class] = old_prob * likelihood

        evidence = sum(unnormalized.values())
        if evidence <= 0:
            # With the positive likelihood floor this should not happen.  If it
            # does, raising is better than silently returning nonsense.
            raise ValueError("Filtering update produced zero evidence.")

        self.range = {hand_class: value / evidence for hand_class, value in unnormalized.items()}
        top_class, top_prob = self.top_k(1)[0]
        self.steps.append(
            FilterStep(
                state_key=state_key_str,
                action_bucket=action_bucket,
                evidence=evidence,
                ess=effective_sample_size(self.range),
                top_class=top_class,
                top_prob=top_prob,
            )
        )
        return self.range

    def run_records(self, records) -> dict[str, float]:
        """Run the filter over pre-flop decision records for one player-hand."""

        for record in records:
            if record.street != "pre-flop":
                continue
            self.update(record.state_key, record.action_bucket)
        return self.range

    def top_k(self, k: int = 10) -> list[tuple[str, float]]:
        """Return the highest-probability hand classes."""

        return sorted(self.range.items(), key=lambda item: item[1], reverse=True)[:k]

    def true_class_probability(self, true_hand_class: str) -> float:
        """Return posterior mass on the known true class, if available."""

        return self.range.get(true_hand_class, 0.0)


def filter_preflop_records(
    records,
    prior_model: PreflopGTOPrior | None = None,
    initial_range: dict[str, float] | None = None,
) -> PreflopRangeFilter:
    """Convenience function for filtering one player-hand record sequence."""

    filt = PreflopRangeFilter(prior_model=prior_model, initial_range=initial_range)
    filt.run_records(records)
    return filt


def uniform_class_distribution() -> dict[str, float]:
    """Return a uniform distribution over 169 classes for sensitivity checks."""

    classes = all_169_classes()
    prob = 1.0 / len(classes)
    return {hand_class: prob for hand_class in classes}

