from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from utils.filter.helpers import (
    FilterStep,
    STRENGTH_BUCKETS,
    STRENGTH_INDEX,
    effective_sample_size,
    normalize,
)
from utils.gto_prior import ACTION_BUCKETS, CBR_LARGE, CBR_MEDIUM, CBR_SMALL, CHECK_CALL, FOLD, StateKey


def _build_postflop_psi() -> Dict[str, Dict[int, float]]:
    """Build the base heuristic psi table: P(action | strength bucket)."""
    raw = {
        "nuts/near-nuts": {FOLD: -3.0, CHECK_CALL: 0.5, CBR_SMALL: 1.2, CBR_MEDIUM: 1.8, CBR_LARGE: 1.5},
        "strong made": {FOLD: -2.0, CHECK_CALL: 1.0, CBR_SMALL: 1.5, CBR_MEDIUM: 1.2, CBR_LARGE: 0.5},
        "medium made": {FOLD: -0.5, CHECK_CALL: 1.5, CBR_SMALL: 0.8, CBR_MEDIUM: 0.3, CBR_LARGE: -0.5},
        "weak made": {FOLD: 0.5, CHECK_CALL: 1.0, CBR_SMALL: 0.2, CBR_MEDIUM: -0.5, CBR_LARGE: -1.5},
        "strong draw": {FOLD: -1.0, CHECK_CALL: 0.8, CBR_SMALL: 1.0, CBR_MEDIUM: 1.2, CBR_LARGE: 0.8},
        "weak draw": {FOLD: 0.5, CHECK_CALL: 1.0, CBR_SMALL: 0.3, CBR_MEDIUM: -0.5, CBR_LARGE: -1.0},
        "air": {FOLD: 1.5, CHECK_CALL: 0.2, CBR_SMALL: -0.3, CBR_MEDIUM: -0.8, CBR_LARGE: -1.5},
    }

    def _softmax(scores: Dict[int, float]) -> Dict[int, float]:
        mx = max(scores.values())
        weights = {action: math.exp(score - mx) for action, score in scores.items()}
        total = sum(weights.values())
        probs = {action: weight / total for action, weight in weights.items()}
        floor = 0.02
        floored = {action: (1.0 - floor * len(probs)) * prob + floor for action, prob in probs.items()}
        norm = sum(floored.values())
        return {action: prob / norm for action, prob in floored.items()}

    return {bucket: _softmax(scores) for bucket, scores in raw.items()}


_BASE_PSI = _build_postflop_psi()


def postflop_likelihood(
    strength_bucket: str,
    action_bucket: int,
    phi: float = 0.0,
) -> float:
    base = _BASE_PSI.get(strength_bucket)
    if not base:
        return 1.0 / len(ACTION_BUCKETS)

    if phi == 0.0:
        return base.get(action_bucket, 0.02)

    strength_rank = STRENGTH_INDEX.get(strength_bucket, 3)
    weakness = strength_rank / 6.0
    utility = {
        FOLD: -phi * weakness,
        CHECK_CALL: phi * (weakness - 0.3),
        CBR_SMALL: phi * (weakness - 0.2) * 0.8,
        CBR_MEDIUM: phi * (weakness - 0.1) * 0.6,
        CBR_LARGE: phi * weakness * 0.5,
    }
    log_base = {action: math.log(max(base.get(action, 0.02), 1e-9)) for action in ACTION_BUCKETS}
    log_mod = {action: log_base[action] + utility.get(action, 0.0) for action in ACTION_BUCKETS}
    mx = max(log_mod.values())
    weights = {action: math.exp(score - mx) for action, score in log_mod.items()}
    total = sum(weights.values())
    return weights.get(action_bucket, 0.0) / total


class PostflopStrengthFilter:
    """Bayesian filter over the 7 post-flop strength buckets."""

    def __init__(
        self,
        initial_strength_dist: Optional[Dict[str, float]] = None,
        phi: float = 0.0,
    ):
        if initial_strength_dist is not None:
            self.strength_dist: Dict[str, float] = normalize(initial_strength_dist)
        else:
            n = len(STRENGTH_BUCKETS)
            self.strength_dist = {bucket: 1.0 / n for bucket in STRENGTH_BUCKETS}

        self.phi = phi
        self.steps: List[FilterStep] = []

    def update(
        self,
        state_key: StateKey | str,
        action_bucket: int,
    ) -> Dict[str, float]:
        state_key_str = state_key.as_string() if isinstance(state_key, StateKey) else state_key
        unnorm = {
            bucket: prob * postflop_likelihood(bucket, action_bucket, phi=self.phi)
            for bucket, prob in self.strength_dist.items()
        }
        evidence = sum(unnorm.values())
        if evidence <= 0:
            raise ValueError(f"Post-flop filter produced zero evidence at action={action_bucket}.")

        self.strength_dist = {bucket: value / evidence for bucket, value in unnorm.items()}
        top_bucket, top_prob = max(self.strength_dist.items(), key=lambda item: item[1])
        self.steps.append(
            FilterStep(
                state_key=state_key_str,
                action_bucket=action_bucket,
                evidence=evidence,
                ess=effective_sample_size(self.strength_dist),
                top_class=top_bucket,
                top_prob=top_prob,
                layer="postflop",
            )
        )
        return self.strength_dist

    def top_k(self, k: int = 7) -> List[Tuple[str, float]]:
        return sorted(self.strength_dist.items(), key=lambda item: item[1], reverse=True)[:k]

    def expected_strength(self) -> float:
        n = len(STRENGTH_BUCKETS)
        return sum(
            ((n - 1 - STRENGTH_INDEX[bucket]) / (n - 1)) * prob
            for bucket, prob in self.strength_dist.items()
        )

    def log_likelihood(self) -> float:
        return sum(math.log(step.evidence) for step in self.steps if step.evidence > 0)
