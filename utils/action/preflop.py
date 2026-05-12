"""Preflop action model: population baseline × explicit ``theta_pre`` tendency tilt."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, Sequence, Tuple

from utils.action.common import dot3, softmax_dict_float
from utils.action.preflop_context import (
    ACTION_BUCKETS,
    CHECK_CALL,
    FOLD,
    RAISE,
    StateKey,
    canonical_preflop_action,
    tendency_deviation_vector as _tendency_deviation_generic,
)
from utils.prior.preflop import PreflopPrior, fit_heuristic_preflop_prior
from utils.strength.preflop import HandClassFeatures, all_169_classes, hand_class_features


def _canonical_action(action_bucket: int) -> int:
    return canonical_preflop_action(action_bucket)


@dataclass
class PreflopActionModel:
    """Multinomial-logit baseline (:class:`PreflopPrior`) × tendency ``theta_pre``."""

    prior: PreflopPrior
    theta_pre: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        t = tuple(float(x) for x in self.theta_pre)
        if len(t) != 3:
            raise ValueError("theta_pre must have length 3")
        self.theta_pre = t

    @property
    def floor(self) -> float:
        return self.prior.floor

    def with_theta(self, theta_pre: Sequence[float]) -> PreflopActionModel:
        return PreflopActionModel(self.prior, tuple(float(x) for x in theta_pre))

    def action_probs_given_theta(
        self,
        hand_class: str,
        state_key: StateKey | str,
        theta_pre: Sequence[float],
    ) -> Dict[int, float]:
        """Action distribution under this baseline ``beta`` and an explicit ``theta_pre``."""
        if isinstance(state_key, str):
            state_key = StateKey.from_string(state_key)

        base_probs = self.prior.baseline_action_probs(hand_class, state_key)
        theta_t = tuple(float(x) for x in theta_pre)
        if len(theta_t) != 3:
            raise ValueError("theta_pre must have length 3")

        modulated_scores = {
            a: log(base_probs[a])
            + dot3(
                theta_t,
                _tendency_deviation_generic(
                    a,
                    base_probs,
                    fold=FOLD,
                    call=CHECK_CALL,
                    raise_=RAISE,
                ),
            )
            for a in ACTION_BUCKETS
        }

        return softmax_dict_float(modulated_scores, floor=self.floor)

    def action_probs(self, hand_class: str, state_key: StateKey | str) -> Dict[int, float]:
        return self.action_probs_given_theta(hand_class, state_key, self.theta_pre)

    def action_probs_with_theta(
        self,
        hand_class: str,
        state_key: StateKey | str,
        theta_pre: Sequence[float],
    ) -> Dict[int, float]:
        return self.action_probs_given_theta(hand_class, state_key, theta_pre)

    def action_probability_given_theta(
        self,
        hand_class: str,
        state_key: StateKey | str,
        action_bucket: int,
        theta_pre: Sequence[float],
    ) -> float:
        action = _canonical_action(action_bucket)
        return self.action_probs_given_theta(hand_class, state_key, theta_pre)[action]

    def action_probability(
        self,
        hand_class: str,
        state_key: StateKey | str,
        action_bucket: int,
    ) -> float:
        action = _canonical_action(action_bucket)
        return self.action_probs(hand_class, state_key)[action]

    def full_table_for_state(self, state_key: StateKey | str) -> Dict[str, Dict[int, float]]:
        return {h: self.action_probs(h, state_key) for h in all_169_classes()}


# --- re-export context + prior for a single import surface ---

from utils.action.preflop_context import (  # noqa: E402
    ACTION_BUCKETS,
    CHECK_CALL,
    FOLD,
    HEURISTIC_BETA_PREFLOP,
    POSITIONS,
    PREFLOP_PHI_DIM,
    RAISE,
    StateKey,
    build_state_key,
    canonical_preflop_action,
    features_matrix_preflop,
    preflop_feature_vector,
    reference_preflop_logits,
    state_key_from_parse_state,
    tendency_deviation_vector,
    theta_pre_from_phi,
    train_baseline_preflop,
)

__all__ = [
    "ACTION_BUCKETS",
    "CHECK_CALL",
    "FOLD",
    "HEURISTIC_BETA_PREFLOP",
    "HandClassFeatures",
    "POSITIONS",
    "PREFLOP_PHI_DIM",
    "PreflopActionModel",
    "PreflopPrior",
    "RAISE",
    "StateKey",
    "build_state_key",
    "canonical_preflop_action",
    "features_matrix_preflop",
    "fit_heuristic_preflop_prior",
    "hand_class_features",
    "preflop_feature_vector",
    "reference_preflop_logits",
    "state_key_from_parse_state",
    "tendency_deviation_vector",
    "theta_pre_from_phi",
    "train_baseline_preflop",
]
