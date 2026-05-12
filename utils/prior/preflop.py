"""Population preflop baseline: multinomial-logit ``beta`` only (no tendency theta)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

from utils.action.common import ActionPhase, UtilityVector, softmax_dict_float
from utils.action.preflop_context import (
    ACTION_BUCKETS,
    CHECK_CALL,
    FOLD,
    HEURISTIC_BETA_PREFLOP,
    PREFLOP_PHI_DIM,
    RAISE,
    StateKey,
    preflop_feature_vector,
    tendency_deviation_vector as _tendency_deviation_generic,
)


@dataclass
class PreflopPrior:
    """Frozen population weights ``beta_preflop``; action probs are baseline softmax only."""

    floor: float = 0.01
    beta_preflop: Optional[np.ndarray] = None
    _beta_store: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bf = self.beta_preflop if self.beta_preflop is not None else HEURISTIC_BETA_PREFLOP.copy()
        if bf.shape != (3, PREFLOP_PHI_DIM):
            raise ValueError(f"beta_preflop must be (3, {PREFLOP_PHI_DIM}), got {bf.shape}")
        self._beta_store = np.asarray(bf, dtype=float)

    @property
    def mode(self) -> ActionPhase:
        return ActionPhase.PREFLOP

    @property
    def beta_preflop_matrix(self) -> np.ndarray:
        return np.asarray(self._beta_store, dtype=float).copy()

    def _base_logits(self, hand_class: str, state_key: StateKey) -> Dict[int, float]:
        phi = preflop_feature_vector(hand_class, state_key)
        b = self._beta_store
        return {
            FOLD: float(b[0] @ phi),
            CHECK_CALL: float(b[1] @ phi),
            RAISE: float(b[2] @ phi),
        }

    def baseline_action_probs(
        self,
        hand_class: str,
        state_key: StateKey | str,
    ) -> Dict[int, float]:
        """Softmax over logits ``beta @ phi`` (population baseline, no tilt)."""
        if isinstance(state_key, str):
            state_key = StateKey.from_string(state_key)
        return softmax_dict_float(self._base_logits(hand_class, state_key), floor=self.floor)

    def action_utility_vectors(
        self,
        hand_class: str,
        state_key: StateKey | str,
    ) -> Dict[int, UtilityVector]:
        """Tendency deviation utilities from the **baseline** distribution only."""
        if isinstance(state_key, str):
            state_key = StateKey.from_string(state_key)

        p_base = self.baseline_action_probs(hand_class, state_key)
        return {
            a: _tendency_deviation_generic(
                a,
                p_base,
                fold=FOLD,
                call=CHECK_CALL,
                raise_=RAISE,
            )
            for a in ACTION_BUCKETS
        }


def fit_heuristic_preflop_prior() -> PreflopPrior:
    """Return a :class:`PreflopPrior` with ridge-fitted heuristic ``beta``."""
    return PreflopPrior()
