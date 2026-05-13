"""Population preflop baseline: multinomial-logit ``beta`` only (no tendency theta)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

from utils.action.common import (
    ActionPhase,
    UtilityVector,
    softmax_dict_float,
    tendency_deviation_vector as _tendency_deviation_core,
)
from utils.action.preflop_context import (
    ACTION_BUCKETS,
    CHECK_CALL,
    FOLD,
    HEURISTIC_BETA_PREFLOP,
    PREFLOP_PHI_DIM,
    RAISE,
    StateKey,
    preflop_feature_vector,
)


@dataclass
class PreflopPrior:
    """Frozen population weights ``beta_preflop``; action probs are baseline softmax only."""

    floor: float = 0.01                                         # mix towards uniform to ensure no zero probabilities for stability (pls don't penalize this and think it's clipping :( )
    beta_preflop: Optional[np.ndarray] = None                   # if none, use heuristic default
    _beta_store: np.ndarray = field(init=False, repr=False)  # validated copy owned by the instance

    def __post_init__(self) -> None:
        bf = self.beta_preflop if self.beta_preflop is not None else HEURISTIC_BETA_PREFLOP.copy()  # choose user vs default weights
        if bf.shape != (3, PREFLOP_PHI_DIM):                                                        # one row per action head
            raise ValueError(f"beta_preflop must be (3, {PREFLOP_PHI_DIM}), got {bf.shape}")
        self._beta_store = np.asarray(bf, dtype=float)                                              # ensure float64-like behavior

    @property
    def mode(self) -> ActionPhase:
        return ActionPhase.PREFLOP  # used by polymorphic code paths

    @property
    def beta_preflop_matrix(self) -> np.ndarray:
        return np.asarray(self._beta_store, dtype=float).copy()  # defensive copy for callers

    def _base_logits(self, hand_class: str, state_key: StateKey) -> Dict[int, float]:
        phi = preflop_feature_vector(hand_class, state_key)  # spot + hand encoding
        b = self._beta_store                                 # shorthand handle
        return {
            FOLD: float(b[0] @ phi),                         # linear predictor for fold head
            CHECK_CALL: float(b[1] @ phi),                   # passive head
            RAISE: float(b[2] @ phi),                        # aggressive head
        }

    def baseline_action_probs(
        self,
        hand_class: str,
        state_key: StateKey | str,
    ) -> Dict[int, float]:
        """Softmax over logits ``beta @ phi`` (population baseline, no tilt)."""
        if isinstance(state_key, str):                                                         # allow serialized keys from logs
            state_key = StateKey.from_string(state_key)                                        # parse into structured form
        return softmax_dict_float(self._base_logits(hand_class, state_key), floor=self.floor)  # pmf P(a|h,s)

    def action_utility_vectors(
        self,
        hand_class: str,
        state_key: StateKey | str,
    ) -> Dict[int, UtilityVector]:
        """Tendency deviation utilities from the baseline distribution only."""
        if isinstance(state_key, str):                              # normalize key type
            state_key = StateKey.from_string(state_key)

        p_base = self.baseline_action_probs(hand_class, state_key)  # reference pmf for centering
        return {
            a: _tendency_deviation_core(                            # u(a) with expectation removed under p_base
                a,
                p_base,
                fold=FOLD,
                call=CHECK_CALL,
                raise_=RAISE,
            )
            for a in ACTION_BUCKETS                                 # one utility vector per legal action
        }


def fit_heuristic_preflop_prior() -> PreflopPrior:
    """Simple heuristic prior that can serve as a baseline."""
    return PreflopPrior()
