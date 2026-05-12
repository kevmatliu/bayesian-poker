"""Shared utilities for multinomial action models (preflop and postflop).

Softmax helpers, the length-3 **utility vector** for tendency tilting, and
:class:`ActionPhase` for tagging which feature family a model uses.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

UtilityVector = Tuple[float, float, float]


class ActionPhase(Enum):
    """Which street family an action model belongs to (preflop vs postflop)."""

    PREFLOP = "preflop"
    POSTFLOP = "postflop"


def softmax_dict_float(scores: Mapping[int, float], floor: float = 0.0) -> Dict[int, float]:
    """Stable softmax over int-keyed logits with optional additive floor smoothing."""
    if not scores:
        raise ValueError("empty scores")
    actions = list(scores.keys())
    vals = np.array([scores[a] for a in actions], dtype=float)
    m = float(np.max(vals))
    w = np.exp(vals - m)
    s = float(np.sum(w))
    probs = {a: float(wi / s) for a, wi in zip(actions, w)}
    if floor <= 0:
        return probs
    n = len(probs)
    if floor * n >= 1.0:
        raise ValueError("floor too large")
    out = {a: (1.0 - floor * n) * p + floor for a, p in probs.items()}
    tot = sum(out.values())
    return {a: p / tot for a, p in out.items()}


def dot3(theta: Sequence[float], u: UtilityVector) -> float:
    """Inner product ``theta · u`` for length-3 tendency vectors."""
    return theta[0] * u[0] + theta[1] * u[1] + theta[2] * u[2]


def tendency_deviation_vector(
    candidate_action: int,
    p_base: Mapping[int, float],
    *,
    fold: int,
    call: int,
    raise_: int,
) -> UtilityVector:
    """Centered one-hot **utility** for tendency tilting: ``u_k = 1[a=k] - P_base(k)``.

    Subtracting the baseline expectation keeps the mapping identifiable up to
    an additive constant in each coordinate; softmax is invariant to that
    constant, but centering improves numerical behavior when combining with
    ``log p_base`` in :class:`utils.action.preflop.PreflopActionModel`.
    """
    return (
        float(candidate_action == fold) - float(p_base[fold]),
        float(candidate_action == call) - float(p_base[call]),
        float(candidate_action == raise_) - float(p_base[raise_]),
    )


def softmax_vec(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a 1-D logits vector (subtract max before exp)."""
    m = float(np.max(logits))
    e = np.exp(logits - m)
    return e / float(np.sum(e))
