"""Shared utilities for multi-class action priors (preflop & postflop).

Contains softmax helpers, the small **utility vector** type used in tendency
tilting, and :class:`PriorMode` for dispatch. Postflop and preflop priors both
use ``theta`` in ``R^3`` paired with 3-vectors ``u(a)`` so the same dot product
structure applies (fold / passive / aggressive semantics).
"""

from __future__ import annotations

from enum import Enum
from math import exp, log
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

UtilityVector = Tuple[float, float, float]


class PriorMode(Enum):
    """Which baseline + feature map family this prior uses."""

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


def softmax_dict_legacy(scores: Mapping[int, float], floor: float = 0.01) -> Dict[int, float]:
    """Original GTOPrior-style softmax (exp logits)."""
    if not scores:
        raise ValueError("Cannot softmax an empty score mapping")
    if floor < 0:
        raise ValueError("floor must be non-negative")
    if floor * len(scores) >= 1.0:
        raise ValueError("floor is too large for the number of actions")
    mx = max(scores.values())
    weights = {a: exp(v - mx) for a, v in scores.items()}
    total = sum(weights.values())
    probs = {a: v / total for a, v in weights.items()}
    if floor == 0:
        return probs
    floored = {a: (1.0 - floor * len(probs)) * p + floor for a, p in probs.items()}
    norm = sum(floored.values())
    return {a: p / norm for a, p in floored.items()}


def softmax_log_probs_dict(log_p: Mapping[int, float], floor_log: float = -1e300) -> Dict[int, float]:
    scores = {a: max(float(v), floor_log) for a, v in log_p.items()}
    return softmax_dict_float(scores, floor=0.0)


def dot3(theta: Sequence[float], u: UtilityVector) -> float:
    """Inner product ``theta · u`` for length-3 tendency vectors."""
    return theta[0] * u[0] + theta[1] * u[1] + theta[2] * u[2]


def tendency_deviation_vector(candidate_action: int, p_base: Mapping[int, float], *, fold: int, call: int, raise_: int) -> UtilityVector:
    """Centered one-hot **utility** for tendency tilting: ``u_k = 1[a=k] - P_base(k)``.

    Subtracting the baseline expectation keeps the mapping identifiable up to
    an additive constant in each coordinate; softmax is invariant to that
    constant, but centering improves numerical behavior when combining with
    ``log p_base`` in :class:`utils.prior.preflop.PreflopPrior`.
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
