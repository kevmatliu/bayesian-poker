"""Shared expectation–maximization helpers.

``normalize_log_weights`` applies the log-sum-exp trick to turn unnormalized
log-masses (E-step outputs) into probability tables over string keys (hand
classes or combo keys).

``minibatch_plan`` selects bundle/hand indices for stochastic M-steps: the
returned scale factor reweights the batch gradient so it is unbiased for the
full-data sum.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

# M-step gradient ascent stops when L2 norm of grad falls below this (after L2 penalty on theta).
M_STEP_GRAD_NORM_TOL = 0.1

# Default minibatch sizes (bundles / hands per gradient step). Use 0 for full-batch in callers.
PREFLOP_M_BATCH_SIZE = 64
POSTFLOP_M_BATCH_SIZE = 64


def normalize_log_weights(log_weights: Dict[str, float]) -> Dict[str, float]:
    """Convert unnormalized log-weights ``log w_i`` to probabilities ``w_i / sum w``.

    Subtracts ``max_j log w_j`` before exponentiating for stability. Raises if
    the dictionary is empty or if the total mass is non-positive (numerical
    underflow or inconsistent inputs).
    """
    if not log_weights:
        raise ValueError("normalize_log_weights: empty input")
    m = max(log_weights.values())
    weights = {h: math.exp(w - m) for h, w in log_weights.items()}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("normalize_log_weights: zero total mass")
    return {h: v / total for h, v in weights.items()}


def effective_sample_size(probabilities: Sequence[float]) -> float:
    """ESS = (sum p)² / sum p² for a discrete distribution (sum p is usually 1)."""
    ps = list(probabilities)
    if not ps:
        return 0.0
    s = sum(ps)
    den = sum(p * p for p in ps)
    if den <= 0.0:
        return 0.0
    return (s * s) / den


def max_effective_sample_size(q_by_item: Sequence[Mapping[str, float]]) -> float:
    """Largest per-item ESS when each mapping is a distribution over latent keys."""
    best = 0.0
    for qmap in q_by_item:
        ess = effective_sample_size(list(qmap.values()))
        if ess > best:
            best = ess
    return best


def minibatch_plan(
    n_items: int,
    m_batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, bool]:
    """Pick indices for one M-step minibatch.

    Returns ``(indices, scale, full_batch)`` where ``scale`` is ``n_items / len(indices)``
    when minibatching (unbiased SG for the sum over all items), else ``1.0``.
    """
    if n_items <= 0:
        raise ValueError("minibatch_plan: n_items must be positive")
    if m_batch_size <= 0 or m_batch_size >= n_items:
        return np.arange(n_items, dtype=int), 1.0, True
    bsz = min(int(m_batch_size), n_items)
    ix = rng.choice(n_items, size=bsz, replace=False)
    return ix, n_items / float(bsz), False
