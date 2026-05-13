"""
Helper functions for EM implementations in both preflop and postflop modules.

``normalize_log_weights``:
- convert unnormalized log-weights to probabilities using log-sum-exp

``effective_sample_size``:
- computes the ESS of a discrete distribution, 1 / sum p^2

``max_effective_sample_size``:
- computes the max ESS across a sequence of distributions for logging and diagnostics purposes

``minibatch_plan``:
- under the EM mini-batching variant, selects bundle/hand indices for stochastic M-steps
- returned scale factor reweights the batch gradient to maintain unbiasedness for the full data
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

M_STEP_GRAD_NORM_TOL = 0.1  # convergence tolerance of the gradient ascent M-step for theta_post

PREFLOP_M_BATCH_SIZE = 64   # default, 0 or negative means no mini-batching (full batch M-step)
POSTFLOP_M_BATCH_SIZE = 64  # mirror default for postflop bundle gradient stacks


def normalize_log_weights(log_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Helper function to normalize log-weights using log-sum-exp
    """
    if not log_weights:
        raise ValueError("normalize_log_weights: empty input")
    m = max(log_weights.values())                                   # log-sum-exp
    weights = {h: math.exp(w - m) for h, w in log_weights.items()} 
    total = sum(weights.values())                                  
    if total <= 0:                                                 
        raise ValueError("normalize_log_weights: zero total mass")
    return {h: v / total for h, v in weights.items()}               # proper pmf over latent keys


def effective_sample_size(probabilities: Sequence[float]) -> float:
    """ESS = (sum p)^2 / sum p^2 for a discrete distribution (sum p is usually 1)."""
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
    best = 0.0                                            # running maximum across items
    for qmap in q_by_item:                                # each hand / timestep posterior
        ess = effective_sample_size(list(qmap.values()))  # ESS of that discrete distribution
        if ess > best:                                    # track peak concentration for diagnostics
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
    if n_items <= 0:                                     # invalid batching problem size
        raise ValueError("minibatch_plan: n_items must be positive")
    if m_batch_size <= 0 or m_batch_size >= n_items:     # use full dataset (exact M-step)
        return np.arange(n_items, dtype=int), 1.0, True
    bsz = min(int(m_batch_size), n_items)                # never request more than available items
    ix = rng.choice(n_items, size=bsz, replace=False)    # uniform subsample without replacement

    return ix, n_items / float(bsz), False               # debias stochastic gradient sum
