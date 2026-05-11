"""Shared expectation–maximization utilities.

``normalize_log_weights`` is the standard log-sum-exp trick for converting
unnormalized log-masses (E-step outputs) into proper probability tables over
string keys (hand classes or combo keys).
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Dict

LOG = logging.getLogger(__name__)

# M-step gradient ascent stops when L2 norm of grad falls below this (after L2 penalty on theta).
M_STEP_GRAD_NORM_TOL = 0.1

# Default minibatch sizes (bundles / hands per gradient step). Use 0 for full-batch in callers.
PREFLOP_M_BATCH_SIZE = 64
POSTFLOP_M_BATCH_SIZE = 64


class EMPhase(Enum):
    """Which tendency vector EM is optimizing."""

    PREFLOP = "preflop"
    POSTFLOP = "postflop"


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
