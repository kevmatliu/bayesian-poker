"""Shared expectation–maximization utilities."""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Dict

LOG = logging.getLogger(__name__)

# M-step gradient ascent stops when L2 norm of grad falls below this (after L2 penalty on theta).
M_STEP_GRAD_NORM_TOL = 1e-5


class EMPhase(Enum):
    """Which tendency vector EM is optimizing."""

    PREFLOP = "preflop"
    POSTFLOP = "postflop"


def normalize_log_weights(log_weights: Dict[str, float]) -> Dict[str, float]:
    """Convert unnormalized log-weights to a normalized probability table."""
    if not log_weights:
        raise ValueError("normalize_log_weights: empty input")
    m = max(log_weights.values())
    weights = {h: math.exp(w - m) for h, w in log_weights.items()}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("normalize_log_weights: zero total mass")
    return {h: v / total for h, v in weights.items()}
