"""Shared helpers for Newton updates on marginal (observed-data) log-likelihood + L2 MAP."""

from __future__ import annotations

import logging
import time
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

# Default finite-difference step for columns of ∇²F from ∂g/∂θ (g = ∇F).
NEWTON_HESSIAN_FD_EPS = 1e-4

# Ridge on the shifted Hessian so the Newton system stays well-conditioned.
NEWTON_HESSIAN_SHIFT_RIDGE = 1e-3

# Backtracking line search: shrink step until MAP objective improves (or min factor hit).
NEWTON_LINE_SEARCH_BACKTRACK = 0.5
NEWTON_LINE_SEARCH_MIN_ALPHA = 1e-6


def log_sum_exp(values: Sequence[float]) -> float:
    """Numerically stable ``log(sum exp(v_i))``; ``-inf`` if empty."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("-inf")
    m = float(np.max(arr))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(arr - m))))


def log_sum_exp_mapping(log_weights: Mapping[str, float]) -> float:
    """``log(sum exp(log_w[k]))`` over a string-keyed map."""
    if not log_weights:
        return float("-inf")
    return log_sum_exp(list(log_weights.values()))


def hessian_from_gradient_fd(
    theta: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    eps: float = NEWTON_HESSIAN_FD_EPS,
    *,
    log: Optional[logging.Logger] = None,
    log_prefix: str = "",
) -> np.ndarray:
    """Symmetric central-difference Hessian ``H_ij ≈ ∂ g_i / ∂ θ_j`` with ``g = ∇F``."""
    k = int(theta.size)
    H = np.zeros((k, k), dtype=float)
    for j in range(k):
        if log is not None:
            log.info(
                "%sHessian FD | column %d/%d (two MAP-gradient evals, each full E-step over bundles)…",
                log_prefix,
                j + 1,
                k,
            )
        t0 = time.perf_counter()
        step = np.zeros(k, dtype=float)
        step[j] = eps
        gp = grad_fn(theta + step)
        gm = grad_fn(theta - step)
        H[:, j] = (gp - gm) / (2.0 * eps)
        if log is not None:
            log.info(
                "%sHessian FD | column %d/%d done | wall_s=%.2f",
                log_prefix,
                j + 1,
                k,
                time.perf_counter() - t0,
            )
    return (H + H.T) * 0.5


def newton_maximization_direction(
    H: np.ndarray,
    g: np.ndarray,
    *,
    ridge: float = NEWTON_HESSIAN_SHIFT_RIDGE,
) -> np.ndarray:
    """Solve ``(H - shift I) d = -g`` with ``shift`` so the system is negative definite (MAP step).

    For a local maximum of ``F``, ``H = ∇² F`` should be negative semi-definite; if finite differencing
    yields indefinite curvature, shifting by ``max(eig(H), 0) + ridge`` stabilizes the update.
    """
    Hs = (H + H.T) * 0.5
    w = np.linalg.eigvalsh(Hs)
    shift = max(float(w.max()), 0.0) + float(ridge)
    H_shifted = Hs - shift * np.eye(len(g), dtype=float)
    return np.linalg.solve(H_shifted, -g)


def backtracking_line_search(
    theta: np.ndarray,
    direction: np.ndarray,
    objective_fn: Callable[[np.ndarray], float],
    current_value: float,
    *,
    initial_alpha: float = 1.0,
    backtrack: float = NEWTON_LINE_SEARCH_BACKTRACK,
    min_alpha: float = NEWTON_LINE_SEARCH_MIN_ALPHA,
    log: Optional[logging.Logger] = None,
    log_prefix: str = "",
) -> tuple[float, np.ndarray, float]:
    """Find ``α`` so ``objective_fn(theta + α d) >= current_value`` (first acceptable step).

    Returns ``(alpha, theta_new, new_value)``.
    """
    alpha = float(initial_alpha)
    d = direction
    trial_n = 0
    while alpha >= min_alpha:
        trial_n += 1
        trial = theta + alpha * d
        if log is not None:
            log.info(
                "%sline search | trial %d | alpha=%.4g | evaluating MAP objective (all bundles)…",
                log_prefix,
                trial_n,
                alpha,
            )
        t0 = time.perf_counter()
        val = float(objective_fn(trial))
        if log is not None:
            log.info(
                "%sline search | trial %d | alpha=%.4g | F=%.6f | objective_eval_s=%.2f",
                log_prefix,
                trial_n,
                alpha,
                val,
                time.perf_counter() - t0,
            )
        if val >= current_value - 1e-12:
            return alpha, trial, val
        if log is not None:
            log.info(
                "%sline search | trial %d rejected (F below baseline); backtrack",
                log_prefix,
                trial_n,
            )
        alpha *= backtrack
    if log is not None:
        log.warning("%sline search | exhausted (alpha < %.1e)", log_prefix, min_alpha)
    return 0.0, theta.copy(), current_value
