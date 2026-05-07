"""Multinomial logistic regression for population baseline policies (no player id)."""

from __future__ import annotations

import numpy as np

from utils.prior.common import softmax_vec


def train_multinomial_3_class(
    X: np.ndarray,
    y: np.ndarray,
    feature_dim: int,
    *,
    learning_rate: float = 0.15,
    max_epochs: int = 2000,
    tol: float = 1e-7,
    l2: float = 0.0,
) -> np.ndarray:
    """Fit beta (3, D) for labels in {0, 1, 2}."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or X.shape[1] != feature_dim:
        raise ValueError(f"X must be (n, {feature_dim})")
    n = X.shape[0]
    if y.shape != (n,):
        raise ValueError("y must be (n,)")

    beta = np.zeros((3, feature_dim), dtype=float)
    for _ in range(max_epochs):
        grad = np.zeros_like(beta)
        for i in range(n):
            logits = beta @ X[i]
            p = softmax_vec(logits)
            yi = y[i]
            if yi not in (0, 1, 2):
                raise ValueError(f"illegal label {yi}")
            for k in range(3):
                grad[k] += (p[k] - (1.0 if yi == k else 0.0)) * X[i]
        grad /= n
        if l2 > 0:
            grad += l2 * beta
        beta -= learning_rate * grad
        if float(np.linalg.norm(grad)) < tol:
            break
    return beta


def train_multinomial_2_class(
    X: np.ndarray,
    y: np.ndarray,
    feature_dim: int,
    *,
    label_a: int,
    label_b: int,
    learning_rate: float = 0.15,
    max_epochs: int = 2000,
    tol: float = 1e-7,
    l2: float = 0.0,
) -> np.ndarray:
    """Fit beta (2, D); y values must be label_a or label_b (mapped to rows 0 and 1)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or X.shape[1] != feature_dim:
        raise ValueError(f"X must be (n, {feature_dim})")
    n = X.shape[0]
    beta = np.zeros((2, feature_dim), dtype=float)
    for _ in range(max_epochs):
        grad = np.zeros_like(beta)
        for i in range(n):
            yi = y[i]
            if yi not in (label_a, label_b):
                raise ValueError(f"expected labels {label_a} or {label_b}, got {yi}")
            local = 0 if yi == label_a else 1
            logits = beta @ X[i]
            p = softmax_vec(logits)
            for k in range(2):
                grad[k] += (p[k] - (1.0 if local == k else 0.0)) * X[i]
        grad /= n
        if l2 > 0:
            grad += l2 * beta
        beta -= learning_rate * grad
        if float(np.linalg.norm(grad)) < tol:
            break
    return beta
