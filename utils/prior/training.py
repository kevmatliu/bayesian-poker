"""Multinomial logistic regression for population baseline policies (no player id).

Each routine implements full-batch gradient descent on the softmax
cross-entropy. Gradients are the classic ``p - one_hot(y)`` form multiplied
by ``X[i]`` and averaged over ``n``. This is simple and robust for moderate
``n``; there is no line search—``learning_rate`` and ``max_epochs`` should
be set for the corpus size (``train.py`` defaults are a reasonable starting point).

L2 penalty (if ``l2 > 0``) is added **on the gradient** as ``l2 * beta``,
i.e. weight decay toward zero, matching the usual MAP interpretation with a
Gaussian prior on weights.
"""

from __future__ import annotations

import numpy as np

from utils.action.common import softmax_vec


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
    """Fit a ``(3, feature_dim)`` weight matrix for 3-class softmax regression.

    Rows of ``beta`` correspond to class logits; there is no separate bias row
    beyond whatever bias column exists inside ``X`` (callers include a constant
    ``1.0`` in ``phi``).

    Args:
        X: Design matrix ``(n_samples, feature_dim)``.
        y: Integer labels in ``{0, 1, 2}``.
        feature_dim: Expected second dimension of ``X`` (explicit for validation).
        learning_rate: Fixed step size for gradient descent.
        max_epochs: Maximum full passes over the dataset.
        tol: Stop when the Frobenius norm of the average gradient falls below this.
        l2: Coefficient for L2 weight decay (applied to ``beta`` on the gradient).

    Returns:
        ``beta`` with shape ``(3, feature_dim)``.
    """
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
    """Fit a ``(2, feature_dim)`` weight matrix for binary softmax regression.

    Used for postflop **no-bet** decisions where legal actions are only
    ``label_a`` and ``label_b`` (typically call vs raise). Externally those
    may be encoded as indices ``1`` and ``2`` to align with the 3-action enum;
    here they are mapped to internal rows ``0`` and ``1``.

    Args:
        X: Design matrix ``(n_samples, feature_dim)``.
        y: Labels, each equal to ``label_a`` or ``label_b``.
        feature_dim: Expected second dimension of ``X``.
        label_a: External integer label mapped to softmax row 0.
        label_b: External integer label mapped to softmax row 1.
        learning_rate, max_epochs, tol, l2: Same semantics as
            :func:`train_multinomial_3_class`.

    Returns:
        ``beta`` with shape ``(2, feature_dim)``.
    """
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
