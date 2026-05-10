"""Multiclass Brier scores for range calibration (169 preflop classes, 1,326 combos postflop)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np

from utils.filter.postflop import parse_combo_key
from utils.eval.logutil import eval_log
from utils.strength.preflop import all_169_classes, get_equivalence_class


def multiclass_brier(prob_vector: np.ndarray, true_index: int) -> float:
    """
    Sum of squared errors against a one-hot label (multiclass Brier / Brier for forecasting).

    ``prob_vector`` should be non-negative and typically normalized to sum to 1.
    """
    p = np.asarray(prob_vector, dtype=float).ravel()
    if p.size == 0 or not (0 <= true_index < p.size):
        return float("nan")
    s = p.sum()
    if s > 0:
        p = p / s
    y = np.zeros_like(p)
    y[int(true_index)] = 1.0
    return float(np.sum((p - y) ** 2))


def class_distribution_to_vector(
    dist: Mapping[str, float],
    classes: Sequence[str],
) -> np.ndarray:
    v = np.array([float(dist.get(c, 0.0)) for c in classes], dtype=float)
    t = v.sum()
    if t > 0:
        v = v / t
    return v


def brier_preflop169(
    dist_169: Mapping[str, float],
    true_class: str,
    *,
    verbose: bool = False,
) -> float:
    """One row: predicted 169-class distribution vs realized equivalence class."""
    classes = all_169_classes()
    if true_class not in classes:
        eval_log(verbose, "brier_preflop169: unknown true_class → nan")
        return float("nan")
    p = class_distribution_to_vector(dist_169, classes)
    score = multiclass_brier(p, classes.index(true_class))
    eval_log(verbose, f"brier_preflop169: true={true_class!r} score={score:.6f}")
    return score


def collapse_combo_distribution_to_169(combo_dist: Mapping[str, float]) -> Dict[str, float]:
    """Aggregate 1,326 combo masses into 169 canonical preflop equivalence classes."""
    acc: MutableMapping[str, float] = defaultdict(float)
    for key, mass in combo_dist.items():
        m = float(mass)
        if m <= 0.0:
            continue
        ca, cb = parse_combo_key(key)
        cls = get_equivalence_class([ca, cb])
        acc[cls] += m
    total = sum(acc.values())
    if total > 0:
        return {k: v / total for k, v in acc.items()}
    return dict(acc)


def brier_preflop_from_combo1326(
    combo_dist: Mapping[str, float],
    true_class: str,
    *,
    verbose: bool = False,
) -> float:
    """Collapse combo probabilities to 169 classes, then Brier vs the realized class."""
    collapsed = collapse_combo_distribution_to_169(combo_dist)
    score = brier_preflop169(collapsed, true_class, verbose=False)
    eval_log(
        verbose,
        f"brier_preflop_from_combo1326 (1,326→169): true={true_class!r} score={score:.6f}",
    )
    return score


def brier_postflop1326(
    combo_dist: Mapping[str, float],
    combo_order: Sequence[str],
    true_combo: str,
    *,
    verbose: bool = False,
) -> float:
    """Full 1,326-combo Brier using ``combo_key`` canonical strings (matches ``all_combo_keys()`` order)."""
    if true_combo not in combo_order:
        eval_log(verbose, "brier_postflop1326: true_combo not in combo_order → nan")
        return float("nan")
    p = np.array([float(combo_dist.get(k, 0.0)) for k in combo_order], dtype=float)
    t = p.sum()
    if t > 0:
        p = p / t
    score = multiclass_brier(p, combo_order.index(true_combo))
    eval_log(verbose, f"brier_postflop1326: true={true_combo!r} score={score:.6f}")
    return score
