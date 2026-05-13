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
    p = np.asarray(prob_vector, dtype=float).ravel()   # force 1-D float work vector
    if p.size == 0 or not (0 <= true_index < p.size):  # undefined score
        return float("nan")
    s = p.sum()                                        # check for unnormalized inputs
    if s > 0:
        p = p / s                                      # project to simplex if needed
    y = np.zeros_like(p)                               # one-hot target vector
    y[int(true_index)] = 1.0                           # realized class gets unit mass
    return float(np.sum((p - y) ** 2))                 # squared error summed over classes


def class_distribution_to_vector(
    dist: Mapping[str, float],
    classes: Sequence[str],
) -> np.ndarray:
    v = np.array([float(dist.get(c, 0.0)) for c in classes], dtype=float)  # align dict to fixed order
    t = v.sum()                                                            # total predicted mass
    if t > 0:
        v = v / t                                                          # normalize if caller passed unnormalized weights
    return v


def brier_preflop169(
    dist_169: Mapping[str, float],
    true_class: str,
    *,
    verbose: bool = False,
) -> float:
    """One row: predicted 169-class distribution vs realized equivalence class."""
    classes = all_169_classes()                             # canonical index order shared across codebase
    if true_class not in classes:                           # unknown label → cannot index
        eval_log(verbose, "brier_preflop169: unknown true_class → nan")
        return float("nan")
    p = class_distribution_to_vector(dist_169, classes)     # pmf vector aligned with ``classes``
    score = multiclass_brier(p, classes.index(true_class))  # Brier vs one-hot true class
    eval_log(verbose, f"brier_preflop169: true={true_class!r} score={score:.6f}")
    return score


def collapse_combo_distribution_to_169(combo_dist: Mapping[str, float]) -> Dict[str, float]:
    """Aggregate 1,326 combo masses into 169 canonical preflop equivalence classes."""
    acc: MutableMapping[str, float] = defaultdict(float)  # running sums per abstract class
    for key, mass in combo_dist.items():                  # iterate sparse or dense combo dict
        m = float(mass)                                   # numeric weight for this combo
        if m <= 0.0:                                      # skip empty support entries
            continue
        ca, cb = parse_combo_key(key)                     # materialize two ``Card`` objects
        cls = get_equivalence_class([ca, cb])             # map ordered hole to ``AKs``-style label
        acc[cls] += m                                     # accumulate all suited/offsuit permutations into one class mass
    total = sum(acc.values())                             # post-aggregation normalization constant
    if total > 0:
        return {k: v / total for k, v in acc.items()}     # return proper pmf on 169-simplex
    return dict(acc)                                      # degenerate all-zero → return empty or zero dict as-is


def brier_preflop_from_combo1326(
    combo_dist: Mapping[str, float],
    true_class: str,
    *,
    verbose: bool = False,
) -> float:
    """Collapse combo probabilities to 169 classes, then Brier vs the realized class."""
    collapsed = collapse_combo_distribution_to_169(combo_dist)      # marginalize suits/isomorphisms
    score = brier_preflop169(collapsed, true_class, verbose=False)  # reuse core Brier (avoid double logging)
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
    """Full 1,326-combo Brier using ``combo_key`` canonical strings (``combo_order`` must match ``all_combo_keys_fast()``)."""
    if true_combo not in combo_order:                                                # cannot locate index of realized combo
        eval_log(verbose, "brier_postflop1326: true_combo not in combo_order → nan")
        return float("nan")
    p = np.array([float(combo_dist.get(k, 0.0)) for k in combo_order], dtype=float)  # dense vector in canonical order
    t = p.sum()                                                                      # may be <1 if dict is sparse / missing tail
    if t > 0:
        p = p / t                                                                    # enforce probabilistic interpretation
    score = multiclass_brier(p, combo_order.index(true_combo))                       # Brier in 1326-way space
    eval_log(verbose, f"brier_postflop1326: true={true_combo!r} score={score:.6f}")
    return score
