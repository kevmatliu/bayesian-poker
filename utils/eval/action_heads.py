"""Softmax action heads, baselines, and metrics for global-prior evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from utils.action.postflop import (
    CALL as POST_CALL,
    HEURISTIC_BETA_FACING,
    HEURISTIC_BETA_NO_BET,
    RAISE as POST_RAISE,
)
from utils.action.preflop import HEURISTIC_BETA_PREFLOP
from utils.eval.brier import multiclass_brier

NLL_EPS = 1e-12  # floor true-class prob to avoid log(0) in nll


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)    # per-row max for numerical stability
    e = np.exp(logits - m)                   # stabilized exponentials of logits
    return e / e.sum(axis=1, keepdims=True)  # normalize rows to probability simplices


def predict_probs(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(N, K) action distribution given (K, D) weight matrix and (N, D) features."""
    return softmax_rows(X @ beta.T)  # linear model then softmax per row


def remap_no_bet_labels(y: np.ndarray) -> np.ndarray:
    """CALL(=1) -> 0, RAISE(=2) -> 1, matching the row order of ``beta_no_bet``."""
    if y.size == 0:                                                      # nothing to remap
        return y.astype(int)                                             # preserve empty shape with int dtype
    out = np.where(y == POST_CALL, 0, np.where(y == POST_RAISE, 1, -1))  # map call/raise to 0/1, else sentinel
    if (out == -1).any():                                                # invalid labels remain after mapping
        bad = np.unique(y[out == -1])                                    # collect offending raw labels for the error
        raise ValueError(f"unexpected no-bet labels: {bad}")             # fail fast on unsupported encodings
    return out                                                           # binary labels aligned with no-bet head columns


def empirical_marginal(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)  # histogram of observed classes
    counts = np.maximum(counts, 1e-12)                                      # avoid zero masses that break division
    return counts / counts.sum()                                            # normalize to a probability vector


def constant_probs(p: np.ndarray, n: int) -> np.ndarray:
    return np.broadcast_to(p, (n, p.size)).copy()  # repeat the same marginal row for every sample


def nll(P: np.ndarray, y: np.ndarray) -> float:
    if P.shape[0] == 0:                                                      # no rows means undefined average nll
        return float("nan")                                                  # signal missing metric explicitly
    p_true = np.clip(P[np.arange(P.shape[0]), y.astype(int)], NLL_EPS, 1.0)  # true-class probs with floor/ceiling
    return float(-np.log(p_true).mean())                                     # mean negative log likelihood over rows


def mean_brier(P: np.ndarray, y: np.ndarray) -> float:
    if P.shape[0] == 0:                                                          # empty batch has no brier
        return float("nan")                                                      # propagate missing-data convention
    return float(
        np.mean([multiclass_brier(P[i], int(y[i])) for i in range(P.shape[0])])  # average per-row brier scores
    )


def top1_accuracy(P: np.ndarray, y: np.ndarray) -> float:
    if P.shape[0] == 0:                                       # no predictions to score
        return float("nan")                                   # undefined accuracy on empty data
    return float((P.argmax(axis=1) == y.astype(int)).mean())  # fraction of argmax matches labels


def confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    M = np.zeros((n_classes, n_classes), dtype=int)           # accumulate class-by-class counts
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):  # iterate paired labels
        M[t, p] += 1                                          # increment cell for (true, predicted)
    return M                                                  # raw confusion counts (not normalized)


def row_normalise(M: np.ndarray) -> np.ndarray:
    row_sums = M.sum(axis=1, keepdims=True).astype(float)  # per-row totals for scaling
    row_sums = np.where(row_sums == 0, 1.0, row_sums)      # guard divide-by-zero on empty rows
    return M / row_sums                                    # convert counts to conditional distributions p(pred|true)


def global_priors_three_head_predictions(
    *,
    beta_pre: np.ndarray,
    beta_facing: np.ndarray,
    beta_no_bet: np.ndarray,
    X_pre_test: np.ndarray,
    Xf_test: np.ndarray,
    Xn_test: np.ndarray,
    marg_pre: np.ndarray,
    marg_facing: np.ndarray,
    marg_no_bet: np.ndarray,
):
    """Return trained / heuristic / marginal (P_pre*, P_f*, P_n*) for the three heads."""
    P_pre_trained = predict_probs(beta_pre, X_pre_test)                  # learned preflop softmax on held-out features
    P_pre_heuristic = predict_probs(HEURISTIC_BETA_PREFLOP, X_pre_test)  # hand-crafted preflop baseline preds
    P_pre_marginal = constant_probs(marg_pre, X_pre_test.shape[0])       # session empirical preflop prior rows

    P_f_trained = predict_probs(beta_facing, Xf_test)                    # learned facing-bet head on test rows
    P_f_heuristic = predict_probs(HEURISTIC_BETA_FACING, Xf_test)        # heuristic facing-bet baseline
    P_f_marginal = constant_probs(marg_facing, Xf_test.shape[0])         # empirical facing marginal per row

    P_n_trained = predict_probs(beta_no_bet, Xn_test)                    # learned no-bet (call/raise) head
    P_n_heuristic = predict_probs(HEURISTIC_BETA_NO_BET, Xn_test)        # heuristic no-bet baseline
    P_n_marginal = constant_probs(marg_no_bet, Xn_test.shape[0])         # empirical no-bet marginal rows
    return (
        (P_pre_trained, P_pre_heuristic, P_pre_marginal),                # preflop triple: trained/heuristic/marginal
        (P_f_trained, P_f_heuristic, P_f_marginal),                      # facing triple
        (P_n_trained, P_n_heuristic, P_n_marginal),                      # no-bet triple
    )


def metric_rows_three_heads(
    preds,
    y_pre: np.ndarray,
    y_f: np.ndarray,
    y_n_local: np.ndarray,
    *,
    metric_fn,
) -> list[tuple[str, str, float]]:
    (Ppt, Pph, Ppm), (Pft, Pfh, Pfm), (Pnt, Pnh, Pnm) = preds                  # unpack three heads × three model kinds
    rows: list[tuple[str, str, float]] = []                                    # collect (head_name, model_kind, metric_value)
    for name, P in (
        ("trained", Ppt),                                                      # evaluate learned preflop preds
        ("heuristic", Pph),                                                    # evaluate heuristic preflop preds
        ("marginal", Ppm),                                                     # evaluate empirical marginal preflop preds
    ):
        rows.append(("preflop", name, metric_fn(P, y_pre)))                    # record metric on preflop labels
    for name, P in (("trained", Pft), ("heuristic", Pfh), ("marginal", Pfm)):  # facing head variants
        rows.append(("facing", name, metric_fn(P, y_f)))                       # metric vs facing labels
    for name, P in (("trained", Pnt), ("heuristic", Pnh), ("marginal", Pnm)):  # no-bet head variants
        rows.append(("no_bet", name, metric_fn(P, y_n_local)))                 # metric vs remapped no-bet labels
    return rows                                                                # flat list of tagged scores for downstream tables


def summarize_metric_rows(
    nll_rows: Iterable[tuple[str, str, float]],
    brier_rows: Iterable[tuple[str, str, float]],
    top1_rows: Iterable[tuple[str, str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    summary: dict[tuple[str, str], dict[str, float]] = {}   # key (head, model) -> metric bundle
    for head, model, v in nll_rows:                         # fold nll values into the summary dict
        summary.setdefault((head, model), {})["NLL"] = v    # attach nll under the head/model key
    for head, model, v in brier_rows:                       # same for brier
        summary.setdefault((head, model), {})["Brier"] = v  # store multiclass brier
    for head, model, v in top1_rows:                        # same for top-1 accuracy
        summary.setdefault((head, model), {})["top-1"] = v  # store argmax hit rate
    return summary                                          # one dict row per (head, model) with all three metrics
