"""Supervised tensors and softmax action-head metrics for ``global_priors.json`` evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from runners.common import (
    collect_postflop_supervised_rows,
    collect_preflop_supervised_rows,
    flatten_hands,
    read_session_names_file,
)
from utils.action.postflop import (
    CALL as POST_CALL,
    HEURISTIC_BETA_FACING,
    HEURISTIC_BETA_NO_BET,
    RAISE as POST_RAISE,
)
from utils.action.preflop import HEURISTIC_BETA_PREFLOP
from utils.eval.brier import multiclass_brier

PrintFn = Callable[[str], None]

NLL_EPS = 1e-12


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class GlobalPriorsSupervisedBundle:
    """Train + held-out supervised design matrices for the three action heads."""

    X_pre_train: np.ndarray
    y_pre_train: np.ndarray
    Xf_train: np.ndarray
    yf_train: np.ndarray
    Xn_train: np.ndarray
    yn_train: np.ndarray
    X_pre_test: np.ndarray
    y_pre_test: np.ndarray
    Xf_test: np.ndarray
    yf_test: np.ndarray
    Xn_test: np.ndarray
    yn_test: np.ndarray
    cache_meta: dict
    loaded_from_cache: bool
    cache_path: Path


def default_supervised_cache_path(repo: Path) -> Path:
    return repo / "artifacts" / "global_priors_eval_supervised.npz"


def load_or_build_global_priors_supervised(
    repo: Path,
    *,
    train_file: Path,
    eval_component_file: Path,
    theta_file: Path,
    cache_path: Path | None = None,
    pluribus_root: Path | None = None,
    postflop_equity_mc: int = 8,
    force_rebuild: bool = False,
    print_fn: PrintFn | None = print,
) -> GlobalPriorsSupervisedBundle:
    """
    Load ``artifacts/global_priors_eval_supervised.npz`` when metadata matches, else build.

    Held-out hands use sessions in ``eval_component_file ∪ theta_file``.
    """
    cache_path = cache_path or default_supervised_cache_path(repo)
    pluribus_root = pluribus_root or (repo / "pluribus")
    log = print_fn or (lambda _s: None)

    train_s = read_session_names_file(train_file)
    eval_s = read_session_names_file(eval_component_file)
    theta_s = read_session_names_file(theta_file)
    held_names = tuple(sorted(set(eval_s) | set(theta_s)))

    cache_meta = {
        "sessions_train_sha256": _sha256_file(train_file),
        "sessions_eval_component_sha256": _sha256_file(eval_component_file),
        "sessions_theta_sha256": _sha256_file(theta_file),
        "held_session_names_union": list(held_names),
        "postflop_equity_mc": int(postflop_equity_mc),
    }

    def gather(refs):
        Xpre, ypre = collect_preflop_supervised_rows(refs)
        Xf, yf, Xn, yn = collect_postflop_supervised_rows(
            refs, postflop_equity_mc=postflop_equity_mc
        )
        return Xpre, ypre, Xf, yf, Xn, yn

    def cache_ok(z: np.lib.npyio.NpzFile) -> bool:
        try:
            got = json.loads(str(z["cache_meta_json"].item()))
        except Exception:
            return False
        return got == cache_meta

    loaded = False
    if cache_path.is_file() and not force_rebuild:
        z = np.load(cache_path, allow_pickle=False)
        try:
            if cache_ok(z):
                bundle = GlobalPriorsSupervisedBundle(
                    X_pre_train=z["X_pre_train"],
                    y_pre_train=z["y_pre_train"],
                    Xf_train=z["Xf_train"],
                    yf_train=z["yf_train"],
                    Xn_train=z["Xn_train"],
                    yn_train=z["yn_train"],
                    X_pre_test=z["X_pre_test"],
                    y_pre_test=z["y_pre_test"],
                    Xf_test=z["Xf_test"],
                    yf_test=z["yf_test"],
                    Xn_test=z["Xn_test"],
                    yn_test=z["yn_test"],
                    cache_meta=cache_meta,
                    loaded_from_cache=True,
                    cache_path=cache_path,
                )
                loaded = True
        finally:
            z.close()

    if loaded:
        log(
            f"session lists — train: {len(train_s)} | eval-side: {len(eval_s)} | "
            f"theta: {len(theta_s)} | held-out union: {len(held_names)}"
        )
        log(f"Loaded supervised rows from {cache_path.relative_to(repo)}")
        log(
            f"  train: preflop n={len(bundle.y_pre_train)} | "
            f"post facing={len(bundle.yf_train)} no_bet={len(bundle.yn_train)}"
        )
        log(
            f"  held-out: preflop n={len(bundle.y_pre_test)} | "
            f"post facing={len(bundle.yf_test)} no_bet={len(bundle.yn_test)}"
        )
        return bundle

    if cache_path.is_file() and not force_rebuild:
        log("Supervised cache present but metadata mismatch — rebuilding…")
    else:
        log("Gathering supervised rows (slow; saved to npz for next run)…")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    train_inputs = [pluribus_root / s for s in train_s]
    held_inputs = [pluribus_root / s for s in held_names]
    train_refs = flatten_hands(train_inputs)
    heldout_refs = flatten_hands(held_inputs)
    log(f"train hands: {len(train_refs)} | held-out hands: {len(heldout_refs)}")

    X_pre_train, y_pre_train, Xf_train, yf_train, Xn_train, yn_train = gather(train_refs)
    X_pre_test, y_pre_test, Xf_test, yf_test, Xn_test, yn_test = gather(heldout_refs)

    meta_json = np.empty((), dtype=object)
    meta_json[()] = json.dumps(cache_meta, sort_keys=True)
    np.savez_compressed(
        cache_path,
        cache_meta_json=meta_json,
        X_pre_train=X_pre_train,
        y_pre_train=y_pre_train,
        Xf_train=Xf_train,
        yf_train=yf_train,
        Xn_train=Xn_train,
        yn_train=yn_train,
        X_pre_test=X_pre_test,
        y_pre_test=y_pre_test,
        Xf_test=Xf_test,
        yf_test=yf_test,
        Xn_test=Xn_test,
        yn_test=yn_test,
    )
    log(f"Wrote {cache_path.relative_to(repo)}")
    return GlobalPriorsSupervisedBundle(
        X_pre_train=X_pre_train,
        y_pre_train=y_pre_train,
        Xf_train=Xf_train,
        yf_train=yf_train,
        Xn_train=Xn_train,
        yn_train=yn_train,
        X_pre_test=X_pre_test,
        y_pre_test=y_pre_test,
        Xf_test=Xf_test,
        yf_test=yf_test,
        Xn_test=Xn_test,
        yn_test=yn_test,
        cache_meta=cache_meta,
        loaded_from_cache=False,
        cache_path=cache_path,
    )


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def predict_probs(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(N, K) action distribution given (K, D) weight matrix and (N, D) features."""
    return softmax_rows(X @ beta.T)


def remap_no_bet_labels(y: np.ndarray) -> np.ndarray:
    """CALL(=1) -> 0, RAISE(=2) -> 1, matching ``beta_no_bet`` row order."""
    if y.size == 0:
        return y.astype(int)
    out = np.where(y == POST_CALL, 0, np.where(y == POST_RAISE, 1, -1))
    if (out == -1).any():
        bad = np.unique(y[out == -1])
        raise ValueError(f"unexpected no-bet labels: {bad}")
    return out


def empirical_marginal(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1e-12)
    return counts / counts.sum()


def _constant_probs(p: np.ndarray, n: int) -> np.ndarray:
    return np.broadcast_to(p, (n, p.size)).copy()


def nll(P: np.ndarray, y: np.ndarray) -> float:
    if P.shape[0] == 0:
        return float("nan")
    p_true = np.clip(P[np.arange(P.shape[0]), y.astype(int)], NLL_EPS, 1.0)
    return float(-np.log(p_true).mean())


def mean_brier(P: np.ndarray, y: np.ndarray) -> float:
    if P.shape[0] == 0:
        return float("nan")
    return float(
        np.mean([multiclass_brier(P[i], int(y[i])) for i in range(P.shape[0])])
    )


def top1_accuracy(P: np.ndarray, y: np.ndarray) -> float:
    if P.shape[0] == 0:
        return float("nan")
    return float((P.argmax(axis=1) == y.astype(int)).mean())


def confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        M[t, p] += 1
    return M


def row_normalise(M: np.ndarray) -> np.ndarray:
    row_sums = M.sum(axis=1, keepdims=True).astype(float)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return M / row_sums


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
    """Trained / heuristic / marginal (P_pre*, P_f*, P_n*) for the three heads."""
    P_pre_trained = predict_probs(beta_pre, X_pre_test)
    P_pre_heuristic = predict_probs(HEURISTIC_BETA_PREFLOP, X_pre_test)
    P_pre_marginal = _constant_probs(marg_pre, X_pre_test.shape[0])

    P_f_trained = predict_probs(beta_facing, Xf_test)
    P_f_heuristic = predict_probs(HEURISTIC_BETA_FACING, Xf_test)
    P_f_marginal = _constant_probs(marg_facing, Xf_test.shape[0])

    P_n_trained = predict_probs(beta_no_bet, Xn_test)
    P_n_heuristic = predict_probs(HEURISTIC_BETA_NO_BET, Xn_test)
    P_n_marginal = _constant_probs(marg_no_bet, Xn_test.shape[0])
    return (
        (P_pre_trained, P_pre_heuristic, P_pre_marginal),
        (P_f_trained, P_f_heuristic, P_f_marginal),
        (P_n_trained, P_n_heuristic, P_n_marginal),
    )


def metric_rows_three_heads(
    preds,
    y_pre: np.ndarray,
    y_f: np.ndarray,
    y_n_local: np.ndarray,
    *,
    metric_fn,
) -> list[tuple[str, str, float]]:
    (Ppt, Pph, Ppm), (Pft, Pfh, Pfm), (Pnt, Pnh, Pnm) = preds
    rows: list[tuple[str, str, float]] = []
    for name, P in (("trained", Ppt), ("heuristic", Pph), ("marginal", Ppm)):
        rows.append(("preflop", name, metric_fn(P, y_pre)))
    for name, P in (("trained", Pft), ("heuristic", Pfh), ("marginal", Pfm)):
        rows.append(("facing", name, metric_fn(P, y_f)))
    for name, P in (("trained", Pnt), ("heuristic", Pnh), ("marginal", Pnm)):
        rows.append(("no_bet", name, metric_fn(P, y_n_local)))
    return rows


def summarize_metric_rows(
    nll_rows: Iterable[tuple[str, str, float]],
    brier_rows: Iterable[tuple[str, str, float]],
    top1_rows: Iterable[tuple[str, str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for head, model, v in nll_rows:
        summary.setdefault((head, model), {})["NLL"] = v
    for head, model, v in brier_rows:
        summary.setdefault((head, model), {})["Brier"] = v
    for head, model, v in top1_rows:
        summary.setdefault((head, model), {})["top-1"] = v
    return summary
