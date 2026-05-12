"""Expected postflop strength statistics under a combo distribution vs a realized holding."""

from __future__ import annotations

from typing import List, Mapping, Optional, Tuple, Union

import numpy as np

from utils.eval.logutil import eval_log
from utils.filter.postflop import parse_combo_key
from utils.parse import Card, parse_cards
from utils.strength.fast_eval import card_to_index, made_percentile_array
from utils.strength.postflop import draw_strength_from_hand, made_strength_percentile


def _board_cards(board_str: str) -> List[Card]:
    s = (board_str or "").strip()
    if len(s) < 6:
        return []
    return parse_cards([s[i : i + 2] for i in range(0, len(s), 2)])


def expected_made_and_draw(
    combo_dist: Mapping[str, float],
    board_str: str,
    *,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Under ``combo_dist`` over ``combo_key`` strings, return
    E[made percentile], E[draw strength] for the given board.
    """
    board = _board_cards(board_str)
    if len(board) < 3:
        eval_log(verbose, "expected_made_and_draw: board too short → (nan, nan)")
        return float("nan"), float("nan")

    n_pos = sum(1 for _k, v in combo_dist.items() if float(v) > 0.0)
    eval_log(
        verbose,
        f"expected_made_and_draw: exact expectation over {n_pos} combos with mass (slow) …",
    )
    emade = 0.0
    edraw = 0.0
    w = 0.0
    for key, mass in combo_dist.items():
        p = float(mass)
        if p <= 0.0:
            continue
        ca, cb = parse_combo_key(key)
        hole = [ca, cb]
        emade += p * made_strength_percentile(hole, board)
        edraw += p * draw_strength_from_hand(hole, board)
        w += p
    if w <= 0.0:
        eval_log(verbose, "expected_made_and_draw: zero mass → (nan, nan)")
        return float("nan"), float("nan")
    out = emade / w, edraw / w
    eval_log(
        verbose,
        f"expected_made_and_draw: E[made]={out[0]:.4f} E[draw]={out[1]:.4f}",
    )
    return out


def expected_made_and_draw_mc(
    combo_dist: Mapping[str, float],
    board_str: str,
    *,
    n_samples: int = 100,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Monte Carlo estimate of ``E[made]``, ``E[draw]`` under ``combo_dist`` (unbiased as ``n_samples → ∞``).

    Monte Carlo sampling avoids the 1,326 × expensive ``made_strength_percentile`` loop in full expectation.
    """
    board = _board_cards(board_str)
    if len(board) < 3:
        eval_log(verbose, "expected_made_and_draw_mc: board too short → (nan, nan)")
        return float("nan"), float("nan")

    items = [(k, float(v)) for k, v in combo_dist.items() if float(v) > 0.0]
    if not items:
        eval_log(verbose, "expected_made_and_draw_mc: no positive mass → (nan, nan)")
        return float("nan"), float("nan")
    keys, probs = zip(*items)
    p = np.array(probs, dtype=float)
    p /= p.sum()
    rng = rng or np.random.default_rng()
    ix = rng.choice(len(keys), size=int(n_samples), p=p)
    em = 0.0
    ed = 0.0
    n = int(n_samples)
    for j in ix:
        ca, cb = parse_combo_key(keys[j])
        hole = [ca, cb]
        em += made_strength_percentile(hole, board)
        ed += draw_strength_from_hand(hole, board)
    out = em / n, ed / n
    eval_log(
        verbose,
        f"expected_made_and_draw_mc: n_samples={n} → E[made]≈{out[0]:.4f} E[draw]≈{out[1]:.4f}",
    )
    return out


def actual_made_and_draw(
    hole_str: str,
    board_str: str,
    *,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Made strength percentile and draw heuristic for one concrete hole (4-char) string."""
    h = (hole_str or "").strip()
    if len(h) != 4:
        eval_log(verbose, "actual_made_and_draw: need 4-char hole → (nan, nan)")
        return float("nan"), float("nan")
    board = _board_cards(board_str)
    if len(board) < 3:
        eval_log(verbose, "actual_made_and_draw: board too short → (nan, nan)")
        return float("nan"), float("nan")
    hole = parse_cards([h[0:2], h[2:4]])
    out = (
        made_strength_percentile(hole, board),
        draw_strength_from_hand(hole, board),
    )
    eval_log(
        verbose,
        f"actual_made_and_draw: made={out[0]:.4f} draw={out[1]:.4f}",
    )
    return out


def made_percentile_vector_1326(board_str: str) -> np.ndarray:
    """Made-strength percentile for every static 1,326-combo row on ``board_str``.

    Indices align with :func:`utils.strength.fast_eval.all_combo_keys_fast` and
    :func:`utils.eval.online_csv.combo_probability_columns` CSV column order.
    Combos that share a card with the board are ``nan``.
    """
    board = _board_cards(board_str)
    if len(board) < 3:
        return np.full(1326, np.nan, dtype=np.float64)
    board_idx = tuple(sorted(card_to_index(c) for c in board))
    out = np.full(1326, np.nan, dtype=np.float64)
    live_rows, perc = made_percentile_array(board_idx)
    out[live_rows] = perc
    return out


def actual_made_percentile_at_combo_index(board_str: str, combo_row_index: int) -> float:
    """Made percentile for one row of the static 1,326 table via :func:`made_percentile_vector_1326`."""
    perc = made_percentile_vector_1326(board_str)
    j = int(combo_row_index)
    if j < 0 or j >= perc.size:
        return float("nan")
    x = float(perc[j])
    return x if np.isfinite(x) else float("nan")


def made_percentile_calibration_stats(
    p: Union[np.ndarray, List[float]],
    perc: np.ndarray,
    true_combo_index: int,
):
    """
    Summarize how the realized made percentile sits under the combo distribution ``p``.

    Uses only **live** combos (finite ``perc``); ``p`` is renormalized on that subset for
    moments, ``midrank``, and ``cdf_le``. ``nll`` uses ``p`` normalized over **all** 1,326
    entries.

    Returns ``(mu, sigma, z, abs_z, nll, midrank, cdf_le)`` where:

    * ``midrank`` = ``P(X < x*) + 0.5 P(X = x*)`` for the induced law of made percentile
      ``X`` (ties weighted). Plot its ECDF across hands to compare predictors on how they
      rank the true holding.
    * ``cdf_le`` = ``P(X ≤ x*)`` (includes ties). More mass at or to the left of the true
      made percentile means the distribution assigns more probability to hands at most as
      strong as the realized holding.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    perc = np.asarray(perc, dtype=np.float64).ravel()
    if p.size != 1326 or perc.size != 1326:
        raise ValueError("p and perc must have length 1,326")
    p = np.maximum(p, 0.0)
    tot = float(p.sum())
    nan7 = (float("nan"),) * 7
    if tot <= 0.0:
        return nan7
    j = int(true_combo_index)
    if j < 0 or j >= 1326:
        return nan7
    x_true = float(perc[j])
    if not np.isfinite(x_true):
        return nan7

    live = np.isfinite(perc)
    w = p[live]
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return nan7
    w = w / w_sum
    xv = perc[live]
    mu = float(np.sum(w * xv))
    var = float(np.sum(w * (xv - mu) ** 2))
    sigma = float(np.sqrt(var)) if var > 0.0 else 0.0
    z = (x_true - mu) / sigma if sigma > 1e-15 else float("nan")
    abs_z = abs(z) if np.isfinite(z) else float("nan")

    p_hat = p / tot
    pt = float(p_hat[j])
    if pt > 0.0:
        nll = float(-np.log(pt))
    else:
        nll = float("inf")

    lt = xv < x_true - 1e-15
    eq = np.isclose(xv, x_true, rtol=0.0, atol=1e-12)
    left_mass = float(np.sum(w[lt]))
    tie_mass = float(np.sum(w[eq]))
    midrank = left_mass + 0.5 * tie_mass
    cdf_le = left_mass + tie_mass
    return mu, sigma, z, abs_z, nll, midrank, cdf_le


def _normalize_combo_prob_vector(p: Union[np.ndarray, List[float]]) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size != 1326:
        raise ValueError(f"expected 1,326 probabilities, got shape {p.shape}")
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if s <= 0.0:
        raise ValueError("probability vector has zero total mass")
    return p / s


def expected_made_mean_and_histogram_mode(
    p: Union[np.ndarray, List[float]],
    board_str: str,
    *,
    n_bins: int = 50,
    hist_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Under a 1,326-combo distribution, summarize the induced made-percentile law.

    Returns ``(E[made], mode_bin_center, bin_edges, bin_weights, perc_1326)`` where
    the mode is the midpoint of the histogram bin (``n_bins`` over ``hist_range``)
    that carries the largest probability mass ``Σ_i p_i`` among combos whose
    percentile falls in that bin. Dead combos (blocked by the board) are dropped
    and ``p`` is renormalized over live combos.

    ``perc_1326`` is from :func:`made_percentile_vector_1326` (``nan`` on dead rows).
    """
    perc = made_percentile_vector_1326(board_str)
    p = _normalize_combo_prob_vector(p)
    live = np.isfinite(perc)
    if not np.any(live):
        return float("nan"), float("nan"), np.array([]), np.array([]), perc
    w = p[live]
    x = perc[live]
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return float("nan"), float("nan"), np.array([]), np.array([]), perc
    w = w / w_sum
    expected = float(np.sum(w * x))
    counts, edges = np.histogram(
        x,
        bins=int(n_bins),
        range=hist_range,
        weights=w,
        density=False,
    )
    j = int(np.argmax(counts))
    mode_center = float(0.5 * (edges[j] + edges[j + 1]))
    return expected, mode_center, edges, counts.astype(np.float64), perc


def plot_made_percentile_weighted_histogram(
    p: Union[np.ndarray, List[float]],
    board_str: str,
    *,
    n_bins: int = 50,
    hist_range: Tuple[float, float] = (0.0, 1.0),
    ax=None,
    title: Optional[str] = None,
):
    """Histogram of made percentiles weighted by the 1,326 combo distribution ``p``."""
    import matplotlib.pyplot as plt

    expected, mode_center, edges, weights, perc = expected_made_mean_and_histogram_mode(
        p,
        board_str,
        n_bins=n_bins,
        hist_range=hist_range,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    ax.bar(centers, weights, width=widths, align="center", edgecolor="black", linewidth=0.5)
    ax.axvline(expected, color="C0", linestyle="--", label=f"E[made]={expected:.3f}")
    ax.axvline(mode_center, color="C1", linestyle=":", label=f"hist mode={mode_center:.3f}")
    ax.set_xlabel("made strength percentile")
    ax.set_ylabel("probability mass in bin")
    ax.set_title(title or "Made percentile under range (weighted histogram)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax, {"expected_made": expected, "mode_bin_center": mode_center, "perc_1326": perc}
