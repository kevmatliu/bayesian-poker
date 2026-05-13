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
    s = (board_str or "").strip()                                    # normalize whitespace for parsing
    if len(s) < 6:                                                   # fewer than three two-char cards means incomplete board
        return []                                                    # signal too-short board with empty list
    return parse_cards([s[i : i + 2] for i in range(0, len(s), 2)])  # split contiguous board string into cards


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
    board = _board_cards(board_str)                                                           # materialize board card objects
    if len(board) < 3:                                                                        # cannot score strength without flop
        eval_log(verbose, "expected_made_and_draw: board too short → (nan, nan)")             # explain early exit
        return float("nan"), float("nan")                                                     # undefined expectations without valid board

    n_pos = sum(1 for _k, v in combo_dist.items() if float(v) > 0.0)                          # count combos with positive mass
    eval_log(
        verbose,
        f"expected_made_and_draw: exact expectation over {n_pos} combos with mass (slow) …",  # warn about cost
    )
    emade = 0.0                                                                               # accumulate weighted made percentile
    edraw = 0.0                                                                               # accumulate weighted draw score
    w = 0.0                                                                                   # total probability mass processed
    for key, mass in combo_dist.items():                                                      # iterate sparse distribution entries
        p = float(mass)                                                                       # coerce to float for comparisons
        if p <= 0.0:                                                                          # skip zero-mass keys
            continue                                                                          # keep loop tight on support
        ca, cb = parse_combo_key(key)                                                         # decode combo key to hole cards
        hole = [ca, cb]                                                                       # list form expected by strength routines
        emade += p * made_strength_percentile(hole, board)                                    # add made contribution
        edraw += p * draw_strength_from_hand(hole, board)                                     # add draw contribution
        w += p                                                                                # track summed weights for normalization
    if w <= 0.0:                                                                              # distribution had no positive mass
        eval_log(verbose, "expected_made_and_draw: zero mass → (nan, nan)")                   # log degenerate input
        return float("nan"), float("nan")                                                     # cannot normalize
    out = emade / w, edraw / w                                                                # return weighted means
    eval_log(
        verbose,
        f"expected_made_and_draw: E[made]={out[0]:.4f} E[draw]={out[1]:.4f}",                 # echo final expectations
    )
    return out                                                                                # tuple of two floats


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
    board = _board_cards(board_str)                                                               # parse board for strength eval
    if len(board) < 3:                                                                            # guard invalid boards
        eval_log(verbose, "expected_made_and_draw_mc: board too short → (nan, nan)")              # trace skip reason
        return float("nan"), float("nan")                                                         # no mc estimate possible

    items = [(k, float(v)) for k, v in combo_dist.items() if float(v) > 0.0]                      # restrict to positive-mass combos
    if not items:                                                                                 # empty support
        eval_log(verbose, "expected_made_and_draw_mc: no positive mass → (nan, nan)")             # log empty dist
        return float("nan"), float("nan")                                                         # undefined sample space
    keys, probs = zip(*items)                                                                     # unzip parallel arrays for numpy sampling
    p = np.array(probs, dtype=float)                                                              # sampling weights as ndarray
    p /= p.sum()                                                                                  # normalize to a true discrete pmf
    rng = rng or np.random.default_rng()                                                          # default reproducible rng if none supplied
    ix = rng.choice(len(keys), size=int(n_samples), p=p)                                          # sample combo indices from dist
    em = 0.0                                                                                      # running sum of made scores
    ed = 0.0                                                                                      # running sum of draw scores
    n = int(n_samples)                                                                            # integer sample count for averaging
    for j in ix:                                                                                  # evaluate each sampled combo
        ca, cb = parse_combo_key(keys[j])                                                         # decode sampled combo
        hole = [ca, cb]                                                                           # hole list for evaluators
        em += made_strength_percentile(hole, board)                                               # accumulate made
        ed += draw_strength_from_hand(hole, board)                                                # accumulate draw
    out = em / n, ed / n                                                                          # simple mc averages
    eval_log(
        verbose,
        f"expected_made_and_draw_mc: n_samples={n} → E[made]≈{out[0]:.4f} E[draw]≈{out[1]:.4f}",  # report estimate
    )
    return out                                                                                    # approximate expectations


def actual_made_and_draw(
    hole_str: str,
    board_str: str,
    *,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Made strength percentile and draw heuristic for one concrete hole (4-char) string."""
    h = (hole_str or "").strip()                                                  # normalize hole token string
    if len(h) != 4:                                                               # need exactly two two-char cards
        eval_log(verbose, "actual_made_and_draw: need 4-char hole → (nan, nan)")  # explain bad hole format
        return float("nan"), float("nan")                                         # cannot parse hole
    board = _board_cards(board_str)                                               # parse community cards
    if len(board) < 3:                                                            # strength undefined preflop / incomplete
        eval_log(verbose, "actual_made_and_draw: board too short → (nan, nan)")   # trace short board
        return float("nan"), float("nan")                                         # no realized strengths
    hole = parse_cards([h[0:2], h[2:4]])                                          # split 4-char string into two cards
    out = (
        made_strength_percentile(hole, board),                                    # realized made percentile vs board
        draw_strength_from_hand(hole, board),                                     # realized draw heuristic score
    )
    eval_log(
        verbose,
        f"actual_made_and_draw: made={out[0]:.4f} draw={out[1]:.4f}",             # echo strengths for debugging
    )
    return out                                                                    # pair of floats


def made_percentile_vector_1326(board_str: str) -> np.ndarray:
    """Made-strength percentile for every static 1,326-combo row on ``board_str``.

    Indices align with :func:`utils.strength.fast_eval.all_combo_keys_fast` and
    :func:`utils.eval.online_csv.combo_probability_columns` CSV column order.
    Combos that share a card with the board are ``nan``.
    """
    board = _board_cards(board_str)                             # board cards for blocking logic
    if len(board) < 3:                                          # no percentile field without flop
        return np.full(1326, np.nan, dtype=np.float64)          # all-dead vector convention
    board_idx = tuple(sorted(card_to_index(c) for c in board))  # canonical board signature for table lookup
    out = np.full(1326, np.nan, dtype=np.float64)               # default dead rows to nan
    live_rows, perc = made_percentile_array(board_idx)          # fetch dense percentiles for legal combos
    out[live_rows] = perc                                       # scatter live values into fixed 1326 layout
    return out                                                  # full vector aligned with combo table order


def actual_made_percentile_at_combo_index(board_str: str, combo_row_index: int) -> float:
    """Made percentile for one row of the static 1,326 table via :func:`made_percentile_vector_1326`."""
    perc = made_percentile_vector_1326(board_str)  # full percentile vector for board
    j = int(combo_row_index)                       # coerce index to int
    if j < 0 or j >= perc.size:                    # bounds check against static length
        return float("nan")                        # invalid row index
    x = float(perc[j])                             # read possibly-nan cell
    return x if np.isfinite(x) else float("nan")   # collapse non-finite to nan for callers


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
    p = np.asarray(p, dtype=np.float64).ravel()                   # coerce combo prob vector to 1d float64
    perc = np.asarray(perc, dtype=np.float64).ravel()             # coerce percentile vector likewise
    if p.size != 1326 or perc.size != 1326:                       # enforce static combo table length
        raise ValueError("p and perc must have length 1,326")     # fail fast on shape mismatch
    p = np.maximum(p, 0.0)                                        # clip negatives before normalization logic
    tot = float(p.sum())                                          # total mass for full-support nll
    nan7 = (float("nan"),) * 7                                    # sentinel tuple for invalid stats paths
    if tot <= 0.0:                                                # completely empty distribution
        return nan7                                               # cannot define calibration metrics
    j = int(true_combo_index)                                     # realized combo row
    if j < 0 or j >= 1326:                                        # guard out-of-range true combo
        return nan7                                               # undefined reference hand
    x_true = float(perc[j])                                       # realized made percentile at true combo
    if not np.isfinite(x_true):                                   # true combo dead vs board or missing data
        return nan7                                               # cannot rank a non-finite truth

    live = np.isfinite(perc)                                      # mask combos that block vs board
    w = p[live]                                                   # restrict probs to live subset
    w_sum = float(w.sum())                                        # mass on live combos
    if w_sum <= 0.0:                                              # no support on playable combos
        return nan7                                               # moments undefined
    w = w / w_sum                                                 # conditional law of X given live rows
    xv = perc[live]                                               # live percentile values aligned with w
    mu = float(np.sum(w * xv))                                    # conditional expectation of made percentile
    var = float(np.sum(w * (xv - mu) ** 2))                       # conditional variance
    sigma = float(np.sqrt(var)) if var > 0.0 else 0.0             # std dev; zero if degenerate
    z = (x_true - mu) / sigma if sigma > 1e-15 else float("nan")  # z-score vs predicted spread
    abs_z = abs(z) if np.isfinite(z) else float("nan")            # magnitude of surprise

    p_hat = p / tot                                               # full simplex normalization for nll on true index
    pt = float(p_hat[j])                                          # model prob on realized combo
    if pt > 0.0:                                                  # finite nll branch
        nll = float(-np.log(pt))                                  # negative log prob of true combo
    else:
        nll = float("inf")                                        # infinite penalty if model assigns zero to truth

    lt = xv < x_true - 1e-15                                      # strict left tail indicator with tiny tie slack
    eq = np.isclose(xv, x_true, rtol=0.0, atol=1e-12)             # tie-at-truth mask
    left_mass = float(np.sum(w[lt]))                              # prob strictly weaker made percentile
    tie_mass = float(np.sum(w[eq]))                               # prob at identical made percentile
    midrank = left_mass + 0.5 * tie_mass                          # tie-aware midrank score
    cdf_le = left_mass + tie_mass                                 # cdf including ties at truth
    return mu, sigma, z, abs_z, nll, midrank, cdf_le              # bundle scalar diagnostics


def _normalize_combo_prob_vector(p: Union[np.ndarray, List[float]]) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()                                 # coerce to 1d probabilities
    if p.size != 1326:                                                          # require full combo support
        raise ValueError(f"expected 1,326 probabilities, got shape {p.shape}")  # explain bad input
    p = np.maximum(p, 0.0)                                                      # enforce non-negativity before sum
    s = float(p.sum())                                                          # total mass check
    if s <= 0.0:                                                                # reject empty vectors
        raise ValueError("probability vector has zero total mass")              # cannot normalize
    return p / s                                                                # return unit-sum nonnegative vector


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
    perc = made_percentile_vector_1326(board_str)                            # percentile field for all combos on board
    p = _normalize_combo_prob_vector(p)                                      # enforce valid simplex on full 1326 vector
    live = np.isfinite(perc)                                                 # playable-combo mask
    if not np.any(live):                                                     # entire board blocks evaluation
        return float("nan"), float("nan"), np.array([]), np.array([]), perc  # empty hist with perc echo
    w = p[live]                                                              # live subset weights
    x = perc[live]                                                           # live percentile values
    w_sum = float(w.sum())                                                   # mass on live subset after clipping negatives already handled upstream
    if w_sum <= 0.0:                                                         # no mass on playable combos despite finiteness mask
        return float("nan"), float("nan"), np.array([]), np.array([]), perc  # cannot build histogram
    w = w / w_sum                                                            # conditional distribution for moments and histogram
    expected = float(np.sum(w * x))                                          # mean made percentile under range
    counts, edges = np.histogram(
        x,                                                                   # sample values are percentiles
        bins=int(n_bins),                                                    # requested bin count
        range=hist_range,                                                    # fix support to unit interval by default
        weights=w,                                                           # probability mass per sample contributes to bin totals
        density=False,                                                       # raw weighted counts, not density
    )
    j = int(np.argmax(counts))                                               # bin index with largest accumulated mass
    mode_center = float(0.5 * (edges[j] + edges[j + 1]))                     # bin midpoint as discrete mode proxy
    return expected, mode_center, edges, counts.astype(np.float64), perc     # return summary + histogram arrays


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
        p,                                                                                     # combo distribution vector
        board_str,                                                                             # board string driving percentile field
        n_bins=n_bins,                                                                         # pass bin resolution
        hist_range=hist_range,                                                                 # pass x-axis span
    )
    if ax is None:                                                                             # create figure when caller did not supply axes
        _, ax = plt.subplots(figsize=(8, 4))                                                   # default single-panel figure size
    centers = 0.5 * (edges[:-1] + edges[1:])                                                   # bin centers for bar plot
    widths = np.diff(edges)                                                                    # bar widths from consecutive edges
    ax.bar(centers, weights, width=widths, align="center", edgecolor="black", linewidth=0.5)   # draw weighted histogram
    ax.axvline(expected, color="C0", linestyle="--", label=f"E[made]={expected:.3f}")          # mark mean
    ax.axvline(mode_center, color="C1", linestyle=":", label=f"hist mode={mode_center:.3f}")   # mark modal bin center
    ax.set_xlabel("made strength percentile")                                                  # annotate x axis
    ax.set_ylabel("probability mass in bin")                                                   # annotate y axis
    ax.set_title(title or "Made percentile under range (weighted histogram)")                  # title or sensible default
    ax.legend()                                                                                # show mean/mode guides
    ax.grid(True, alpha=0.3)                                                                   # light grid for readability
    return ax, {"expected_made": expected, "mode_bin_center": mode_center, "perc_1326": perc}  # return axes + stats dict
