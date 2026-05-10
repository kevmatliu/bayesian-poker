"""Expected postflop strength statistics under a combo distribution vs a realized holding."""

from __future__ import annotations

from typing import List, Mapping, Optional, Tuple

import numpy as np

from utils.eval.logutil import eval_log
from utils.filter.postflop import parse_combo_key
from utils.strength.common import Card, parse_cards
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
    n_samples: int = 96,
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
