"""Held-out action prediction metrics for per-player θ in ``player_thetas.json``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from runners.common import hole_cards_to_hand_class, preflop_decisions_for_hand
from utils.action.postflop import (
    PostflopActionModel,
    PostflopPrior,
    FOLD as POST_FOLD,
    CALL as POST_CALL,
    RAISE as POST_RAISE,
)
from utils.action.preflop import (
    PreflopActionModel,
    PreflopPrior,
    canonical_preflop_action,
    FOLD as PRE_FOLD,
    CHECK_CALL as PRE_CALL,
    RAISE as PRE_RAISE,
)
from utils.eval.global_priors_evaluation_helpers import mean_brier, nll
from utils.postflop_runner_bridge import collect_postflop_observations_known_hole_cards

L2_PENALTY = 0.25


def load_global_betas(priors: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B_pre = np.asarray(priors["preflop"]["beta_preflop"], dtype=float)
    B_f = np.asarray(priors["postflop"]["beta_facing"], dtype=float)
    B_n = np.asarray(priors["postflop"]["beta_no_bet"], dtype=float)
    return B_pre, B_f, B_n


def load_player_thetas(payload: dict) -> tuple[list[str], dict[str, tuple], dict[str, tuple]]:
    players = list(payload["players"].keys())
    theta_pre = {p: tuple(payload["players"][p]["theta_pre"]) for p in players}
    theta_post = {p: tuple(payload["players"][p]["theta_post"]) for p in players}
    return players, theta_pre, theta_post


def collect_player_preflop(refs, player: str) -> list[tuple]:
    rows = []
    for ref in refs:
        if player not in ref.hand.player_names:
            continue
        hc = hole_cards_to_hand_class(ref.hand.hole_cards.get(player, "") or "")
        if hc is None:
            continue
        for dec in preflop_decisions_for_hand(ref.hand, player, ref.global_index):
            rows.append((hc, dec.state_key, canonical_preflop_action(dec.action_bucket)))
    return rows


def collect_player_postflop(refs, player: str) -> tuple[list, list]:
    facing, no_bet = [], []
    for ref in refs:
        if player not in ref.hand.player_names:
            continue
        obs = collect_postflop_observations_known_hole_cards(ref.hand, player, ref.global_index)
        if obs is None:
            continue
        for feat, action in obs.decisions:
            if feat.facing_bet:
                facing.append((feat, int(action)))
            else:
                if int(action) == POST_FOLD:
                    continue
                no_bet.append((feat, int(action)))
    return facing, no_bet


def _to_array_3(probs_dict, idx_a, idx_b, idx_c):
    return np.array([probs_dict[idx_a], probs_dict[idx_b], probs_dict[idx_c]], dtype=float)


def predict_preflop(rows, prior):
    if not rows:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)
    P = np.zeros((len(rows), 3))
    y = np.zeros(len(rows), dtype=int)
    for i, (hc, sk, a) in enumerate(rows):
        probs = prior.action_probs(hc, sk)
        P[i] = _to_array_3(probs, PRE_FOLD, PRE_CALL, PRE_RAISE)
        y[i] = int(a)
    return P, y


def predict_postflop_facing(rows, prior):
    if not rows:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)
    P = np.zeros((len(rows), 3))
    y = np.zeros(len(rows), dtype=int)
    for i, (feat, a) in enumerate(rows):
        probs = prior.action_probs(feat)
        P[i] = _to_array_3(probs, POST_FOLD, POST_CALL, POST_RAISE)
        y[i] = int(a)
    return P, y


def predict_postflop_no_bet(rows, prior):
    if not rows:
        return np.zeros((0, 2)), np.zeros((0,), dtype=int)
    P = np.zeros((len(rows), 2))
    y = np.zeros(len(rows), dtype=int)
    for i, (feat, a) in enumerate(rows):
        probs = prior.action_probs(feat)
        P[i, 0] = probs[POST_CALL]
        P[i, 1] = probs[POST_RAISE]
        y[i] = 0 if int(a) == POST_CALL else 1
    return P, y


def make_priors_for_player(
    player: str,
    *,
    B_pre: np.ndarray,
    B_facing: np.ndarray,
    B_no_bet: np.ndarray,
    theta_pre: tuple,
    theta_post: tuple,
):
    pre0 = PreflopActionModel(PreflopPrior(beta_preflop=B_pre), (0.0, 0.0, 0.0))
    preH = PreflopActionModel(PreflopPrior(beta_preflop=B_pre), tuple(theta_pre))
    post0 = PostflopActionModel(
        PostflopPrior(beta_facing=B_facing, beta_no_bet=B_no_bet), (0.0, 0.0, 0.0)
    )
    postH = PostflopActionModel(
        PostflopPrior(beta_facing=B_facing, beta_no_bet=B_no_bet), tuple(theta_post)
    )
    return pre0, preH, post0, postH


def build_player_prediction_dict(
    players: list[str],
    *,
    B_pre: np.ndarray,
    B_facing: np.ndarray,
    B_no_bet: np.ndarray,
    theta_pre: dict[str, tuple],
    theta_post: dict[str, tuple],
    em_refs,
    online_refs,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``PRED[(player, split, head)] = (P0, P_theta, y)`` with split ``em`` or ``online``."""
    data_pre = {
        (p, sp): collect_player_preflop(refs, p)
        for p in players
        for sp, refs in (("em", em_refs), ("online", online_refs))
    }
    data_post = {
        (p, sp): collect_player_postflop(refs, p)
        for p in players
        for sp, refs in (("em", em_refs), ("online", online_refs))
    }
    pred: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for player in players:
        pre0, preH, post0, postH = make_priors_for_player(
            player,
            B_pre=B_pre,
            B_facing=B_facing,
            B_no_bet=B_no_bet,
            theta_pre=theta_pre[player],
            theta_post=theta_post[player],
        )
        for sp in ("em", "online"):
            rows_pre = data_pre[(player, sp)]
            P0, y = predict_preflop(rows_pre, pre0)
            Pt, _ = predict_preflop(rows_pre, preH)
            pred[(player, sp, "preflop")] = (P0, Pt, y)

            rows_f, rows_n = data_post[(player, sp)]
            P0, y = predict_postflop_facing(rows_f, post0)
            Pt, _ = predict_postflop_facing(rows_f, postH)
            pred[(player, sp, "facing")] = (P0, Pt, y)

            P0, y = predict_postflop_no_bet(rows_n, post0)
            Pt, _ = predict_postflop_no_bet(rows_n, postH)
            pred[(player, sp, "no_bet")] = (P0, Pt, y)
    return pred


def flip_rate(P0: np.ndarray, Pt: np.ndarray) -> float:
    if P0.shape[0] == 0:
        return float("nan")
    return float((P0.argmax(axis=1) != Pt.argmax(axis=1)).mean())


def realised_shifts(P0: np.ndarray, Pt: np.ndarray, y: np.ndarray) -> np.ndarray:
    if P0.shape[0] == 0:
        return np.zeros(0)
    idx = np.arange(P0.shape[0])
    return np.abs(Pt[idx, y.astype(int)] - P0[idx, y.astype(int)])


def heldout_gradient_norm_components(
    Pt: np.ndarray, y: np.ndarray, theta_hat: tuple, n_actions: int
) -> np.ndarray:
    if Pt.shape[0] == 0:
        return np.zeros(n_actions)
    g = np.zeros(n_actions)
    for i in range(Pt.shape[0]):
        e = np.zeros(n_actions)
        e[int(y[i])] = 1.0
        g += e - Pt[i]
    g /= Pt.shape[0]
    g -= L2_PENALTY * np.asarray(theta_hat[:n_actions])
    return g


def online_summary_rows(
    pred: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    players: list[str],
    theta_pre: dict[str, tuple],
    theta_post: dict[str, tuple],
) -> list[dict]:
    shift_arrays = {key: realised_shifts(P0, Pt, y) for key, (P0, Pt, y) in pred.items()}
    rows_summary: list[dict] = []
    for (p, sp, head), (P0, Pt, y) in pred.items():
        if sp != "online":
            continue
        s = shift_arrays[(p, sp, head)]
        if head == "preflop":
            theta_hat, n_actions = theta_pre[p], 3
        else:
            theta_hat, n_actions = theta_post[p], 3 if head == "facing" else 2
        g = heldout_gradient_norm_components(Pt, y, theta_hat, n_actions)
        rows_summary.append(
            {
                "player": p,
                "head": head,
                "N": int(y.size),
                "ΔNLL": nll(P0, y) - nll(Pt, y),
                "ΔBrier": mean_brier(P0, y) - mean_brier(Pt, y),
                "acc_θ": (
                    float((Pt.argmax(axis=1) == y.astype(int)).mean())
                    if y.size
                    else float("nan")
                ),
                "flips": flip_rate(P0, Pt),
                "mean|Δ|": float(s.mean()) if s.size else float("nan"),
                "||g||": float(np.linalg.norm(g)),
            }
        )
    return rows_summary
