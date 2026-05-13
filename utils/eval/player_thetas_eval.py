"""Held-out action prediction metrics for per-player θ in ``player_thetas.json``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from runners.common import (
    flatten_hands,
    hole_cards_to_hand_class,
    preflop_decisions_for_hand,
    read_session_names_file,
)
from utils.action.preflop import (
    PreflopActionModel,
    PreflopPrior,
    canonical_preflop_action,
    FOLD as PRE_FOLD,
    CHECK_CALL as PRE_CALL,
    RAISE as PRE_RAISE,
)
from utils.action.postflop import (
    PostflopActionModel,
    PostflopPrior,
    FOLD as POST_FOLD,
    CALL as POST_CALL,
    RAISE as POST_RAISE,
)
from utils.eval.action_heads import mean_brier, nll
from utils.eval.repo_paths import resolve_session_filter_path, resolve_session_theta_path
from utils.postflop_runner_bridge import collect_postflop_observations_known_hole_cards

L2_PENALTY = 0.25  # shrinkage weight subtracted from held-out gradient norm proxy


def load_global_betas(priors: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B_pre = np.asarray(priors["preflop"]["beta_preflop"], dtype=float)  # global preflop softmax weights
    B_f = np.asarray(priors["postflop"]["beta_facing"], dtype=float)    # global facing-bet weights
    B_n = np.asarray(priors["postflop"]["beta_no_bet"], dtype=float)    # global no-bet head weights
    return B_pre, B_f, B_n                                              # tuple unpacked by callers building models


def load_player_thetas(payload: dict) -> tuple[list[str], dict[str, tuple], dict[str, tuple]]:
    players = list(payload["players"].keys())                                      # roster from json payload
    theta_pre = {p: tuple(payload["players"][p]["theta_pre"]) for p in players}    # per-player preflop offsets
    theta_post = {p: tuple(payload["players"][p]["theta_post"]) for p in players}  # per-player postflop offsets
    return players, theta_pre, theta_post                                          # parallel structures for evaluation loops


def em_and_online_refs(repo: Path, *, pluribus_root: Path | None = None):
    pluribus_root = pluribus_root or (repo / "pluribus")                   # default pluribus root
    em_s = read_session_names_file(resolve_session_theta_path(repo))       # em-fitted session list
    online_s = read_session_names_file(resolve_session_filter_path(repo))  # online eval session list
    if not online_s:                                                       # guard empty filter list
        raise FileNotFoundError("Online/filter session list is empty.")    # fail loudly: split undefined
    em_refs = flatten_hands([pluribus_root / s for s in em_s])             # em hand references
    online_refs = flatten_hands([pluribus_root / s for s in online_s])     # online hand references
    return em_refs, online_refs, em_s, online_s                            # return both refs and raw session names for logging


def collect_player_preflop(refs, player: str) -> list[tuple]:
    rows = []                                                                              # list of (hand_class, state_key, canonical_action) tuples
    for ref in refs:                                                                       # scan candidate hands
        if player not in ref.hand.player_names:                                            # player must be dealt in
            continue                                                                       # skip irrelevant hands
        hc = hole_cards_to_hand_class(ref.hand.hole_cards.get(player, "") or "")           # abstract starting hand
        if hc is None:                                                                     # missing/invalid hole cards
            continue                                                                       # cannot label preflop class
        for dec in preflop_decisions_for_hand(ref.hand, player, ref.global_index):         # each preflop decision point
            rows.append((hc, dec.state_key, canonical_preflop_action(dec.action_bucket)))  # supervised row
    return rows                                                                            # feed into softmax predictors


def collect_player_postflop(refs, player: str) -> tuple[list, list]:
    facing, no_bet = [], []                                                                       # separate lists for the two postflop heads
    for ref in refs:                                                                              # walk all candidate hands
        if player not in ref.hand.player_names:                                                   # require participation
            continue                                                                              # skip
        obs = collect_postflop_observations_known_hole_cards(ref.hand, player, ref.global_index)  # structured rows
        if obs is None:                                                                           # parsing failed or no postflop
            continue                                                                              # nothing to score
        for feat, action in obs.decisions:                                                        # each postflop decision with engineered features
            if feat.facing_bet:                                                                   # branch for facing-bet head dataset
                facing.append((feat, int(action)))                                                # store feature row and discrete label
            else:
                if int(action) == POST_FOLD:                                                      # folds omitted from no-bet modeling slice
                    continue                                                                      # keep no-bet rows pure call/raise
                no_bet.append((feat, int(action)))                                                # no-bet head supervised tuples
    return facing, no_bet                                                                         # return disjoint postflop datasets


def _to_array_3(probs_dict, idx_a, idx_b, idx_c):
    return np.array([probs_dict[idx_a], probs_dict[idx_b], probs_dict[idx_c]], dtype=float)  # stack three action probs


def predict_preflop(rows, prior):
    if not rows:                                                  # handle empty slice without shape errors
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)        # empty preds and labels
    P = np.zeros((len(rows), 3))                                  # rows × {fold,call,raise} prob matrix
    y = np.zeros(len(rows), dtype=int)                            # integer labels aligned with P rows
    for i, (hc, sk, a) in enumerate(rows):                        # iterate supervised tuples
        probs = prior.action_probs(hc, sk)                        # dict of action->prob for this information set
        P[i] = _to_array_3(probs, PRE_FOLD, PRE_CALL, PRE_RAISE)  # fixed column order for metrics
        y[i] = int(a)                                             # store realized bucket index
    return P, y                                                   # ready for nll/brier


def predict_postflop_facing(rows, prior):
    if not rows:                                                     # empty facing split
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)           # degenerate outputs
    P = np.zeros((len(rows), 3))                                     # fold/call/raise probabilities
    y = np.zeros(len(rows), dtype=int)                               # labels
    for i, (feat, a) in enumerate(rows):                             # each facing-bet decision
        probs = prior.action_probs(feat)                             # postflop softmax outputs
        P[i] = _to_array_3(probs, POST_FOLD, POST_CALL, POST_RAISE)  # align columns with global ids
        y[i] = int(a)                                                # realized postflop action
    return P, y                                                      # matrix form for vectorised metrics


def predict_postflop_no_bet(rows, prior):
    if not rows:                                            # empty no-bet slice
        return np.zeros((0, 2)), np.zeros((0,), dtype=int)  # two-way softmax only
    P = np.zeros((len(rows), 2))                            # call vs raise probabilities
    y = np.zeros(len(rows), dtype=int)                      # remapped binary labels
    for i, (feat, a) in enumerate(rows):                    # each no-bet decision
        probs = prior.action_probs(feat)                    # still returns full dict; we subset
        P[i, 0] = probs[POST_CALL]                          # call/check mass
        P[i, 1] = probs[POST_RAISE]                         # raise mass (folds already filtered out)
        y[i] = 0 if int(a) == POST_CALL else 1              # binary encoding for metrics
    return P, y                                             # two-column prob matrix


def make_priors_for_player(
    player: str,
    *,
    B_pre: np.ndarray,
    B_facing: np.ndarray,
    B_no_bet: np.ndarray,
    theta_pre: tuple,
    theta_post: tuple,
):
    pre0 = PreflopActionModel(PreflopPrior(beta_preflop=B_pre), (0.0, 0.0, 0.0))      # population-only preflop model
    preH = PreflopActionModel(PreflopPrior(beta_preflop=B_pre), tuple(theta_pre))     # personalized preflop model
    post0 = PostflopActionModel(
        PostflopPrior(beta_facing=B_facing, beta_no_bet=B_no_bet), (0.0, 0.0, 0.0)    # population postflop model
    )
    postH = PostflopActionModel(
        PostflopPrior(beta_facing=B_facing, beta_no_bet=B_no_bet), tuple(theta_post)  # personalized postflop offsets
    )
    return pre0, preH, post0, postH                                                   # pair of models for ablation vs player tilt


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
    """``PRED[(player, split, head)] = (P0, P_theta, y)`` with split in ``{\"em\",\"online\"}``."""
    data_pre = {
        (p, sp): collect_player_preflop(refs, p)                                      # memoize preflop rows per player/split
        for p in players                                                              # outer product over players
        for sp, refs in (("em", em_refs), ("online", online_refs))                    # and eval splits
    }
    data_post = {
        (p, sp): collect_player_postflop(refs, p)                                     # memoize postflop facing/no-bet rows
        for p in players                                                              # all players
        for sp, refs in (("em", em_refs), ("online", online_refs))                    # both splits
    }
    pred: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # predictions keyed by triple
    for player in players:                                                            # fill predictions player by player
        pre0, preH, post0, postH = make_priors_for_player(
            player,                                                                   # current subject
            B_pre=B_pre,                                                              # shared global preflop weights
            B_facing=B_facing,                                                        # shared facing weights
            B_no_bet=B_no_bet,                                                        # shared no-bet weights
            theta_pre=theta_pre[player],                                              # player-specific preflop offsets
            theta_post=theta_post[player],                                            # player-specific postflop offsets
        )
        for sp in ("em", "online"):                                                   # identical head loop for each split
            rows_pre = data_pre[(player, sp)]                                         # cached preflop rows
            P0, y = predict_preflop(rows_pre, pre0)                                   # baseline probs under θ=0
            Pt, _ = predict_preflop(rows_pre, preH)                                   # personalized probs (reuse labels)
            pred[(player, sp, "preflop")] = (P0, Pt, y)                               # store triple for metrics

            rows_f, rows_n = data_post[(player, sp)]                                  # unpack postflop caches
            P0, y = predict_postflop_facing(rows_f, post0)                            # facing baseline
            Pt, _ = predict_postflop_facing(rows_f, postH)                            # facing personalized
            pred[(player, sp, "facing")] = (P0, Pt, y)                                # record facing head preds

            P0, y = predict_postflop_no_bet(rows_n, post0)                            # no-bet baseline
            Pt, _ = predict_postflop_no_bet(rows_n, postH)                            # no-bet personalized
            pred[(player, sp, "no_bet")] = (P0, Pt, y)                                # record no-bet preds
    return pred                                                                       # full nested dict for summary and diagnostics


def flip_rate(P0: np.ndarray, Pt: np.ndarray) -> float:
    if P0.shape[0] == 0:                                           # empty batch guard
        return float("nan")                                        # undefined flip rate
    return float((P0.argmax(axis=1) != Pt.argmax(axis=1)).mean())  # fraction of label changes after personalization


def realised_shifts(P0: np.ndarray, Pt: np.ndarray, y: np.ndarray) -> np.ndarray:
    if P0.shape[0] == 0:                                            # no rows -> empty shifts
        return np.zeros(0)                                          # match expected empty semantics
    idx = np.arange(P0.shape[0])                                    # row indices for advanced indexing
    return np.abs(Pt[idx, y.astype(int)] - P0[idx, y.astype(int)])  # per-row prob movement on realized action


def heldout_gradient_norm_components(
    Pt: np.ndarray, y: np.ndarray, theta_hat: tuple, n_actions: int
) -> np.ndarray:
    """Supervised M-step-style gradient (degenerate q) minus L2, averaged over rows."""
    if Pt.shape[0] == 0:                                 # no data -> zero gradient vector
        return np.zeros(n_actions)                       # shape matches action dimensionality
    g = np.zeros(n_actions)                              # accumulate mean negative grad of loglik wrt logits proxy
    for i in range(Pt.shape[0]):                         # explicit loop for clarity (small N typical)
        e = np.zeros(n_actions)                          # one-hot realized action
        e[int(y[i])] = 1.0                               # mark observed class
        g += e - Pt[i]                                   # softmax cross-entropy grad contribution (up to scale)
    g /= Pt.shape[0]                                     # average over minibatch of hands
    g -= L2_PENALTY * np.asarray(theta_hat[:n_actions])  # subtract l2 prior gradient slice
    return g                                             # vector used for norm in summary table


def online_summary_rows(
    pred: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    players: list[str],
    theta_pre: dict[str, tuple],
    theta_post: dict[str, tuple],
) -> list[dict]:
    shift_arrays = {
        key: realised_shifts(P0, Pt, y)                                         # precompute abs prob shifts per key
        for key, (P0, Pt, y) in pred.items()                                    # over all player/split/head combos
    }
    rows_summary: list[dict] = []                                               # flat list of metric dicts for pandas/csv
    for (p, sp, head), (P0, Pt, y) in pred.items():                             # walk all prediction bundles
        if sp != "online":                                                      # user-facing summary focuses on online split only
            continue                                                            # skip em rows here
        s = shift_arrays[(p, sp, head)]                                         # aligned shift vector for this key
        if head == "preflop":                                                   # choose theta vector matching head width
            theta_hat, n_actions = theta_pre[p], 3                              # three-way preflop parameter count
        else:
            theta_hat, n_actions = theta_post[p], 3 if head == "facing" else 2  # postflop: 3 facing, 2 no-bet
        g = heldout_gradient_norm_components(Pt, y, theta_hat, n_actions)       # gradient proxy for regularization scale
        rows_summary.append(
            {
                "player": p,                                                    # subject name
                "head": head,                                                   # which action head evaluated
                "N": int(y.size),                                               # sample count as int for serialization
                "ΔNLL": nll(P0, y) - nll(Pt, y),                                # nll improvement from personalization
                "ΔBrier": mean_brier(P0, y) - mean_brier(Pt, y),                # brier improvement
                "acc_θ": (
                    float((Pt.argmax(axis=1) == y.astype(int)).mean())
                    if y.size
                    else float("nan")                                           # undefined acc on empty slice
                ),                                                              # personalized top-1 accuracy when labels exist
                "flips": flip_rate(P0, Pt),                                     # how often argmax changes
                "mean|Δ|": float(s.mean()) if s.size else float("nan"),         # mean absolute prob shift on truth
                "||g||": float(np.linalg.norm(g)),                              # euclidean norm of regularized grad proxy
            }
        )
    return rows_summary                                                         # ready for tabular display
