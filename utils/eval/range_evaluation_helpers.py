"""Preflop range posteriors and ``filter_sessions_range_history.csv`` evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from runners.common import hole_cards_to_hand_class, preflop_decisions_for_hand
from utils.action.preflop import PreflopActionModel, PreflopPrior
from utils.em.preflop import PreflopEMDecision, PreflopEMHandBundle, e_step_hand_class_posterior
from utils.eval.common import em_and_online_refs
from utils.eval.online_csv import (
    add_calibration_columns,
    combo_probability_columns,
    enrich_online_range_dataframe,
    load_hand_pluribus,
)
from utils.eval.strength import actual_made_and_draw
from utils.eval.table import player_alive_at_street_end
from utils.filter.common import initial_class_prior, normalize
from utils.parse import Hand

EPS = 1e-12

_POSTFLOP_STREETS: tuple[str, ...] = ("flop", "turn", "river")
_STREET_PRIOR: dict[str, str] = {"flop": "pre-flop", "turn": "flop", "river": "turn"}


def range_csv_theta_tag(csv_path: Path) -> Optional[str]:
    """Parse ``em`` / ``newton`` / … from ``filter_sessions_range_history_<tag>.csv``."""
    name = csv_path.name
    prefix, suffix = "filter_sessions_range_history_", ".csv"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    mid = name[len(prefix) : -len(suffix)]
    return mid or None


def combo_prob_fingerprint(
    df: pd.DataFrame, combo_cols: list[str], *, n_rows: int = 64
) -> str:
    """Stable hash of the first ``n_rows`` of combo probability columns (detect identical CSVs)."""
    if df.empty or not combo_cols:
        return ""
    n = min(int(n_rows), len(df))
    mat = df.loc[:, combo_cols].iloc[:n].to_numpy(dtype=np.float64, copy=False)
    mat = np.ascontiguousarray(mat)
    return hashlib.sha256(mat.tobytes()).hexdigest()

# Order for Part A “target still in through street S” (each street must exist on the hand).
_PART_A_STREET_CHAIN: tuple[str, ...] = ("pre-flop", "flop", "turn", "river")


def target_alive_through_street_end(hand: Hand, player: str, last_street: str) -> bool:
    """True iff ``hand`` reached each street up to ``last_street`` and ``player`` is alive at each street end."""
    if last_street not in _PART_A_STREET_CHAIN:
        raise ValueError(
            f"last_street must be one of {_PART_A_STREET_CHAIN}, got {last_street!r}"
        )
    end_i = _PART_A_STREET_CHAIN.index(last_street)
    for s in _PART_A_STREET_CHAIN[: end_i + 1]:
        sts = hand.states.get(s) or []
        if not sts:
            return False
        if not player_alive_at_street_end(hand, s, player):
            return False
    return True


def em_and_online_hand_refs(repo: Path, *, pluribus_root: Path | None = None):
    """Same hand refs as ``em_and_online_refs`` but without returning session id lists."""
    em_refs, online_refs, _, _ = em_and_online_refs(repo, pluribus_root=pluribus_root)
    return em_refs, online_refs


def build_bundles_with_truth(
    refs,
    target: str,
    observer: str,
    *,
    require_target_in_hand_through: Optional[str] = None,
) -> list[tuple[PreflopEMHandBundle, str, dict]]:
    """Build preflop EM bundles with realized 169-class labels.

    If ``require_target_in_hand_through`` is set to ``\"pre-flop\"`` / ``\"flop\"`` / …, only
    hands where ``target`` is still in at the **end** of that street and every earlier street
    in the chain was reached (used by Part A NLL / rank cells so metrics ignore hands where the
    target is already out on those streets).
    """
    rows = []
    for ref in refs:
        names = ref.hand.player_names
        if target not in names or observer not in names:
            continue
        if require_target_in_hand_through:
            if not target_alive_through_street_end(
                ref.hand, target, require_target_in_hand_through
            ):
                continue
        true_class = hole_cards_to_hand_class(ref.hand.hole_cards.get(target, "") or "")
        if true_class is None:
            continue
        decisions = preflop_decisions_for_hand(ref.hand, target, ref.global_index)
        if not decisions:
            continue
        dead = ref.hand.hole_cards.get(observer, "") or ""
        initial_range = normalize(initial_class_prior(dead_cards=dead))
        bundle = PreflopEMHandBundle(
            tuple(PreflopEMDecision(d.state_key, d.action_bucket) for d in decisions),
            initial_range,
        )
        rows.append(
            (
                bundle,
                true_class,
                {"global_index": ref.global_index, "n_decisions": len(decisions)},
            )
        )
    return rows


def run_estep(rows, prior):
    return [
        (true_class, e_step_hand_class_posterior(bundle, prior))
        for bundle, true_class, _meta in rows
    ]


def prior_only_results(rows):
    return [(true_class, dict(bundle.initial_range)) for bundle, true_class, _meta in rows]


def compute_all_posterior_results(
    players: list[str],
    observer_for_target: dict[str, str],
    *,
    beta_pre: np.ndarray,
    theta_pre: dict[str, tuple],
    em_refs,
    online_refs,
    require_target_in_hand_through: Optional[str] = "flop",
) -> dict[tuple[str, str, str], list[tuple[str, dict]]]:
    """Keys ``(player, split, predictor)`` with split ``em``/``online``, predictor three-way.

    ``require_target_in_hand_through`` controls Part A hand inclusion (see
    :func:`build_bundles_with_truth`). Use ``None`` to include all hands that pass the usual
    hole-card / preflop-decision checks (legacy behaviour).
    """
    data: dict[tuple[str, str], list] = {}
    for player in players:
        observer = observer_for_target[player]
        kw = {"require_target_in_hand_through": require_target_in_hand_through}
        data[(player, "em")] = build_bundles_with_truth(em_refs, player, observer, **kw)
        data[(player, "online")] = build_bundles_with_truth(online_refs, player, observer, **kw)

    results: dict[tuple[str, str, str], list[tuple[str, dict]]] = {}
    for player in players:
        prior_pop = PreflopActionModel(
            PreflopPrior(beta_preflop=beta_pre), (0.0, 0.0, 0.0)
        )
        prior_pl = PreflopActionModel(
            PreflopPrior(beta_preflop=beta_pre), tuple(theta_pre[player])
        )
        for split in ("em", "online"):
            rows = data[(player, split)]
            results[(player, split, "prior_only")] = prior_only_results(rows)
            results[(player, split, "population")] = run_estep(rows, prior_pop)
            results[(player, split, "player")] = run_estep(rows, prior_pl)
    return results


def true_hand_nll(results: list[tuple[str, dict]]) -> float:
    if not results:
        return float("nan")
    nlls = []
    for true_class, q in results:
        p = float(q.get(true_class, 0.0))
        nlls.append(-np.log(max(p, EPS)))
    return float(np.mean(nlls))


def ranks_from_results(
    results: list[tuple[str, dict]], *, all_169: list[str], index_of: dict[str, int]
) -> np.ndarray:
    out = []
    for true_class, q in results:
        p = np.array([q.get(h, 0.0) for h in all_169], dtype=float)
        order = np.argsort(-p, kind="stable")
        rank = int(np.where(order == index_of[true_class])[0][0]) + 1
        out.append(rank)
    return np.asarray(out, dtype=int)


def default_observer_map(players: list[str]) -> dict[str, str]:
    return {p: players[(i + 1) % len(players)] for i, p in enumerate(players)}


def load_filter_sessions_range_csv(path: Path, *, low_memory: bool = False) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=low_memory)


def observer_folded_this_postflop_street(hand: Hand, street: str, observer: str) -> bool:
    """True iff ``observer`` was in at the end of the prior street but out at the end of ``street``."""
    prior = _STREET_PRIOR.get(street)
    if prior is None:
        return False
    return player_alive_at_street_end(hand, prior, observer) and not player_alive_at_street_end(
        hand, street, observer
    )


def both_players_alive_at_street_end(hand: Hand, street: str, a: str, b: str) -> bool:
    return player_alive_at_street_end(hand, street, a) and player_alive_at_street_end(hand, street, b)


def evaluate_observer_fold_vs_target_realized_strength(
    df: pd.DataFrame,
    pluribus_root: Path,
    *,
    streets: Iterable[str] = _POSTFLOP_STREETS,
    require_target_alive_at_street_end: bool = True,
    return_detail: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Empirical fold-vs-strength check when observer and target were both in, then observer folds.

    Each CSV row is an observer→target range snapshot. This routine replays the linked ``.phh`` and
    selects **postflop** rows where, at the **start** of the row's ``street``, both players were
    still in, and the **observer** folded on that same ``street`` (alive at the prior street's end,
    not alive at this street's end). For those rows it compares **realized** made strength percentile
    and draw heuristic (same definitions as ``actual_made_pct`` / ``actual_draw`` in
    :func:`utils.eval.online_csv.add_calibration_columns`) for observer vs target on the row's
    board.

    Interpretation (made or draw, higher = stronger):

    * **Consistent fold**: observer's realized value is **below** the target's (folded the weaker
      show-down value on that axis).
    * **Inconsistent fold**: observer's value is **above** the target's (folded while ahead on that
      axis). This is a crude sanity check only; it ignores action, stack depth, and future cards.

    Requires enriched columns ``community_cards`` and ``target_hole_cards`` (from
    :func:`utils.eval.online_csv.enrich_online_range_dataframe`). Optional
    ``target_still_in_hand`` is used when ``require_target_alive_at_street_end`` is true
    (same semantics as :func:`enrich_online_range_dataframe`).

    Returns a dict with counts/rates and optionally a per-row ``detail`` :class:`pandas.DataFrame`.
    """
    need = {"session", "hand_number", "street", "observer", "target", "community_cards"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"evaluate_observer_fold_vs_target_realized_strength: missing columns {sorted(miss)}")

    street_set = {str(s) for s in streets}
    cache: dict[tuple[str, int | str], Hand | None] = {}

    def _hn_key(hn: Any) -> int | str:
        if isinstance(hn, int):
            return hn
        s = str(hn)
        return int(s) if s.isdigit() else hn

    def get_hand(sess: str, hn: Any) -> Hand | None:
        key = (str(sess), _hn_key(hn))
        if key not in cache:
            cache[key] = load_hand_pluribus(pluribus_root, str(sess), hn)
        return cache[key]

    rows_out: list[dict[str, Any]] = []
    n_seen_postflop = 0

    for _, row in df.iterrows():
        street = str(row["street"])
        if street not in street_set:
            continue
        n_seen_postflop += 1
        if require_target_alive_at_street_end and "target_still_in_hand" in df.columns:
            if not bool(row["target_still_in_hand"]):
                continue

        hand = get_hand(str(row["session"]), row["hand_number"])
        if hand is None:
            continue

        observer = str(row["observer"])
        target = str(row["target"])
        if observer not in hand.player_names or target not in hand.player_names:
            continue

        prior = _STREET_PRIOR.get(street)
        if prior is None:
            continue

        if not both_players_alive_at_street_end(hand, prior, observer, target):
            continue
        if not observer_folded_this_postflop_street(hand, street, observer):
            continue

        board = str(row.get("community_cards", "") or "")
        obs_h = str(hand.hole_cards.get(observer, "") or "")
        tgt_h = str(hand.hole_cards.get(target, "") or "")
        tgt_h_csv = str(row.get("target_hole_cards", "") or "")
        if len(tgt_h_csv) == 4:
            tgt_h = tgt_h_csv

        o_made, o_draw = actual_made_and_draw(obs_h, board)
        t_made, t_draw = actual_made_and_draw(tgt_h, board)
        if not (np.isfinite(o_made) and np.isfinite(t_made)):
            continue

        made_consistent = o_made < t_made
        made_inconsistent = o_made > t_made
        made_tie = o_made == t_made

        draw_ok = np.isfinite(o_draw) and np.isfinite(t_draw)
        draw_consistent = float(np.nan) if not draw_ok else float(o_draw < t_draw)
        draw_inconsistent = float(np.nan) if not draw_ok else float(o_draw > t_draw)
        draw_tie = float(np.nan) if not draw_ok else float(o_draw == t_draw)

        rows_out.append(
            {
                "session": row["session"],
                "hand_number": row["hand_number"],
                "street": street,
                "observer": observer,
                "target": target,
                "observer_made_pct": o_made,
                "target_made_pct": t_made,
                "observer_draw": o_draw,
                "target_draw": t_draw,
                "made_fold_consistent": made_consistent,
                "made_fold_inconsistent": made_inconsistent,
                "made_tie": made_tie,
                "draw_fold_consistent": draw_consistent,
                "draw_fold_inconsistent": draw_inconsistent,
                "draw_tie": draw_tie,
                "draw_defined": draw_ok,
            }
        )

    if not rows_out:
        summary = {
            "n_csv_rows_postflop_street": n_seen_postflop,
            "n_eligible_observer_folds": 0,
            "made_consistent": 0,
            "made_inconsistent": 0,
            "made_tie": 0,
            "made_consistent_rate_excl_ties": float("nan"),
            "draw_consistent": 0,
            "draw_inconsistent": 0,
            "draw_tie": 0,
            "draw_consistent_rate_excl_ties": float("nan"),
            "n_draw_defined": 0,
        }
        return {"summary": summary, "detail": pd.DataFrame() if return_detail else None}

    d = pd.DataFrame(rows_out)
    n = len(d)
    m_ties = int(d["made_tie"].sum())
    d_mask = d["draw_defined"].astype(bool)
    d_m = int(d_mask.sum())
    m_exc = n - m_ties
    d_ties = int((d.loc[d_mask, "draw_tie"] == 1.0).sum()) if d_m else 0
    d_exc = d_m - d_ties

    made_cons = int(d["made_fold_consistent"].sum())
    made_inc = int(d["made_fold_inconsistent"].sum())
    dr_cons = int((d.loc[d_mask, "draw_fold_consistent"] == 1.0).sum())
    dr_inc = int((d.loc[d_mask, "draw_fold_inconsistent"] == 1.0).sum())
    summary = {
        "n_csv_rows_postflop_street": n_seen_postflop,
        "n_eligible_observer_folds": n,
        "made_consistent": made_cons,
        "made_inconsistent": made_inc,
        "made_tie": m_ties,
        "made_consistent_rate_excl_ties": float(made_cons / m_exc) if m_exc else float("nan"),
        "draw_consistent": dr_cons,
        "draw_inconsistent": dr_inc,
        "draw_tie": d_ties,
        "draw_consistent_rate_excl_ties": float(dr_cons / d_exc) if d_exc else float("nan"),
        "n_draw_defined": d_m,
    }
    if verbose:
        print(summary)

    out: dict[str, Any] = {"summary": summary}
    if return_detail:
        out["detail"] = d
    return out


_RANGE_CALIB_SUMMARY_COLS: tuple[str, ...] = (
    "brier",
    "expected_made_pct",
    "expected_draw",
    "actual_made_pct",
    "actual_draw",
    "made_dist_mu",
    "made_dist_sigma",
    "made_pct_z",
    "made_pct_abs_z",
    "combo_nll",
    "made_pct_midrank",
    "made_pct_cdf_le",
)


def aggregate_range_calibration_row_stats(df: pd.DataFrame) -> dict[str, Any]:
    """NaN-ignoring means of standard calibration columns (one pass per row)."""
    out: dict[str, Any] = {"n_rows": int(len(df))}
    if "street" in df.columns:
        vc = df["street"].value_counts(dropna=False)
        out["n_by_street"] = {str(k): int(v) for k, v in vc.items()}
    for c in _RANGE_CALIB_SUMMARY_COLS:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float, copy=False)
        finite = np.isfinite(s)
        out[f"mean_{c}"] = float(np.nanmean(s)) if finite.any() else float("nan")
        out[f"n_finite_{c}"] = int(finite.sum())
    return out


def aggregate_range_calibration_by_hand(df: pd.DataFrame) -> dict[str, Any]:
    """Mean each metric within (session, hand_number, observer, target), then mean across hands."""
    keys = ["session", "hand_number", "observer", "target"]
    if not all(k in df.columns for k in keys) or len(df) == 0:
        return {"n_hands": 0, "n_rows": int(len(df)), "note": "missing keys or empty frame"}
    cols = [c for c in _RANGE_CALIB_SUMMARY_COLS if c in df.columns]
    if not cols:
        return {"n_hands": int(df.groupby(keys, dropna=False).ngroups), "n_rows": int(len(df)), "note": "no calibration cols"}
    g = df.groupby(keys, dropna=False)[cols].mean()
    out: dict[str, Any] = {"n_hands": int(len(g)), "n_rows": int(len(df))}
    for c in cols:
        v = g[c].to_numpy(dtype=float, copy=False)
        finite = np.isfinite(v)
        m = float(np.nanmean(v)) if finite.any() else float("nan")
        out[f"mean_across_hands_of_mean_{c}_per_hand"] = m
    return out


def run_filter_sessions_range_evaluation(
    repo: Path,
    *,
    csv_path: Path | None = None,
    theta_optimizer: Optional[str] = None,
    pluribus_root: Path | None = None,
    strength_mc_samples: int = 100,
    rng_seed: int = 0,
    max_rows: int | None = None,
    verbose: bool = True,
    progress_every_enrich: int = 100,
    progress_every_calib: int = 50,
    require_target_in_hand: bool = True,
) -> dict[str, Any]:
    """Load filter-session range CSV, enrich from Pluribus, add calibration columns.

    When ``require_target_in_hand`` is true (default), calibration and combo-range metrics
    run only on rows where ``target_still_in_hand`` is true (see
    :func:`enrich_online_range_dataframe`): ``target`` survived through the row's ``street``
    for pre-flop, and for postflop streets also entered that street still contesting (prior
    street end alive) and remained through the end of the row's ``street``. Rows are dropped
    before calibration; see return keys ``calibration_row_aggregates`` and
    ``calibration_hand_aggregates`` for pooled summaries.

    **EM vs Newton:** Row counts and ``n_rows_excluded_not_in_hand`` depend only on Pluribus
    replay (who is in on each street), so they match whenever the CSV has the same metadata rows.
    ``mean_brier`` / ``mean_combo_nll`` etc. should differ when combo columns differ; use
    ``combo_prob_fingerprint_raw`` and ``range_csv_theta_tag`` to confirm you loaded distinct
    files. If ``csv_path`` is omitted and ``theta_optimizer`` is ``\"em\"`` or ``\"newton\"``,
    the CSV path defaults to ``artifacts/filter_sessions_range_history_{tag}.csv``.
    """
    if csv_path is None:
        if theta_optimizer in ("em", "newton"):
            csv_path = repo / "artifacts" / f"filter_sessions_range_history_{theta_optimizer}.csv"
        else:
            csv_path = repo / "artifacts" / "filter_sessions_range_history.csv"
    else:
        csv_path = csv_path.expanduser().resolve()
    pluribus_root = pluribus_root or (repo / "pluribus")
    df_full = load_filter_sessions_range_csv(csv_path)
    combo_cols_full = combo_probability_columns(df_full)
    fp_raw = combo_prob_fingerprint(df_full, combo_cols_full)
    theta_tag = range_csv_theta_tag(csv_path)
    df_raw = df_full
    if max_rows is not None:
        df_raw = df_raw.iloc[: int(max_rows)].copy()
    df_enriched = enrich_online_range_dataframe(
        df_raw,
        pluribus_root,
        verbose=verbose,
        progress_every=progress_every_enrich,
    )
    combo_cols = combo_probability_columns(df_enriched)
    n_input = len(df_enriched)
    if require_target_in_hand and "target_still_in_hand" in df_enriched.columns:
        alive = df_enriched["target_still_in_hand"].astype(bool)
        df_for_calib = df_enriched.loc[alive].copy()
        n_excluded = int((~alive).sum())
    else:
        df_for_calib = df_enriched
        n_excluded = 0
    rng = np.random.default_rng(rng_seed)
    df_calib = add_calibration_columns(
        df_for_calib,
        combo_cols,
        strength_mc_samples=strength_mc_samples,
        strength_rng=rng,
        verbose=verbose,
        progress_every=progress_every_calib,
    )
    n_combo_prob_cols = len(combo_probability_columns(df_calib))
    row_agg = aggregate_range_calibration_row_stats(df_calib)
    hand_agg = aggregate_range_calibration_by_hand(df_calib)
    return {
        "csv_path": csv_path,
        "range_csv_theta_tag": theta_tag,
        "combo_prob_fingerprint_raw": fp_raw,
        "n_rows": len(df_calib),
        "n_rows_input": n_input,
        "n_rows_excluded_not_in_hand": n_excluded,
        "n_combo_prob_cols": n_combo_prob_cols,
        "streets": df_calib["street"].value_counts(dropna=False).to_dict()
        if "street" in df_calib.columns
        else {},
        "calibration_row_aggregates": row_agg,
        "calibration_hand_aggregates": hand_agg,
        "df_raw": df_raw,
        "df_enriched": df_enriched,
        "df_calib": df_calib,
    }
