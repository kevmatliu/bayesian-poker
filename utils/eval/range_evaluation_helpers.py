"""Preflop range posteriors and ``filter_sessions_range_history.csv`` evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
)
from utils.filter.common import initial_class_prior, normalize

EPS = 1e-12


def em_and_online_hand_refs(repo: Path, *, pluribus_root: Path | None = None):
    """Same hand refs as ``em_and_online_refs`` but without returning session id lists."""
    em_refs, online_refs, _, _ = em_and_online_refs(repo, pluribus_root=pluribus_root)
    return em_refs, online_refs


def build_bundles_with_truth(
    refs,
    target: str,
    observer: str,
) -> list[tuple[PreflopEMHandBundle, str, dict]]:
    rows = []
    for ref in refs:
        names = ref.hand.player_names
        if target not in names or observer not in names:
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
) -> dict[tuple[str, str, str], list[tuple[str, dict]]]:
    """Keys ``(player, split, predictor)`` with split ``em``/``online``, predictor three-way."""
    data: dict[tuple[str, str], list] = {}
    for player in players:
        observer = observer_for_target[player]
        data[(player, "em")] = build_bundles_with_truth(em_refs, player, observer)
        data[(player, "online")] = build_bundles_with_truth(online_refs, player, observer)

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


def run_filter_sessions_range_evaluation(
    repo: Path,
    *,
    csv_path: Path | None = None,
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
    run only on rows where ``target`` is still in the pot at the **end** of that row's
    ``street`` (see ``target_still_in_hand`` from :func:`enrich_online_range_dataframe`).
    Rows where the target folded on an earlier street are dropped before calibration.
    """
    csv_path = csv_path or (repo / "artifacts" / "filter_sessions_range_history.csv")
    pluribus_root = pluribus_root or (repo / "pluribus")
    df_raw = load_filter_sessions_range_csv(csv_path)
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
    return {
        "csv_path": csv_path,
        "n_rows": len(df_calib),
        "n_rows_input": n_input,
        "n_rows_excluded_not_in_hand": n_excluded,
        "n_combo_prob_cols": n_combo_prob_cols,
        "streets": df_calib["street"].value_counts(dropna=False).to_dict()
        if "street" in df_calib.columns
        else {},
        "df_raw": df_raw,
        "df_enriched": df_enriched,
        "df_calib": df_calib,
    }
