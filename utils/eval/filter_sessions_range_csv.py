"""End-to-end evaluation driver for ``filter_sessions_range_history.csv``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.eval.online_csv import (
    add_calibration_columns,
    combo_probability_columns,
    enrich_online_range_dataframe,
)


def load_filter_sessions_range_csv(path: Path, *, low_memory: bool = False) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=low_memory)  # load full history table (optionally dtype-per-chunk)


def run_filter_sessions_range_evaluation(
    repo: Path,
    *,
    csv_path: Path | None = None,                                                      # override default artifact location
    pluribus_root: Path | None = None,                                                 # root folder containing parsed .phh trees
    strength_mc_samples: int = 100,                                                    # MC draws when scoring strength calibration
    rng_seed: int = 0,                                                                 # deterministic calibration noise
    max_rows: int | None = None,                                                       # optional subsample for quick dry runs
    verbose: bool = True,                                                              # print progress from downstream enrich/calib steps
    progress_every_enrich: int = 100,                                                  # row cadence for enrich logging
    progress_every_calib: int = 50,                                                    # row cadence for calibration logging
) -> dict[str, Any]:
    """
    Load the filter-session range CSV, enrich from Pluribus, add calibration columns.

    Returns a dict with ``df_raw``, ``df_enriched``, ``df_calib``, and simple column stats.
    """
    csv_path = csv_path or (repo / "artifacts" / "filter_sessions_range_history.csv")  # standard artifact path
    pluribus_root = pluribus_root or (repo / "pluribus")                               # default corpus location beside repo root
    df_raw = load_filter_sessions_range_csv(csv_path)                                  # baseline dataframe before joins
    if max_rows is not None:                                                           # optional head() for debugging / CI smoke tests
        df_raw = df_raw.iloc[: int(max_rows)].copy()                                   # copy so later steps do not mutate a view
    df_enriched = enrich_online_range_dataframe(                                       # attach parsed-hand columns used by metrics
        df_raw,
        pluribus_root,
        verbose=verbose,
        progress_every=progress_every_enrich,
    )
    combo_cols = combo_probability_columns(df_enriched)                                # identify model output columns to score
    rng = np.random.default_rng(rng_seed)                                              # reproducible stochastic strength checks
    df_calib = add_calibration_columns(                                                # append per-row calibration diagnostics
        df_enriched,
        combo_cols,
        strength_mc_samples=strength_mc_samples,
        strength_rng=rng,
        verbose=verbose,
        progress_every=progress_every_calib,
    )
    n_combo_prob_cols = len(combo_probability_columns(df_calib))                       # recount after new columns may appear
    return {
        "csv_path": csv_path,                                                          # echo resolved input path
        "n_rows": len(df_calib),                                                       # final row count after filtering inside enrich
        "n_combo_prob_cols": n_combo_prob_cols,                                        # how many combo logits/probs were evaluated
        "streets": df_calib["street"].value_counts(dropna=False).to_dict()             # street mix for sanity checks
        if "street" in df_calib.columns
        else {},
        "df_raw": df_raw,                                                              # untouched (or truncated) ingest copy
        "df_enriched": df_enriched,                                                    # post-join feature table
        "df_calib": df_calib,                                                          # final evaluation-ready frame
    }
