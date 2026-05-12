"""Evaluation helpers: range calibration (Brier) and strength distributions."""

from utils.eval.logutil import eval_log

from utils.eval.brier import (
    brier_postflop1326,
    brier_preflop169,
    brier_preflop_from_combo1326,
    collapse_combo_distribution_to_169,
    class_distribution_to_vector,
    multiclass_brier,
)
from utils.eval.strength import (
    actual_made_and_draw,
    actual_made_percentile_at_combo_index,
    expected_made_and_draw,
    expected_made_and_draw_mc,
    expected_made_mean_and_histogram_mode,
    made_percentile_calibration_stats,
    made_percentile_vector_1326,
    plot_made_percentile_weighted_histogram,
)
from utils.eval.online_csv import (
    MADE_STRENGTH_PCT_COLUMN_PREFIX,
    META_COLUMNS,
    add_calibration_columns,
    combo_made_percentile_column_names,
    combo_probability_columns,
    enrich_online_range_dataframe,
    load_hand_pluribus,
    meta_columns_present,
)
from utils.eval.table import (
    board_at_street_end,
    hand_number_from_path,
    seat_columns,
    session_phh_files,
)

__all__ = [
    "MADE_STRENGTH_PCT_COLUMN_PREFIX",
    "META_COLUMNS",
    "eval_log",
    "actual_made_and_draw",
    "actual_made_percentile_at_combo_index",
    "add_calibration_columns",
    "board_at_street_end",
    "brier_postflop1326",
    "brier_preflop169",
    "brier_preflop_from_combo1326",
    "class_distribution_to_vector",
    "collapse_combo_distribution_to_169",
    "combo_made_percentile_column_names",
    "combo_probability_columns",
    "enrich_online_range_dataframe",
    "expected_made_and_draw",
    "expected_made_and_draw_mc",
    "expected_made_mean_and_histogram_mode",
    "made_percentile_calibration_stats",
    "made_percentile_vector_1326",
    "plot_made_percentile_weighted_histogram",
    "hand_number_from_path",
    "load_hand_pluribus",
    "meta_columns_present",
    "multiclass_brier",
    "seat_columns",
    "session_phh_files",
]
