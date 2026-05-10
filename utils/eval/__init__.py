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
    expected_made_and_draw,
    expected_made_and_draw_mc,
)
from utils.eval.online_csv import (
    META_COLUMNS,
    add_calibration_columns,
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
    "META_COLUMNS",
    "eval_log",
    "actual_made_and_draw",
    "add_calibration_columns",
    "board_at_street_end",
    "brier_postflop1326",
    "brier_preflop169",
    "brier_preflop_from_combo1326",
    "class_distribution_to_vector",
    "collapse_combo_distribution_to_169",
    "combo_probability_columns",
    "enrich_online_range_dataframe",
    "expected_made_and_draw",
    "expected_made_and_draw_mc",
    "hand_number_from_path",
    "load_hand_pluribus",
    "meta_columns_present",
    "multiclass_brier",
    "seat_columns",
    "session_phh_files",
]
