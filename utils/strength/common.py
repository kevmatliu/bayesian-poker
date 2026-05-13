"""Strength feature mode only; card primitives live in :mod:`utils.parse`."""

from __future__ import annotations

from enum import Enum


class StrengthMode(Enum):
    """Which strength feature family applies."""

    PREFLOP = "preflop"  # abstract 169-class strength before community cards
    POSTFLOP = "postflop"  # board-aware strength after at least the flop
