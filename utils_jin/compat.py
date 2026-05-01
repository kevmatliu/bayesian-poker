"""Compatibility helpers for the existing ``utils`` folder.

The legacy parser in ``utils/parse.py`` imports ``hand_map`` and ``action_map``
as top-level modules.  That works in notebooks that run from inside ``utils``,
but it does not work reliably when another package imports the parser from the
repository root.  Rather than edit the original parser immediately, the helper
below adds ``utils`` to ``sys.path`` before importing legacy modules.
"""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root assuming this file lives in ``utils_jin``."""

    return Path(__file__).resolve().parents[1]


def ensure_legacy_utils_path() -> Path:
    """Make the old ``utils`` directory importable and return its path."""

    utils_path = repo_root() / "utils"
    path_str = str(utils_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return utils_path


def load_legacy_session_class():
    """Import and return the original ``Session`` parser class."""

    ensure_legacy_utils_path()
    from parse import Session  # type: ignore

    return Session


def load_legacy_hand_mapper():
    """Import and return the original post-flop hand strength mapper."""

    ensure_legacy_utils_path()
    from hand_map import poker_hand_mapper  # type: ignore

    return poker_hand_mapper

