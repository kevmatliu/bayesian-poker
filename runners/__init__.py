"""CLI entrypoints for filtering, training, and per-player theta (see ``command_*.sh``)."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_sys_path() -> Path:
    """Insert ``<repo>/utils`` then ``<repo>`` at the front of ``sys.path``."""
    root = Path(__file__).resolve().parents[1]
    utils_dir = root / "utils"
    for path in (str(utils_dir), str(root)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    return root


ensure_sys_path()
