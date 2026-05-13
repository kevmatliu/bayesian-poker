"""Locate the repository root and default session list paths for evaluation notebooks."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents from ``start`` (default: cwd) until ``runners/common.py`` is found."""
    cur = (start or Path.cwd()).resolve()         # absolute starting directory
    for p in (cur, *cur.parents):                 # check cwd then every ancestor up to filesystem root
        if (p / "runners" / "common.py").is_file():  # sentinel file marking repo root
            return p                              # first match wins
    raise FileNotFoundError(                      # explicit failure instead of silent wrong root
        "Could not find runners/common.py; run the notebook from inside the repo "
        "or pass an explicit repo root."
    )


def _first_existing(repo: Path, relative_candidates: tuple[str, ...]) -> Path | None:
    for rel in relative_candidates:  # try legacy vs split-folder layouts in order
        p = repo / rel               # candidate absolute path
        if p.is_file():              # must be a real file, not a dangling symlink
            return p                 # stop at first hit
    return None                      # caller decides how to handle missing lists


def resolve_session_train_path(repo: Path) -> Path:
    p = _first_existing(  # prefer flat file then nested split directory
        repo,
        ("sessions_train.txt", "sessions_split/sessions_train.txt"),
    )
    if p is None:         # neither layout present
        raise FileNotFoundError(
            f"No sessions_train.txt under {repo} (tried sessions_train.txt, "
            "sessions_split/sessions_train.txt)."
        )
    return p              # training session id list for supervised prior fitting


def resolve_session_theta_path(repo: Path) -> Path:
    p = _first_existing(  # theta / tendency estimation split
        repo,
        ("sessions_theta.txt", "sessions_split/sessions_theta.txt"),
    )
    if p is None:
        raise FileNotFoundError(
            f"No sessions_theta.txt under {repo} (tried sessions_theta.txt, "
            "sessions_split/sessions_theta.txt)."
        )
    return p


def resolve_session_eval_path(repo: Path) -> Path | None:
    """Optional explicit eval list (``sessions_eval.txt``)."""
    return _first_existing(  # may legitimately be absent; caller falls back to filter list
        repo,
        ("sessions_eval.txt", "sessions_split/sessions_eval.txt"),
    )


def resolve_session_filter_path(repo: Path) -> Path | None:
    """Filter / online-holdout session list used when no ``sessions_eval`` file exists."""
    return _first_existing(  # online range pipeline export list
        repo,
        ("sessions_filter.txt", "sessions_split/sessions_filter.txt"),
    )


def default_global_priors_eval_session_paths(repo: Path) -> tuple[Path, Path, Path]:
    """
    Return ``(train_file, eval_union_component, theta_file)`` for global-prior supervised rows.

    Held-out session names are ``sorted(set(eval_names) | set(theta_names))``.
    If ``sessions_eval.txt`` is missing, ``sessions_filter.txt`` (or split dir) supplies
    the eval-side list so ``train | (theta ∪ filter)`` matches the common repo layout.
    """
    train = resolve_session_train_path(repo)  # always required
    theta = resolve_session_theta_path(repo)  # tendency split
    eval_p = resolve_session_eval_path(repo)  # preferred held-out list when present
    if eval_p is not None:                    # standard three-way split on disk
        return train, eval_p, theta
    filt = resolve_session_filter_path(repo)  # substitute eval list for online experiments
    if filt is None:                          # cannot define an eval side at all
        raise FileNotFoundError(
            "Need either sessions_eval.txt or sessions_filter.txt (or sessions_split/*)."
        )
    return train, filt, theta                 # train + filter-as-eval + theta
