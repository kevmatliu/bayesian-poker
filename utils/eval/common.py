"""Shared paths, logging, and session → hand ref loading for evaluation."""

from __future__ import annotations

from pathlib import Path

from runners.common import flatten_hands, read_session_names_file


def eval_log(verbose: bool, msg: str, *, prefix: str = "[eval]") -> None:
    if verbose:
        print(f"{prefix} {msg}", flush=True)


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for p in (cur, *cur.parents):
        if (p / "runners" / "common.py").is_file():
            return p
    raise FileNotFoundError(
        "Could not find runners/common.py; run the notebook from inside the repo "
        "or pass an explicit repo root."
    )


def _first_existing(repo: Path, relative_candidates: tuple[str, ...]) -> Path | None:
    for rel in relative_candidates:
        p = repo / rel
        if p.is_file():
            return p
    return None


def resolve_session_train_path(repo: Path) -> Path:
    p = _first_existing(repo, ("sessions_train.txt", "sessions_split/sessions_train.txt"))
    if p is None:
        raise FileNotFoundError(
            f"No sessions_train.txt under {repo} (tried sessions_train.txt, "
            "sessions_split/sessions_train.txt)."
        )
    return p


def resolve_session_theta_path(repo: Path) -> Path:
    p = _first_existing(repo, ("sessions_theta.txt", "sessions_split/sessions_theta.txt"))
    if p is None:
        raise FileNotFoundError(
            f"No sessions_theta.txt under {repo} (tried sessions_theta.txt, "
            "sessions_split/sessions_theta.txt)."
        )
    return p


def resolve_session_eval_path(repo: Path) -> Path | None:
    return _first_existing(repo, ("sessions_eval.txt", "sessions_split/sessions_eval.txt"))


def resolve_session_filter_path(repo: Path) -> Path | None:
    return _first_existing(repo, ("sessions_filter.txt", "sessions_split/sessions_filter.txt"))


def default_global_priors_eval_session_paths(repo: Path) -> tuple[Path, Path, Path]:
    """
    ``(train_file, eval_union_component, theta_file)`` for global-prior supervised rows.

    Held-out sessions are ``sorted(set(eval_names) | set(theta_names))``.
    If ``sessions_eval.txt`` is missing, ``sessions_filter.txt`` supplies the eval-side list.
    """
    train = resolve_session_train_path(repo)
    theta = resolve_session_theta_path(repo)
    eval_p = resolve_session_eval_path(repo)
    if eval_p is not None:
        return train, eval_p, theta
    filt = resolve_session_filter_path(repo)
    if filt is None:
        raise FileNotFoundError(
            "Need either sessions_eval.txt or sessions_filter.txt (or sessions_split/*)."
        )
    return train, filt, theta


def em_and_online_refs(repo: Path, *, pluribus_root: Path | None = None):
    """
    Flattened hand refs for EM (``sessions_theta``) and online (``sessions_filter``) splits.

    Returns ``(em_refs, online_refs, em_session_names, online_session_names)``.
    """
    pluribus_root = pluribus_root or (repo / "pluribus")
    em_s = read_session_names_file(resolve_session_theta_path(repo))
    online_s = read_session_names_file(resolve_session_filter_path(repo))
    if not online_s:
        raise FileNotFoundError("Online/filter session list is empty.")
    em_refs = flatten_hands([pluribus_root / s for s in em_s])
    online_refs = flatten_hands([pluribus_root / s for s in online_s])
    return em_refs, online_refs, em_s, online_s
