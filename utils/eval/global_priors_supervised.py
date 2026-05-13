"""Build or load cached supervised tensors for ``global_priors.json`` evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from runners.common import (
    collect_postflop_supervised_rows,
    collect_preflop_supervised_rows,
    flatten_hands,
    read_session_names_file,
)


PrintFn = Callable[[str], None]  # simple logger signature used throughout


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()  # stable hash of a small text file


@dataclass(frozen=True)
class GlobalPriorsSupervisedBundle:
    """Train + held-out supervised design matrices for the three action heads."""

    X_pre_train: np.ndarray  # preflop train design matrix
    y_pre_train: np.ndarray  # preflop train labels
    Xf_train: np.ndarray  # postflop facing-bet train features
    yf_train: np.ndarray  # postflop facing-bet train labels
    Xn_train: np.ndarray  # postflop no-bet train features
    yn_train: np.ndarray  # postflop no-bet train labels
    X_pre_test: np.ndarray  # preflop held-out design matrix
    y_pre_test: np.ndarray  # preflop held-out labels
    Xf_test: np.ndarray  # postflop facing held-out features
    yf_test: np.ndarray  # postflop facing held-out labels
    Xn_test: np.ndarray  # postflop no-bet held-out features
    yn_test: np.ndarray  # postflop no-bet held-out labels
    cache_meta: dict  # metadata dict used to validate cached npz
    loaded_from_cache: bool  # whether tensors came from disk vs fresh build
    cache_path: Path  # path to the supervised npz artifact


def default_supervised_cache_path(repo: Path) -> Path:
    return repo / "artifacts" / "global_priors_eval_supervised.npz"  # conventional cache location


def load_or_build_global_priors_supervised(
    repo: Path,
    *,
    train_file: Path,
    eval_component_file: Path,
    theta_file: Path,
    cache_path: Path | None = None,
    pluribus_root: Path | None = None,
    postflop_equity_mc: int = 8,
    force_rebuild: bool = False,
    print_fn: PrintFn | None = print,
) -> GlobalPriorsSupervisedBundle:
    """
    Load ``artifacts/global_priors_eval_supervised.npz`` when metadata matches, else build.

    Held-out hands are the union of sessions listed in ``eval_component_file`` and
    ``theta_file`` (same convention as the rooted ``global_priors_evaluation`` notebook).
    """
    cache_path = cache_path or default_supervised_cache_path(repo)                         # resolve default npz path
    pluribus_root = pluribus_root or (repo / "pluribus")                                   # default pluribus hand root
    log = print_fn or (lambda _s: None)                                                    # no-op logger when print suppressed

    train_s = read_session_names_file(train_file)                                          # training session names
    eval_s = read_session_names_file(eval_component_file)                                  # eval-side session names
    theta_s = read_session_names_file(theta_file)                                          # theta-json session names
    held_names = tuple(sorted(set(eval_s) | set(theta_s)))                                 # unique union for held-out split

    cache_meta = {
        "sessions_train_sha256": _sha256_file(train_file),                                 # invalidate if train list changes
        "sessions_eval_component_sha256": _sha256_file(eval_component_file),               # track eval list edits
        "sessions_theta_sha256": _sha256_file(theta_file),                                 # track theta list edits
        "held_session_names_union": list(held_names),                                      # explicit held-out set for debugging
        "postflop_equity_mc": int(postflop_equity_mc),                                     # tie postflop feature gen settings to cache
    }

    def gather(refs):
        Xpre, ypre = collect_preflop_supervised_rows(refs)  # preflop rows from hand refs
        Xf, yf, Xn, yn = collect_postflop_supervised_rows(
            refs, postflop_equity_mc=postflop_equity_mc     # postflop rows with chosen mc equity depth
        )
        return Xpre, ypre, Xf, yf, Xn, yn                   # single tuple for train or held-out gather

    def cache_ok(z: np.lib.npyio.NpzFile) -> bool:
        try:
            got = json.loads(str(z["cache_meta_json"].item()))  # parse embedded json metadata
        except Exception:
            return False                                        # corrupt or missing meta means unsafe to trust cache
        return got == cache_meta                                # require exact metadata match for reuse

    loaded = False                                                                         # flag set when cache successfully hydrates bundle
    if cache_path.is_file() and not force_rebuild:                                         # attempt cache read when allowed
        z = np.load(cache_path, allow_pickle=False)                                        # mmap-friendly npz reader
        try:
            if cache_ok(z):                                                                # only accept cache when fingerprints align
                bundle = GlobalPriorsSupervisedBundle(
                    X_pre_train=z["X_pre_train"],                                          # train tensors from npz
                    y_pre_train=z["y_pre_train"],
                    Xf_train=z["Xf_train"],
                    yf_train=z["yf_train"],
                    Xn_train=z["Xn_train"],
                    yn_train=z["yn_train"],
                    X_pre_test=z["X_pre_test"],                                            # held-out tensors from npz
                    y_pre_test=z["y_pre_test"],
                    Xf_test=z["Xf_test"],
                    yf_test=z["yf_test"],
                    Xn_test=z["Xn_test"],
                    yn_test=z["yn_test"],
                    cache_meta=cache_meta,                                                 # attach current meta even when loaded
                    loaded_from_cache=True,                                                # mark provenance for callers
                    cache_path=cache_path,                                                 # record artifact path on bundle
                )
                loaded = True                                                              # skip rebuild path
        finally:
            z.close()                                                                      # release file handles promptly

    if loaded:                                                                             # fast return on valid cache hit
        log(
            f"session lists — train: {len(train_s)} | eval-side: {len(eval_s)} | "         # human-readable split sizes
            f"theta: {len(theta_s)} | held-out union: {len(held_names)}"
        )
        log(f"Loaded supervised rows from {cache_path.relative_to(repo)}")                 # confirm relative cache path
        log(
            f"  train: preflop n={len(bundle.y_pre_train)} | "                             # train row counts per head
            f"post facing={len(bundle.yf_train)} no_bet={len(bundle.yn_train)}"
        )
        log(
            f"  held-out: preflop n={len(bundle.y_pre_test)} | "                           # held-out row counts per head
            f"post facing={len(bundle.yf_test)} no_bet={len(bundle.yn_test)}"
        )
        return bundle                                                                      # done when cache satisfied

    if cache_path.is_file() and not force_rebuild:                                         # cache exists but failed validation
        log("Supervised cache present but metadata mismatch — rebuilding…")                # explain stale npz
    else:
        log("Gathering supervised rows (slow; saved to npz for next run)…")                # first build or forced rebuild

    cache_path.parent.mkdir(parents=True, exist_ok=True)                                   # ensure artifacts dir exists
    train_inputs = [pluribus_root / s for s in train_s]                                    # per-session directories for training
    held_inputs = [pluribus_root / s for s in held_names]                                  # directories for held-out sessions
    train_refs = flatten_hands(train_inputs)                                               # expand to hand-level refs for training
    heldout_refs = flatten_hands(held_inputs)                                              # expand held-out hands
    log(f"train hands: {len(train_refs)} | held-out hands: {len(heldout_refs)}")           # log hand counts

    X_pre_train, y_pre_train, Xf_train, yf_train, Xn_train, yn_train = gather(train_refs)  # build train matrices
    X_pre_test, y_pre_test, Xf_test, yf_test, Xn_test, yn_test = gather(heldout_refs)      # build held-out matrices

    meta_json = np.empty((), dtype=object)                                                 # 0-d object array slot for json string
    meta_json[()] = json.dumps(cache_meta, sort_keys=True)                                 # deterministic json for npz equality
    np.savez_compressed(
        cache_path,                                                                        # atomic-ish write target for supervised bundle
        cache_meta_json=meta_json,                                                         # embed validation metadata
        X_pre_train=X_pre_train,                                                           # persist train block
        y_pre_train=y_pre_train,
        Xf_train=Xf_train,
        yf_train=yf_train,
        Xn_train=Xn_train,
        yn_train=yn_train,
        X_pre_test=X_pre_test,                                                             # persist held-out block
        y_pre_test=y_pre_test,
        Xf_test=Xf_test,
        yf_test=yf_test,
        Xn_test=Xn_test,
        yn_test=yn_test,
    )
    log(f"Wrote {cache_path.relative_to(repo)}")                                           # confirm write location
    return GlobalPriorsSupervisedBundle(
        X_pre_train=X_pre_train,                                                           # return freshly built in-memory bundle
        y_pre_train=y_pre_train,
        Xf_train=Xf_train,
        yf_train=yf_train,
        Xn_train=Xn_train,
        yn_train=yn_train,
        X_pre_test=X_pre_test,
        y_pre_test=y_pre_test,
        Xf_test=Xf_test,
        yf_test=yf_test,
        Xn_test=Xn_test,
        yn_test=yn_test,
        cache_meta=cache_meta,                                                             # echo meta for consumers
        loaded_from_cache=False,                                                           # mark rebuild path
        cache_path=cache_path,                                                             # path just written
    )
