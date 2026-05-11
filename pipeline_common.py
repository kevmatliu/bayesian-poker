"""Shared helpers for train, find_theta, tests, and runner-style pipelines.

This module is the **data plane** between parsed Pluribus ``.phh`` hands and
the statistical objects used elsewhere:

- **HandRef**: stable `(source_path, global_index, Hand)` references.
- **flatten_hands**: resolve CLI paths (file / session / pluribus root) and
  parse ``.phh`` in parallel.
- **collect_*_supervised_rows**: build design matrices for population baseline
  training when hole cards are known.
- **gather_*_bundles_for_target_player**: build EM inputs when inferring
  ``theta`` from observed actions and an observer's dead cards.

Data layout: a *Pluribus root* (e.g. ``pluribus/``) contains one subdirectory
per *session*; each session folder holds numbered ``.phh`` files, one per hand.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

LOG_PC = logging.getLogger(__name__)
for path in (str(REPO_ROOT), str(REPO_ROOT / "utils")):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.em import PostflopEMHandBundle, PreflopEMDecision, PreflopEMHandBundle
from utils.filter.helpers import initial_class_prior, normalize
from utils.parse import Hand
from utils.postflop_runner_bridge import (
    collect_postflop_em_bundle_for_hand,
    collect_postflop_observations_known_hole_cards,
    postflop_target_decisions_for_hand,
)
from utils.prior.postflop import (
    FOLD,
    PHI_DIM,
    feature_vector,
    train_baseline_facing_bet,
    train_baseline_no_bet,
)
from utils.prior.preflop import (
    PREFLOP_PHI_DIM,
    canonical_preflop_action,
    preflop_feature_vector,
    state_key_from_parse_state,
    train_baseline_preflop,
)
from utils.strength.common import parse_card
from utils.strength.preflop import get_equivalence_class


def preflop_phi_column_labels() -> List[str]:
    """Human-readable names for each column of ``preflop_feature_vector`` (order must match code)."""
    pos = ["pos_sb", "pos_bb", "pos_utg", "pos_hj", "pos_co", "pos_btn"]
    act = ["active_heads_up", "active_three_way", "active_multiway"]
    rc = ["raises_0", "raises_1", "raises_2plus"]
    spr = ["spr_deep", "spr_medium", "spr_shallow"]
    base = [
        "bias",
        "hand_strength",
        "high_over_14",
        "low_over_14",
        "is_pair",
        "suited",
        "gap_over_12",
        "broadways_half",
        "has_ace",
        "playability",
        "premium_flag",
        "speculative_flag",
        "facing_bet_flag",
    ]
    return base + pos + act + rc + spr


def postflop_phi_column_labels() -> List[str]:
    """Names for each entry of the length-``PHI_DIM`` postflop feature vector.

    Tracks the base + rich + equity blocks defined in
    :mod:`utils.prior.postflop`. Bumped from 13 to 30 alongside Method A
    (rich per-combo categorical features) and Method E (rollout equity).
    """
    from utils.prior.postflop import phi_column_labels as _labels

    return list(_labels())


def _verify_phi_dims() -> None:
    """Assert label tables match compiled feature dimensions (fails fast on drift)."""
    if len(preflop_phi_column_labels()) != PREFLOP_PHI_DIM:
        raise RuntimeError("preflop_phi_column_labels length mismatch vs PREFLOP_PHI_DIM")
    from utils.prior.postflop import PHI_DIM

    if len(postflop_phi_column_labels()) != PHI_DIM:
        raise RuntimeError("postflop_phi_column_labels length mismatch vs PHI_DIM")


_verify_phi_dims()


def hole_cards_to_hand_class(hole: str) -> Optional[str]:
    """Map a 4-char hole string (e.g. ``AhKd``) to a 169-style label, or ``None``.

    Uses :func:`utils.strength.preflop.get_equivalence_class` so suited/offsuit
    and pairs are canonical for preflop features and EM priors.
    """
    hole = (hole or "").strip()
    if len(hole) < 4:
        return None
    try:
        c1 = parse_card(hole[0:2])
        c2 = parse_card(hole[2:4])
    except ValueError:
        return None
    return get_equivalence_class([c1, c2])


def _numeric_name_sort_key(path: Path) -> Tuple[int, int | str]:
    """Sort session folders numerically when the directory name is digits (``30`` before ``9``)."""
    s = path.name
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def iter_phh_files(session_dir: Path) -> List[Path]:
    """List numbered ``.phh`` files inside a single *session* directory."""
    session_dir = session_dir.expanduser().resolve()
    if session_dir.is_file():
        if session_dir.suffix.lower() != ".phh":
            raise ValueError(f"Not a .phh file: {session_dir}")
        return [session_dir]
    if not session_dir.is_dir():
        raise FileNotFoundError(session_dir)
    files = sorted(
        (p for p in session_dir.iterdir() if p.is_file() and p.suffix.lower() == ".phh"),
        key=lambda p: int(p.stem),
    )
    return files


def session_dirs_under_root(root: Path) -> List[Path]:
    """Immediate child folders of ``root`` that contain at least one ``.phh`` (Pluribus layout)."""
    out: List[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if any(f.is_file() and f.suffix.lower() == ".phh" for f in child.iterdir()):
            out.append(child)
    return sorted(out, key=_numeric_name_sort_key)


def expand_data_path(
    raw: Path,
    *,
    session_names: Optional[set[str]] = None,
    max_sessions: Optional[int] = None,
) -> List[Path]:
    """
    Resolve one CLI path to a list of load units:

    - A ``.phh`` file → that file only.
    - A *session* directory (contains ``.phh`` files directly) → that directory.
    - A *Pluribus root* (no ``.phh`` in the directory itself, but child folders have hands) →
      each child session directory, optionally filtered and truncated.
    """
    p = raw.expanduser().resolve()
    if p.is_file():
        if p.suffix.lower() != ".phh":
            raise ValueError(f"Not a .phh file: {p}")
        return [p]

    if not p.is_dir():
        raise FileNotFoundError(p)

    direct_phh = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".phh"]
    if direct_phh:
        if session_names is not None and p.name not in session_names:
            return []
        return [p]

    subs = session_dirs_under_root(p)
    if session_names is not None:
        subs = [s for s in subs if s.name in session_names]
    if max_sessions is not None:
        subs = subs[: max(0, int(max_sessions))]
    if not subs:
        raise ValueError(
            f"No session folders with .phh files under {p} "
            f"(session filter={session_names!r}, max_sessions={max_sessions!r})."
        )
    return subs


@dataclass(frozen=True)
class HandRef:
    """One parsed hand with a stable index for EM keys and logging.

    ``global_index`` is the position of this hand in the flattened list
    produced by :func:`flatten_hands` (0-based, contiguous). It is embedded in
    supervised combo keys and bundle metadata so multiple hands never collide.
    """

    source: str
    global_index: int
    hand: Hand


def _parse_phh_worker(job: Tuple[str, str]) -> Tuple[str, Hand]:
    """Parse one ``.phh`` in a worker process (``job`` = ``(repo_root, phh_path)``)."""
    repo_root, phh_str = job
    pr = Path(repo_root)
    for p in (str(pr), str(pr / "utils")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from utils.parse import Hand as HandCls

    ph = Path(phh_str).expanduser().resolve()
    return str(ph), HandCls.from_file(ph)


def flatten_hands(
    inputs: Sequence[str | Path],
    *,
    session_names: Optional[Sequence[str]] = None,
    max_sessions: Optional[int] = None,
    parse_workers: int = 0,
    progress_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> List[HandRef]:
    """
    Load hands from each path. A Pluribus *root* (e.g. ``pluribus/``) is expanded to all session
    subfolders. ``session_names`` limits to those folder names (e.g. ``30``, ``99``). ``max_sessions``
    keeps the first N session folders after numeric sort (per root path).

    ``parse_workers``: parallel ``Hand.from_file`` workers. ``0`` = ``min(8, cpu_count or 4)``,
    ``1`` = sequential (slowest but simplest). Parsing dominates pipeline startup time.

    ``progress_hook``: optional ``hook(phase, detail_dict)`` for UI / logging (e.g. stdout).
    """
    name_set = None if session_names is None else {str(x) for x in session_names}
    jobs: List[Tuple[str, str]] = []
    repo = str(REPO_ROOT)
    for raw in inputs:
        p = Path(raw).expanduser().resolve()
        units = expand_data_path(p, session_names=name_set, max_sessions=max_sessions)
        for unit in units:
            for phh in iter_phh_files(unit):
                jobs.append((repo, str(phh.resolve())))

    if progress_hook:
        progress_hook("flatten_hands_jobs", {"n_files": len(jobs)})

    n = len(jobs)
    if n == 0:
        return []

    workers = int(parse_workers)
    if workers == 0:
        workers = min(8, max(2, os.cpu_count() or 4))
    workers = max(1, workers)

    results: List[Optional[Tuple[str, Hand]]] = [None] * n
    report_every = max(1, min(100, n // 20 or 1))

    if workers == 1:
        for i, job in enumerate(jobs):
            src, hand = _parse_phh_worker(job)
            results[i] = (src, hand)
            if progress_hook and (i + 1 == n or (i + 1) % report_every == 0):
                progress_hook("flatten_hands_parse", {"done": i + 1, "total": n})
        ordered = [results[i] for i in range(n)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_parse_phh_worker, job): idx for idx, job in enumerate(jobs)}
            done = 0
            for fut in as_completed(futs):
                idx = futs[fut]
                results[idx] = fut.result()
                done += 1
                if progress_hook and (done == n or done % report_every == 0):
                    progress_hook("flatten_hands_parse", {"done": done, "total": n, "workers": workers})
        ordered = [results[i] for i in range(n)]

    out: List[HandRef] = []
    for g, pair in enumerate(ordered):
        assert pair is not None
        src, hand = pair
        out.append(HandRef(source=src, global_index=g, hand=hand))
    return out


def split_hand_refs(
    refs: Sequence[HandRef],
    *,
    train_frac: float,
    find_theta_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[HandRef], List[HandRef], List[HandRef]]:
    """Shuffle then partition ``HandRef`` lists into train / EM-holdout / test.

    Fractions are **renormalized** by their sum so callers can pass
    ``(0.7, 0.2, 0.1)`` or ``(70, 20, 10)`` equivalently. Allocation uses
    integer floor sizes so every hand is assigned exactly once (no gaps when
    ``n`` is small).
    """
    tw = train_frac + find_theta_frac + test_frac
    if abs(tw - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {tw}")
    order = list(refs)
    rng = random.Random(seed)
    rng.shuffle(order)
    n = len(order)
    if n == 0:
        return [], [], []

    # Floor allocation so chunks partition ``n`` without round-off gaps (small-n safe).
    nt = min(n, max(0, int(n * train_frac / tw)))
    nf = min(n - nt, max(0, int(n * find_theta_frac / tw)))
    ns = n - nt - nf
    train = order[:nt]
    mid = order[nt : nt + nf]
    test = order[nt + nf :]
    assert len(train) + len(mid) + len(test) == n
    return train, mid, test


def preflop_decisions_for_hand(hand: Hand, target: str, hand_index: int):
    """Enumerate preflop actions taken by ``target`` with parse-time state keys.

    Mirrors the runner's decision extraction so ``find_theta`` and training
    share the same ``state_key`` strings. A lightweight nested dataclass is
    used here to avoid importing the full runner module.
    """
    from dataclasses import dataclass

    @dataclass
    class PreflopDecision:
        hand_index: int
        observer: str
        target: str
        action_index: int
        state_key: str
        action_bucket: int
        amount: int

    decisions: List[PreflopDecision] = []
    preflop_actions = hand.actions.get("pre-flop", {})
    preflop_states = hand.states.get("pre-flop", [])

    for action_index in sorted(preflop_actions):
        actor, _, amount = preflop_actions[action_index]
        if actor != target:
            continue
        if action_index >= len(preflop_states):
            continue

        state = preflop_states[action_index]
        state_key = state_key_from_parse_state(state, target).as_string()
        action_bucket = preflop_actions[action_index][1][0]
        decisions.append(
            PreflopDecision(
                hand_index=hand_index,
                observer="",
                target=target,
                action_index=action_index,
                state_key=state_key,
                action_bucket=action_bucket,
                amount=amount,
            )
        )

    return decisions


def collect_grouped_em_bundles_refs(
    refs: Sequence[HandRef],
    observers: Sequence[str],
    targets: Sequence[str],
) -> Dict[Tuple[str, str], List[PreflopEMHandBundle]]:
    """Group preflop EM bundles by ``(observer, target)`` player-name pairs.

    For each hand and each ordered pair with ``observer != target``, builds a
    :class:`utils.em.preflop.PreflopEMHandBundle` using the observer's known
    hole cards as **dead cards** in ``initial_class_prior``. Uses each hand's
    ``global_index`` in decision keys (same semantics as the interactive runner).
    """
    groups: Dict[Tuple[str, str], List[PreflopEMHandBundle]] = defaultdict(list)
    for ref in refs:
        hand = ref.hand
        hi = ref.global_index
        for observer in observers:
            for target in targets:
                if observer == target:
                    continue
                decisions = preflop_decisions_for_hand(hand, target, hi)
                if not decisions:
                    continue
                dead = hand.hole_cards.get(observer, "")
                initial_range = normalize(initial_class_prior(dead_cards=dead))
                bundle = PreflopEMHandBundle(
                    tuple(PreflopEMDecision(d.state_key, d.action_bucket) for d in decisions),
                    initial_range,
                )
                groups[(observer, target)].append(bundle)
    return groups


def pool_bundles_for_target(
    grouped: Dict[Tuple[str, str], List[PreflopEMHandBundle]],
    target: str,
) -> List[PreflopEMHandBundle]:
    """Concatenate EM bundles whose *target* role matches ``target`` (exact player name string)."""
    out: List[PreflopEMHandBundle] = []
    for (obs, tgt), bundles in grouped.items():
        if tgt == target:
            out.extend(bundles)
    return out


def collect_preflop_supervised_rows(refs: Sequence[HandRef]) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(X, y)`` for population preflop baseline training.

    One row per ``(hand, player, preflop_decision)`` where that player's hole
    cards map to a 169 class. ``X`` rows are :func:`utils.prior.preflop.preflop_feature_vector`
    and ``y`` entries are canonical actions ``{FOLD, CHECK_CALL, RAISE}``.
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for ref in refs:
        hand = ref.hand
        for player in hand.player_names:
            hc = hole_cards_to_hand_class(hand.hole_cards.get(player, "") or "")
            if hc is None:
                continue
            for dec in preflop_decisions_for_hand(hand, player, ref.global_index):
                phi = preflop_feature_vector(hc, dec.state_key)
                xs.append(phi)
                ys.append(canonical_preflop_action(dec.action_bucket))
    if not xs:
        return np.zeros((0, PREFLOP_PHI_DIM)), np.zeros((0,), dtype=int)
    return np.stack(xs, axis=0), np.asarray(ys, dtype=int)


def collect_postflop_supervised_rows(
    refs: Sequence[HandRef],
    *,
    progress_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    postflop_equity_mc: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into facing-bet (3-class) and no-bet (CALL vs RAISE) training rows.

    ``progress_hook``: optional ``hook(phase, detail)`` with ``phase="postflop_rows_progress"``
    and ``detail`` containing ``done``, ``total`` (1-based hand index through ``refs``).

    ``postflop_equity_mc``: flop rollout Monte Carlo sample count for Method E
    equity when building rows (smaller is faster; ``32`` matches older defaults).
    """
    xf, yf = [], []
    xn, yn = [], []
    n = len(refs)
    report_every = max(1, min(100, n // 20 or 1)) if n else 1
    for i, ref in enumerate(refs):
        for player in ref.hand.player_names:
            obs = collect_postflop_observations_known_hole_cards(
                ref.hand,
                player,
                ref.global_index,
                equity_mc_samples=postflop_equity_mc,
            )
            if obs is None:
                continue
            for feat, action in obs.decisions:
                xv = feature_vector(feat)
                if feat.facing_bet:
                    xf.append(xv)
                    yf.append(int(action))
                else:
                    if int(action) == FOLD:
                        continue
                    xn.append(xv)
                    yn.append(int(action))
        if progress_hook and n and (
            i + 1 == 1 or i + 1 == n or (i + 1) % report_every == 0
        ):
            progress_hook("postflop_rows_progress", {"done": i + 1, "total": n})
    if not xf:
        Xf = np.zeros((0, PHI_DIM))
        Yf = np.zeros((0,), dtype=int)
    else:
        Xf = np.stack(xf, axis=0)
        Yf = np.asarray(yf, dtype=int)
    if not xn:
        Xn = np.zeros((0, PHI_DIM))
        Yn = np.zeros((0,), dtype=int)
    else:
        Xn = np.stack(xn, axis=0)
        Yn = np.asarray(yn, dtype=int)
    return Xf, Yf, Xn, Yn


def dump_json(path: Path, payload: object) -> Path:
    """Write ``payload`` as pretty-printed JSON; create parent dirs as needed."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_json(path: Path) -> object:
    """Load JSON from disk into native Python types (no schema validation)."""
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def read_session_names_file(path: str | Path) -> List[str]:
    """One session directory name per non-empty line (e.g. ``30`` → ``pluribus/30``)."""
    p = Path(path).expanduser().resolve()
    names: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            names.append(s)
    return names


def split_session_names(
    names: Sequence[str],
    *,
    train_frac: float,
    em_frac: float,
    online_frac: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Partition session names into train / EM / online (same counting semantics as ``split_hand_refs``)."""
    tw = train_frac + em_frac + online_frac
    if abs(tw - 1.0) > 1e-6:
        raise ValueError(f"Session fractions must sum to 1.0, got {tw}")
    order = list(names)
    rng = random.Random(seed)
    rng.shuffle(order)
    n = len(order)
    if n == 0:
        return [], [], []
    nt = min(n, max(0, int(n * train_frac / tw)))
    nem = min(n - nt, max(0, int(n * em_frac / tw)))
    non = n - nt - nem
    train = order[:nt]
    em_part = order[nt : nt + nem]
    online = order[nt + nem : nt + nem + non]
    assert len(train) + len(em_part) + len(online) == n
    return train, em_part, online


def handref_session_hand(path_str: str) -> Tuple[str, str]:
    """``(session_dir_name, hand_stem)`` from a ``.phh`` path string."""
    p = Path(path_str)
    return p.parent.name, p.stem


def gather_preflop_bundles_for_target_player(
    refs: Sequence[HandRef],
    all_players: Sequence[str],
    target_player: str,
    *,
    restrict_observers: Optional[Sequence[str]] = None,
) -> Tuple[List[PreflopEMHandBundle], List[Dict[str, Any]]]:
    """EM bundles where ``target_player`` acts preflop; one meta dict per bundle (session, hand, observer).

    If ``restrict_observers`` is set, only those observer names are used (e.g. a single observer|target pair).
    Otherwise every name in ``all_players`` except ``target_player`` is considered.
    """
    if restrict_observers is not None:
        observers_iter = list(dict.fromkeys(restrict_observers))
    else:
        observers_iter = [p for p in all_players if p != target_player]

    bundles: List[PreflopEMHandBundle] = []
    metas: List[Dict[str, Any]] = []
    for ref in refs:
        if target_player not in ref.hand.player_names:
            continue
        for observer in observers_iter:
            if observer == target_player:
                continue
            decisions = preflop_decisions_for_hand(ref.hand, target_player, ref.global_index)
            if not decisions:
                continue
            dead = ref.hand.hole_cards.get(observer, "")
            initial_range = normalize(initial_class_prior(dead_cards=dead))
            bundle = PreflopEMHandBundle(
                tuple(PreflopEMDecision(d.state_key, d.action_bucket) for d in decisions),
                initial_range,
            )
            sess, stem = handref_session_hand(ref.source)
            bundles.append(bundle)
            metas.append(
                {
                    "session": sess,
                    "hand_number": int(stem) if stem.isdigit() else stem,
                    "phh_path": ref.source,
                    "observer": observer,
                    "target": target_player,
                    "n_target_preflop_decisions": len(decisions),
                }
            )
    return bundles, metas


def gather_postflop_bundles_for_target_player(
    refs: Sequence[HandRef],
    all_players: Sequence[str],
    target_player: str,
    *,
    restrict_observers: Optional[Sequence[str]] = None,
    postflop_require_observer: Optional[str] = None,
) -> Tuple[List[PostflopEMHandBundle], List[Dict[str, Any]]]:
    """Postflop EM bundles: same prior as preflop EM (169 → 1,326) with observer dead cards.

    One bundle per (hand, observer) where the target has postflop actions. If
    ``postflop_require_observer`` is set, skip hands where that name is not seated.
    """
    if restrict_observers is not None:
        observers_iter = list(dict.fromkeys(restrict_observers))
    else:
        observers_iter = [p for p in all_players if p != target_player]

    bundles: List[PostflopEMHandBundle] = []
    metas: List[Dict[str, Any]] = []
    n_seen = 0
    n_refs_checked = 0
    progress_every = 500
    if len(refs) <= 200:
        progress_every = 5
    elif len(refs) <= 2000:
        progress_every = 50
    for ref in refs:
        n_refs_checked += 1
        if target_player not in ref.hand.player_names:
            continue
        if postflop_require_observer is not None and postflop_require_observer not in ref.hand.player_names:
            continue
        n_seen += 1
        if n_seen == 1 or n_seen % progress_every == 0:
            LOG_PC.info(
                "gather_postflop_bundles | target=%s | refs_checked=%d | "
                "candidate_hands_target_seated=%d | bundles_so_far=%d",
                target_player,
                n_refs_checked,
                n_seen,
                len(bundles),
            )
        target_actions = postflop_target_decisions_for_hand(ref.hand, target_player)
        if not target_actions:
            continue
        for observer in observers_iter:
            if observer == target_player:
                continue
            bundle = collect_postflop_em_bundle_for_hand(
                ref.hand,
                observer,
                target_player,
                ref.global_index,
                target_actions=target_actions,
            )
            if bundle is None:
                continue
            sess, stem = handref_session_hand(ref.source)
            bundles.append(bundle)
            metas.append(
                {
                    "session": sess,
                    "hand_number": int(stem) if stem.isdigit() else stem,
                    "phh_path": ref.source,
                    "observer": observer,
                    "target": target_player,
                    "n_target_postflop_timesteps": len(bundle.decisions),
                    "n_combos_in_prior": len(bundle.initial_combo_range),
                }
            )
    return bundles, metas
