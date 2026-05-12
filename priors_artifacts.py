"""Build :class:`PreflopActionModel` / :class:`PostflopActionModel` from training / θ JSON artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple

from runners.find_theta import load_global_priors
from runners.models import PREFLOP_PRIOR_FLOOR
from utils.action.postflop import PostflopActionModel
from utils.action.preflop import PreflopActionModel
from utils.prior.postflop import PostflopPrior
from utils.prior.preflop import PreflopPrior


def preflop_postflop_priors_for_target(
    target: str,
    global_priors_path: Path,
    players_block: Mapping[str, Any],
    *,
    preflop_floor: float = PREFLOP_PRIOR_FLOOR,
    postflop_floor: float = 1e-6,
) -> Tuple[PreflopActionModel, PostflopActionModel]:
    """Population ``beta`` from ``global_priors_path``; per-target ``theta_*`` from ``players_block``."""
    beta_preflop, beta_facing, beta_no_bet = load_global_priors(Path(global_priors_path))
    if target not in players_block:
        raise ValueError(
            f"Target {target!r} missing from player θ JSON (not seen during EM?). "
            f"Keys: {sorted(players_block.keys())}"
        )
    entry = players_block[target]
    tp = entry["theta_pre"]
    ts = entry["theta_post"]
    return (
        PreflopActionModel(
            PreflopPrior(floor=preflop_floor, beta_preflop=beta_preflop),
            tuple(float(x) for x in tp),
        ),
        PostflopActionModel(
            PostflopPrior(
                floor=postflop_floor,
                beta_facing=beta_facing,
                beta_no_bet=beta_no_bet,
            ),
            tuple(float(x) for x in ts),
        ),
    )
