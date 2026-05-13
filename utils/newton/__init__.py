"""Direct Newton optimization on marginal log-likelihood for tendency ``theta`` (MAP with L2).

Preflop and postflop use the same latent marginalization as EM; gradients match the observed-data
score when posteriors are evaluated at the current ``theta``. Hessians are obtained by finite
differencing that score (small ``theta`` dimension: 3).
"""

from __future__ import annotations

from utils.newton.common import (
    NEWTON_HESSIAN_FD_EPS,
    NEWTON_HESSIAN_SHIFT_RIDGE,
    NEWTON_LINE_SEARCH_BACKTRACK,
    NEWTON_LINE_SEARCH_MIN_ALPHA,
    backtracking_line_search,
    hessian_from_gradient_fd,
    log_sum_exp,
    log_sum_exp_mapping,
    newton_maximization_direction,
)
from utils.newton.postflop import (
    postflop_map_gradient,
    postflop_map_objective,
    run_postflop_theta_newton,
)
from utils.newton.preflop import (
    preflop_map_gradient,
    preflop_map_objective,
    run_preflop_newton,
)

__all__ = [
    "NEWTON_HESSIAN_FD_EPS",
    "NEWTON_HESSIAN_SHIFT_RIDGE",
    "NEWTON_LINE_SEARCH_BACKTRACK",
    "NEWTON_LINE_SEARCH_MIN_ALPHA",
    "backtracking_line_search",
    "hessian_from_gradient_fd",
    "log_sum_exp",
    "log_sum_exp_mapping",
    "newton_maximization_direction",
    "postflop_map_gradient",
    "postflop_map_objective",
    "preflop_map_gradient",
    "preflop_map_objective",
    "run_postflop_theta_newton",
    "run_preflop_newton",
]
