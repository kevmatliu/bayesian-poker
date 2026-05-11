"""Expectation–maximization for latent tendency parameters (preflop and postflop).

Preflop (169 abstract classes):

    P(a | h, s, theta) ∝ p_base(a | h, s) * exp(theta · u(h, s, a))

Postflop (combo / features):

    P(a | x, theta) ∝ P_base(a | x) * exp(theta · u(x, a))

Use ``TendencyEM`` as the namespace for phase-specific runners.
"""

from __future__ import annotations

from utils.em.common import (
    EMPhase,
    M_STEP_GRAD_NORM_TOL,
    POSTFLOP_M_BATCH_SIZE,
    PREFLOP_M_BATCH_SIZE,
    normalize_log_weights,
)
from utils.em.postflop import (
    PostflopEMHandBundle,
    PostflopEMTimestep,
    PostflopThetaObservation,
    e_step_combo_posterior,
    e_step_postflop_bundle,
    m_step_theta_post_gradient_ascent,
    m_step_theta_post_gradient_ascent_bundles,
    postflop_theta_gradient,
    postflop_theta_gradient_bundles,
    run_postflop_theta_em,
    single_hand_em_gradient_sample,
)
from utils.em.preflop import (
    PreflopEMDecision,
    PreflopEMHandBundle,
    e_step_hand_class_posterior,
    m_step_theta_pre,
    run_preflop_em,
)


class TendencyEM:
    """Phase-specific EM entry points (preflop vs postflop)."""

    preflop = staticmethod(run_preflop_em)
    postflop = staticmethod(run_postflop_theta_em)


__all__ = [
    "EMPhase",
    "M_STEP_GRAD_NORM_TOL",
    "POSTFLOP_M_BATCH_SIZE",
    "PREFLOP_M_BATCH_SIZE",
    "PostflopEMHandBundle",
    "PostflopEMTimestep",
    "PostflopThetaObservation",
    "PreflopEMDecision",
    "PreflopEMHandBundle",
    "TendencyEM",
    "e_step_combo_posterior",
    "e_step_hand_class_posterior",
    "e_step_postflop_bundle",
    "m_step_theta_post_gradient_ascent",
    "m_step_theta_post_gradient_ascent_bundles",
    "normalize_log_weights",
    "postflop_theta_gradient",
    "postflop_theta_gradient_bundles",
    "run_postflop_theta_em",
    "run_preflop_em",
    "single_hand_em_gradient_sample",
]
