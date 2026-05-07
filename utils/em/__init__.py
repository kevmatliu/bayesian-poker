"""Expectation–maximization for latent tendency parameters (preflop and postflop).

Preflop (169 abstract classes):

    P(a | h, s, theta) ∝ p_base(a | h, s) * exp(theta · u(h, s, a))

Postflop (combo / features):

    P(a | x, theta) ∝ P_base(a | x) * exp(theta · u(x, a))

Use ``TendencyEM`` as the namespace for phase-specific runners.
"""

from __future__ import annotations

from utils.em.common import EMPhase, normalize_log_weights
from utils.em.postflop import (
    PostflopThetaObservation,
    e_step_combo_posterior,
    m_step_theta_post_gradient_ascent,
    postflop_theta_gradient,
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
    "PostflopThetaObservation",
    "PreflopEMDecision",
    "PreflopEMHandBundle",
    "TendencyEM",
    "e_step_combo_posterior",
    "e_step_hand_class_posterior",
    "m_step_theta_post",
    "m_step_theta_post_gradient_ascent",
    "normalize_log_weights",
    "postflop_theta_gradient",
    "run_postflop_theta_em",
    "run_preflop_em",
    "single_hand_em_gradient_sample",
]
