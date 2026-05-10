"""Preflop EM over abstract hand classes (169) and theta_pre."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.em.common import M_STEP_GRAD_NORM_TOL, normalize_log_weights

LOG = logging.getLogger(__name__)
from utils.prior.preflop import (
    ACTION_BUCKETS,
    PreflopPrior,
    canonical_preflop_action,
)
from utils.strength.preflop import all_169_classes


@dataclass(frozen=True)
class PreflopEMDecision:
    state_key: str
    action_bucket: int


@dataclass(frozen=True)
class PreflopEMHandBundle:
    """Target decisions in one hand plus the observer-implied prior over hand classes."""

    decisions: Tuple[PreflopEMDecision, ...]
    initial_range: Dict[str, float]


def e_step_hand_class_posterior(
    bundle: PreflopEMHandBundle,
    prior: PreflopPrior,
) -> Dict[str, float]:
    """Posterior q(h) ∝ pi_0(h) * prod_t P(a_t | h, s_t, theta)."""
    log_q: Dict[str, float] = {}
    for h in all_169_classes():
        p0 = bundle.initial_range.get(h, 0.0)
        if p0 <= 0.0:
            continue
        logp = math.log(p0)
        for dec in bundle.decisions:
            probs = prior.action_probs(h, dec.state_key)
            a = canonical_preflop_action(dec.action_bucket)
            logp += math.log(probs[a])
        log_q[h] = logp
    if not log_q:
        raise ValueError("E-step: no hand class with positive prior mass")
    return normalize_log_weights(log_q)


def m_step_theta_pre(
    bundles: List[PreflopEMHandBundle],
    q_by_hand: List[Dict[str, float]],
    prior_template: PreflopPrior,
    theta_init: Sequence[float],
    l2: float = 0.25,
    lr: float = 0.05,
    steps: int = 100,
    *,
    grad_norm_tol: float = M_STEP_GRAD_NORM_TOL,
) -> Tuple[np.ndarray, int]:
    """Gradient ascent on the M-step objective with L2 penalty on theta_pre.

    Returns ``(theta, m_iterations_used)`` where ``m_iterations_used`` is the number of
    gradient steps taken (including early stop).
    """
    theta = np.asarray(theta_init, dtype=float).copy()
    baseline = PreflopPrior(
        theta_pre=(0.0, 0.0, 0.0),
        floor=prior_template.floor,
        beta_preflop=prior_template._beta_store.copy(),
    )

    used = steps
    for step_i in range(steps):
        grad = np.zeros(3, dtype=float)
        for bundle, q_h in zip(bundles, q_by_hand):
            for dec in bundle.decisions:
                observed_a = canonical_preflop_action(dec.action_bucket)
                for hand_class, q in q_h.items():
                    if q <= 0.0:
                        continue
                    probs = prior_template.action_probs_with_theta(
                        hand_class=hand_class,
                        state_key=dec.state_key,
                        theta_pre=theta,
                    )
                    utilities = baseline.action_utility_vectors(hand_class, dec.state_key)
                    u_obs = np.asarray(utilities[observed_a], dtype=float)
                    exp_u = sum(
                        probs[a] * np.asarray(utilities[a], dtype=float)
                        for a in ACTION_BUCKETS
                    )
                    grad += q * (u_obs - exp_u)
        grad -= l2 * theta
        gn = float(np.linalg.norm(grad))
        if gn < grad_norm_tol:
            used = step_i + 1
            LOG.info(
                "Preflop EM M-step early stop | grad iter %d/%d | |grad|=%.2e < %.2e | "
                "theta=[%.5f, %.5f, %.5f]",
                step_i + 1,
                steps,
                gn,
                grad_norm_tol,
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
            )
            break
        theta += lr * grad
        if step_i == 0 or step_i == steps - 1 or (step_i + 1) % max(1, steps // 10) == 0:
            LOG.info(
                "Preflop EM M-step | grad iter %d/%d | theta=[%.5f, %.5f, %.5f] |grad|=%.5f",
                step_i + 1,
                steps,
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
                gn,
            )
        elif LOG.isEnabledFor(logging.DEBUG) and (step_i + 1) % max(1, steps // 4) == 0:
            LOG.debug(
                "M-step grad step %d/%d: theta=[%.5f, %.5f, %.5f] |grad|≈%.5f",
                step_i + 1,
                steps,
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
                gn,
            )

    return theta, used


def run_preflop_em(
    bundles: List[PreflopEMHandBundle],
    *,
    prior_floor: float = 0.01,
    beta_preflop: np.ndarray | None = None,
    theta_init: Sequence[float] | None = None,
    num_em_iters: int = 5,
    m_l2: float = 0.25,
    m_lr: float = 0.05,
    m_steps: int = 100,
    m_grad_norm_tol: float = M_STEP_GRAD_NORM_TOL,
    history_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    bundle_meta: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Alternate E and M steps; return final theta and terminal posteriors q(h)."""
    theta = (
        np.zeros(3, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).copy()
    )
    if beta_preflop is None:
        prior = PreflopPrior(theta_pre=tuple(float(x) for x in theta), floor=prior_floor)
    else:
        prior = PreflopPrior(
            theta_pre=tuple(float(x) for x in theta),
            floor=prior_floor,
            beta_preflop=np.asarray(beta_preflop, dtype=float),
        )
    last_q: List[Dict[str, float]] = []

    for outer in range(num_em_iters):
        LOG.info(
            "Preflop EM outer %d/%d: E-step over %d bundles",
            outer + 1,
            num_em_iters,
            len(bundles),
        )
        if history_hook is not None:
            history_hook(
                {
                    "kind": "preflop_em_outer_start",
                    "em_outer": outer,
                    "num_em_iters": num_em_iters,
                    "n_bundles": len(bundles),
                }
            )
        nb = len(bundles)
        prog = max(1, nb // 20) if nb > 20 else 1
        last_q = []
        for bi, b in enumerate(bundles):
            last_q.append(e_step_hand_class_posterior(b, prior))
            if bi == 0 or bi == nb - 1 or (bi + 1) % prog == 0:
                LOG.info(
                    "Preflop EM E-step | outer %d/%d | bundle %d/%d",
                    outer + 1,
                    num_em_iters,
                    bi + 1,
                    nb,
                )
        max_q = max(max(d.values()) for d in last_q) if last_q else 0.0
        LOG.debug("E-step: max posterior mass in any bundle max_h q(h)=%.4f", max_q)
        LOG.info(
            "Preflop EM E-step done | outer %d/%d | max_h q(h) in any bundle ≈ %.4f",
            outer + 1,
            num_em_iters,
            max_q,
        )

        if history_hook is not None:
            e_stride = 1
            if nb > 2000:
                e_stride = max(1, nb // 2000)
                LOG.info(
                    "Preflop EM | jsonl subsample: recording preflop_e_step every %d bundle(s) "
                    "(n_bundles=%d; avoids millions of disk flushes)",
                    e_stride,
                    nb,
                )
            for bi, (bundle, qmap) in enumerate(zip(bundles, last_q)):
                if bi != 0 and bi != nb - 1 and bi % e_stride != 0:
                    continue
                meta: Dict[str, Any] = {}
                if bundle_meta is not None and bi < len(bundle_meta):
                    meta = dict(bundle_meta[bi])
                qs = list(qmap.values())
                ess = (sum(qs) ** 2) / sum(v * v for v in qs) if qs else 0.0
                history_hook(
                    {
                        "kind": "preflop_e_step",
                        "em_outer": outer,
                        "em_timestep": outer,
                        "bundle_index": bi,
                        "e_step_subsample_stride": e_stride,
                        "n_target_preflop_decisions": len(bundle.decisions),
                        "theta_pre": [float(x) for x in theta],
                        "max_q": float(max(qmap.values())) if qmap else None,
                        "ess": float(ess),
                        **meta,
                    }
                )
            history_hook(
                {
                    "kind": "preflop_e_step_finished",
                    "em_outer": outer,
                    "n_bundles": len(bundles),
                    "max_q_any_bundle": float(max_q),
                    "e_step_subsample_stride": e_stride,
                }
            )

        LOG.info(
            "Preflop EM M-step start | outer %d/%d | %d gradient steps | lr=%g | l2=%g",
            outer + 1,
            num_em_iters,
            m_steps,
            m_lr,
            m_l2,
        )
        if history_hook is not None:
            history_hook(
                {
                    "kind": "preflop_m_step_start",
                    "em_outer": outer,
                    "m_steps": m_steps,
                    "m_lr": m_lr,
                    "m_l2": m_l2,
                }
            )
        theta, m_grad_iters = m_step_theta_pre(
            bundles,
            last_q,
            prior_template=prior,
            theta_init=theta,
            l2=m_l2,
            lr=m_lr,
            steps=m_steps,
            grad_norm_tol=m_grad_norm_tol,
        )
        if beta_preflop is None:
            prior = PreflopPrior(theta_pre=tuple(float(x) for x in theta), floor=prior_floor)
        else:
            prior = PreflopPrior(
                theta_pre=tuple(float(x) for x in theta),
                floor=prior_floor,
                beta_preflop=np.asarray(beta_preflop, dtype=float),
            )
        LOG.info(
            "Preflop EM outer %d/%d complete | theta_pre=[%.6f, %.6f, %.6f] | M-step grad iters used=%d/%d",
            outer + 1,
            num_em_iters,
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            m_grad_iters,
            m_steps,
        )

        if history_hook is not None:
            history_hook(
                {
                    "kind": "preflop_m_step",
                    "em_outer": outer,
                    "em_timestep": outer,
                    "theta_pre": [float(x) for x in theta],
                    "n_bundles": len(bundles),
                    "m_gradient_steps_used": m_grad_iters,
                    "m_gradient_steps_cap": m_steps,
                }
            )

    LOG.info(
        "Preflop EM finished | outer_iters=%d | returning theta_pre for downstream (e.g. postflop bundle build)",
        num_em_iters,
    )
    return theta, last_q
