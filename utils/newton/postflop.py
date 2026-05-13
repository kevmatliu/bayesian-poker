"""Newton optimization on the marginal postflop log-likelihood for ``theta_post`` (MAP with L2)."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.action.postflop import PostflopActionModel, feature_vector
from utils.em.postflop import (
    PostflopEMHandBundle,
    e_step_postflop_bundle,
    postflop_theta_gradient_bundles,
)
from utils.newton.common import (
    NEWTON_HESSIAN_FD_EPS,
    NEWTON_HESSIAN_SHIFT_RIDGE,
    backtracking_line_search,
    hessian_from_gradient_fd,
    log_sum_exp_mapping,
    newton_maximization_direction,
)
from utils.prior.postflop import PostflopPrior

LOG = logging.getLogger(__name__)


def _hand_progress_stride(n_hands: int) -> int:
    if n_hands <= 0:
        return 1
    return max(1, n_hands // 25)


def _e_step_postflop_all_with_progress(
    bundles: List[PostflopEMHandBundle],
    prior: PostflopActionModel,
    *,
    log: logging.Logger,
    label: str,
) -> List[Dict[str, float]]:
    n = len(bundles)
    if n == 0:
        return []
    stride = _hand_progress_stride(n)
    t0 = time.perf_counter()
    out: List[Dict[str, float]] = []
    for hi, b in enumerate(bundles):
        out.append(e_step_postflop_bundle(b, prior))
        if hi == 0 or hi == n - 1 or (hi + 1) % stride == 0:
            log.info(
                "%s | E-step hand %d/%d | elapsed_s=%.2f",
                label,
                hi + 1,
                n,
                time.perf_counter() - t0,
            )
    log.info("%s | E-step finished | hands=%d | wall_s=%.2f", label, n, time.perf_counter() - t0)
    return out


def _postflop_bundle_log_marginal_evidence(bundle: PostflopEMHandBundle, prior: PostflopActionModel) -> float:
    """``log sum_c pi_0(c) prod_t P(a_t | x_t(c), theta)`` (same latent support as the EM E-step)."""
    initial = bundle.initial_combo_range
    log_q: Dict[str, float] = {
        combo: math.log(p0) for combo, p0 in initial.items() if p0 > 0.0
    }
    if not log_q:
        return float("-inf")
    alive = set(log_q.keys())

    for step in bundle.decisions:
        feat_map = dict(step.features_by_combo)
        live = [c for c in alive if c in feat_map]
        if not live:
            return float("-inf")
        phi = np.stack([feature_vector(feat_map[c]) for c in live], axis=0)
        facing = np.fromiter(
            (feat_map[c].facing_bet for c in live),
            dtype=bool,
            count=len(live),
        )
        log_probs = prior.action_log_probs_matrix(phi, facing)
        log_pa = log_probs[:, int(step.action)]
        for combo, lp in zip(live, log_pa):
            log_q[combo] += float(lp)
        alive = set(live)

    log_q = {c: v for c, v in log_q.items() if c in alive}
    if not log_q:
        return float("-inf")
    return log_sum_exp_mapping(log_q)


def postflop_map_objective(
    bundles: Sequence[PostflopEMHandBundle],
    beta_source: PostflopPrior,
    theta_vec: np.ndarray,
    *,
    l2: float,
    log: Optional[logging.Logger] = None,
    log_label: str = "postflop_map_objective",
) -> float:
    """Marginal log-likelihood of observed postflop actions minus ``(l2/2)||theta||^2``."""
    prior = PostflopActionModel(
        beta_source,
        tuple(float(x) for x in theta_vec),
    )
    n = len(bundles)
    ell = 0.0
    t0 = time.perf_counter()
    if log is not None and n > 0:
        stride = _hand_progress_stride(n)
        log.info("%s | MAP objective | marginal log-evidence over %d hand-bundles…", log_label, n)
        for hi, b in enumerate(bundles):
            ell += _postflop_bundle_log_marginal_evidence(b, prior)
            if hi == 0 or hi == n - 1 or (hi + 1) % stride == 0:
                log.info(
                    "%s | MAP objective | hand %d/%d | partial_ell=%.4f | elapsed_s=%.2f",
                    log_label,
                    hi + 1,
                    n,
                    ell,
                    time.perf_counter() - t0,
                )
        log.info(
            "%s | MAP objective | done | ell=%.4f | prior_pen=%.4f | wall_s=%.2f",
            log_label,
            ell,
            0.5 * float(l2) * float(np.dot(theta_vec, theta_vec)),
            time.perf_counter() - t0,
        )
    else:
        for b in bundles:
            ell += _postflop_bundle_log_marginal_evidence(b, prior)
    return ell - 0.5 * float(l2) * float(np.dot(theta_vec, theta_vec))


def postflop_map_gradient(
    bundles: List[PostflopEMHandBundle],
    beta_source: PostflopPrior,
    theta_vec: np.ndarray,
    *,
    l2: float,
    log: Optional[logging.Logger] = None,
    log_label: str = "postflop_map_gradient",
) -> np.ndarray:
    """Gradient of :func:`postflop_map_objective` (E-step posteriors at the same ``theta``)."""
    live = PostflopActionModel(
        beta_source,
        tuple(float(x) for x in theta_vec),
    )
    n = len(bundles)
    if log is not None and n > 0:
        log.info("%s | MAP gradient | E-step over %d hand-bundles…", log_label, n)
        q_by_hand = _e_step_postflop_all_with_progress(bundles, live, log=log, label=log_label)
        log.info("%s | MAP gradient | batched tilt-score accumulation…", log_label)
    else:
        q_by_hand = [e_step_postflop_bundle(b, live) for b in bundles]
    t_acc = time.perf_counter()
    grad = postflop_theta_gradient_bundles(live, bundles, q_by_hand, l2=l2)
    if log is not None:
        log.info(
            "%s | MAP gradient | done | |grad|=%.5f | tilt_accum_s=%.2f",
            log_label,
            float(np.linalg.norm(grad)),
            time.perf_counter() - t_acc,
        )
    return grad


def run_postflop_theta_newton(
    bundles_by_hand: Sequence[PostflopEMHandBundle],
    *,
    prior_floor: float = 1e-6,
    theta_init: Sequence[float] | None = None,
    beta_facing: np.ndarray | None = None,
    beta_no_bet: np.ndarray | None = None,
    max_newton_iters: int = 15,
    m_l2: float = 0.25,
    clip: float = 3.0,
    center_each_step: bool = True,
    hessian_fd_eps: float = NEWTON_HESSIAN_FD_EPS,
    hessian_shift_ridge: float = NEWTON_HESSIAN_SHIFT_RIDGE,
    grad_norm_tol: float = 0.1,
    history_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    hand_meta: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """MAP Newton on marginal postflop log-likelihood; mirrors :func:`utils.em.postflop.run_postflop_theta_em`."""
    theta = (
        np.zeros(3, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).copy()
    )
    base_prior = PostflopPrior(
        floor=prior_floor,
        beta_facing=beta_facing,
        beta_no_bet=beta_no_bet,
    )
    bundles_list = list(bundles_by_hand)

    last_per_hand: List[Dict[str, float]] = []
    completed = 0
    for outer in range(max_newton_iters):
        completed = outer + 1
        iter_label = f"Postflop Newton iter {completed}/{max_newton_iters}"
        t0 = time.perf_counter()

        def _obj(tv: np.ndarray) -> float:
            return postflop_map_objective(
                bundles_list,
                base_prior,
                tv,
                l2=m_l2,
                log=LOG,
                log_label=f"{iter_label} | MAP objective",
            )

        def _grad(tv: np.ndarray) -> np.ndarray:
            return postflop_map_gradient(
                bundles_list,
                base_prior,
                tv,
                l2=m_l2,
                log=LOG,
                log_label=f"{iter_label} | MAP gradient",
            )

        LOG.info(
            "Postflop Newton iter %d/%d | %d hand-bundles | l2=%g | clip=%g | "
            "sub-steps: objective → gradient → Hessian FD (6× grad) → line search",
            completed,
            max_newton_iters,
            len(bundles_list),
            m_l2,
            clip,
        )
        if history_hook is not None:
            rec: Dict[str, Any] = {
                "kind": "postflop_newton_iter_start",
                "newton_iter": outer,
                "max_newton_iters": max_newton_iters,
                "n_hands": len(bundles_list),
                "theta_post": [float(x) for x in theta],
                "m_l2": m_l2,
            }
            if hand_meta is not None:
                rec["hand_meta_len"] = len(hand_meta)
            history_hook(rec)

        LOG.info("%s | (1/4) MAP objective F(θ) at current θ…", iter_label)
        t_sub = time.perf_counter()
        f0 = _obj(theta)
        LOG.info("%s | (1/4) MAP objective done | F=%.6f | wall_s=%.2f", iter_label, f0, time.perf_counter() - t_sub)

        LOG.info("%s | (2/4) MAP gradient ∇F(θ)…", iter_label)
        t_sub = time.perf_counter()
        g = _grad(theta)
        gn = float(np.linalg.norm(g))
        LOG.info(
            "%s | (2/4) MAP gradient done | |grad|=%.5f | wall_s=%.2f | cumulative_iter_s=%.2f",
            iter_label,
            gn,
            time.perf_counter() - t_sub,
            time.perf_counter() - t0,
        )
        if gn < grad_norm_tol:
            prior = PostflopActionModel(base_prior, tuple(float(x) for x in theta))
            last_per_hand = _e_step_postflop_all_with_progress(
                bundles_list,
                prior,
                log=LOG,
                label=f"{iter_label} | sync q(c) after early stop",
            )
            LOG.info(
                "Postflop Newton early stop | iter %d/%d | |grad|=%.2e | elapsed_s=%.2f",
                outer + 1,
                max_newton_iters,
                gn,
                time.perf_counter() - t0,
            )
            if history_hook is not None:
                history_hook(
                    {
                        "kind": "postflop_newton_step",
                        "newton_iter": outer,
                        "theta_post": [float(x) for x in theta],
                        "grad_norm": gn,
                        "map_objective": f0,
                        "early_stop": True,
                    }
                )
            break

        LOG.info("%s | (3/4) Hessian via finite differences on ∇F…", iter_label)
        t_sub = time.perf_counter()
        H = hessian_from_gradient_fd(
            theta,
            _grad,
            eps=hessian_fd_eps,
            log=LOG,
            log_prefix=f"{iter_label} | ",
        )
        LOG.info("%s | (3/4) Hessian assembled | wall_s=%.2f", iter_label, time.perf_counter() - t_sub)

        direction = newton_maximization_direction(H, g, ridge=hessian_shift_ridge)
        LOG.info(
            "%s | Newton direction | |d|=%.5f | (4/4) backtracking line search…",
            iter_label,
            float(np.linalg.norm(direction)),
        )
        t_sub = time.perf_counter()
        alpha, theta_try, f1 = backtracking_line_search(
            theta,
            direction,
            _obj,
            f0,
            log=LOG,
            log_prefix=f"{iter_label} | ",
        )
        LOG.info("%s | (4/4) line search finished | wall_s=%.2f", iter_label, time.perf_counter() - t_sub)
        if alpha <= 0.0:
            LOG.warning(
                "Postflop Newton line search failed at iter %d; keeping theta | |grad|=%.4f",
                outer + 1,
                gn,
            )
            prior = PostflopActionModel(base_prior, tuple(float(x) for x in theta))
            last_per_hand = _e_step_postflop_all_with_progress(
                bundles_list,
                prior,
                log=LOG,
                label=f"{iter_label} | sync q(c) after line search failure",
            )
            if history_hook is not None:
                history_hook(
                    {
                        "kind": "postflop_newton_step",
                        "newton_iter": outer,
                        "theta_post": [float(x) for x in theta],
                        "grad_norm": gn,
                        "map_objective": f0,
                        "line_search_alpha": alpha,
                        "line_search_failed": True,
                    }
                )
            break

        theta = theta_try
        if center_each_step:
            theta -= float(np.mean(theta))
        theta = np.clip(theta, -clip, clip)

        LOG.info(
            "Postflop Newton step | iter %d/%d | alpha=%.4g | F: %.4f -> %.4f | "
            "theta=[%.5f, %.5f, %.5f] | |g|=%.4f | wall_s=%.2f",
            outer + 1,
            max_newton_iters,
            alpha,
            f0,
            f1,
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            gn,
            time.perf_counter() - t0,
        )
        if history_hook is not None:
            history_hook(
                {
                    "kind": "postflop_newton_step",
                    "newton_iter": outer,
                    "theta_post": [float(x) for x in theta],
                    "grad_norm": gn,
                    "map_objective": f1,
                    "line_search_alpha": alpha,
                    "hessian_fd_eps": hessian_fd_eps,
                }
            )

        prior = PostflopActionModel(base_prior, tuple(float(x) for x in theta))
        last_per_hand = _e_step_postflop_all_with_progress(
            bundles_list,
            prior,
            log=LOG,
            label=f"{iter_label} | sync q(c) for next iter",
        )

    if not last_per_hand:
        prior = PostflopActionModel(base_prior, tuple(float(x) for x in theta))
        last_per_hand = _e_step_postflop_all_with_progress(
            bundles_list,
            prior,
            log=LOG,
            label="Postflop Newton | final E-step (fallback)",
        )

    LOG.info(
        "Postflop Newton finished | iters_used=%d | theta_post=[%.6f, %.6f, %.6f]",
        completed,
        float(theta[0]),
        float(theta[1]),
        float(theta[2]),
    )
    return theta, last_per_hand
