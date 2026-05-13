"""Newton optimization on the marginal preflop log-likelihood for ``theta_pre`` (MAP with L2)."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.action.preflop import PreflopActionModel, canonical_preflop_action
from utils.em.preflop import (
    PreflopEMHandBundle,
    _accumulate_theta_pre_grad,
    e_step_hand_class_posterior,
)
from utils.newton.common import (
    NEWTON_HESSIAN_FD_EPS,
    NEWTON_HESSIAN_SHIFT_RIDGE,
    backtracking_line_search,
    hessian_from_gradient_fd,
    log_sum_exp,
    newton_maximization_direction,
)
from utils.prior.preflop import PreflopPrior
from utils.strength.preflop import all_169_classes

LOG = logging.getLogger(__name__)


def _bundle_progress_stride(n_bundles: int) -> int:
    """~25 INFO ticks over a full pass for large ``n_bundles``."""
    if n_bundles <= 0:
        return 1
    return max(1, n_bundles // 25)


def _e_step_all_bundles_with_progress(
    bundles: List[PreflopEMHandBundle],
    model: PreflopActionModel,
    *,
    log: logging.Logger,
    label: str,
) -> List[Dict[str, float]]:
    """Run preflop E-step per bundle with periodic INFO (expensive for large ``n``)."""
    n = len(bundles)
    if n == 0:
        return []
    stride = _bundle_progress_stride(n)
    t0 = time.perf_counter()
    out: List[Dict[str, float]] = []
    for bi, b in enumerate(bundles):
        out.append(e_step_hand_class_posterior(b, model))
        if bi == 0 or bi == n - 1 or (bi + 1) % stride == 0:
            log.info(
                "%s | E-step bundle %d/%d | elapsed_s=%.2f",
                label,
                bi + 1,
                n,
                time.perf_counter() - t0,
            )
    log.info("%s | E-step finished | bundles=%d | wall_s=%.2f", label, n, time.perf_counter() - t0)
    return out


def _preflop_bundle_log_marginal_evidence(bundle: PreflopEMHandBundle, model: PreflopActionModel) -> float:
    """``log sum_h pi_0(h) prod_t P(a_t | h, s_t, theta)`` for one hand bundle."""
    log_terms: List[float] = []
    for h in all_169_classes():
        p0 = bundle.initial_range.get(h, 0.0)
        if p0 <= 0.0:
            continue
        logp = float(np.log(p0))
        for dec in bundle.decisions:
            probs = model.action_probs(h, dec.state_key)
            a = canonical_preflop_action(dec.action_bucket)
            logp += float(np.log(max(probs[a], 1e-300)))
        log_terms.append(logp)
    if not log_terms:
        return float("-inf")
    return log_sum_exp(log_terms)


def preflop_map_objective(
    bundles: Sequence[PreflopEMHandBundle],
    baseline: PreflopPrior,
    theta_vec: np.ndarray,
    *,
    l2: float,
    log: Optional[logging.Logger] = None,
    log_label: str = "preflop_map_objective",
) -> float:
    """Observed-data log-likelihood plus Gaussian log-prior ``-(l2/2)||theta||^2`` (same MAP penalty as EM)."""
    model = PreflopActionModel(baseline, tuple(float(x) for x in theta_vec))
    n = len(bundles)
    ell = 0.0
    t0 = time.perf_counter()
    if log is not None and n > 0:
        stride = _bundle_progress_stride(n)
        log.info("%s | MAP objective | summing marginal log-evidence over %d bundles…", log_label, n)
        for bi, b in enumerate(bundles):
            ell += _preflop_bundle_log_marginal_evidence(b, model)
            if bi == 0 or bi == n - 1 or (bi + 1) % stride == 0:
                log.info(
                    "%s | MAP objective | bundle %d/%d | partial_ell=%.4f | elapsed_s=%.2f",
                    log_label,
                    bi + 1,
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
            ell += _preflop_bundle_log_marginal_evidence(b, model)
    return ell - 0.5 * float(l2) * float(np.dot(theta_vec, theta_vec))


def preflop_map_gradient(
    bundles: List[PreflopEMHandBundle],
    baseline: PreflopPrior,
    theta_vec: np.ndarray,
    tilt_eval: PreflopActionModel,
    *,
    l2: float,
    log: Optional[logging.Logger] = None,
    log_label: str = "preflop_map_gradient",
) -> np.ndarray:
    """Gradient of :func:`preflop_map_objective` at ``theta_vec`` (posterior ``q(h|data,theta)`` each bundle)."""
    model = PreflopActionModel(baseline, tuple(float(x) for x in theta_vec))
    n = len(bundles)
    if log is not None and n > 0:
        log.info("%s | MAP gradient | E-step over %d bundles (169 classes each)…", log_label, n)
        q_by_hand = _e_step_all_bundles_with_progress(bundles, model, log=log, label=log_label)
        log.info("%s | MAP gradient | accumulating tilt-score terms…", log_label)
    else:
        q_by_hand = [e_step_hand_class_posterior(b, model) for b in bundles]
    batch_ix = np.arange(n, dtype=int)
    t_acc = time.perf_counter()
    grad, *_ = _accumulate_theta_pre_grad(
        bundles,
        q_by_hand,
        batch_ix,
        baseline,
        tilt_eval,
        theta_vec,
        scale=1.0,
        profile_first_iter=False,
    )
    grad -= float(l2) * theta_vec
    if log is not None:
        log.info(
            "%s | MAP gradient | done | |grad|=%.5f | tilt_accum_s=%.2f",
            log_label,
            float(np.linalg.norm(grad)),
            time.perf_counter() - t_acc,
        )
    return grad


def run_preflop_newton(
    bundles: List[PreflopEMHandBundle],
    *,
    prior_floor: float = 0.01,
    beta_preflop: np.ndarray | None = None,
    theta_init: Sequence[float] | None = None,
    max_newton_iters: int = 15,
    m_l2: float = 0.25,
    hessian_fd_eps: float = NEWTON_HESSIAN_FD_EPS,
    hessian_shift_ridge: float = NEWTON_HESSIAN_SHIFT_RIDGE,
    grad_norm_tol: float = 0.1,
    history_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    bundle_meta: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Directly maximize marginal log ``p(actions | theta)`` minus L2 (MAP), mirroring :func:`utils.em.preflop.run_preflop_em`.

    At each iteration the latent posteriors ``q(h)`` match the EM E-step at the **current** ``theta``, and the
    tilt gradient matches the marginal log-likelihood gradient (Louis / missing-information identity).

    Returns ``(theta_pre, q_by_hand)`` where ``q_by_hand`` is the posterior from the final ``theta``.
    """
    theta = (
        np.zeros(3, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).copy()
    )
    if beta_preflop is None:
        baseline = PreflopPrior(floor=prior_floor)
    else:
        baseline = PreflopPrior(
            floor=prior_floor,
            beta_preflop=np.asarray(beta_preflop, dtype=float),
        )
    tilt_eval = PreflopActionModel(baseline, (0.0, 0.0, 0.0))

    last_q: List[Dict[str, float]] = []
    completed = 0
    for outer in range(max_newton_iters):
        completed = outer + 1
        iter_label = f"Preflop Newton iter {completed}/{max_newton_iters}"
        t0 = time.perf_counter()

        def _obj(tv: np.ndarray) -> float:
            return preflop_map_objective(
                bundles,
                baseline,
                tv,
                l2=m_l2,
                log=LOG,
                log_label=f"{iter_label} | MAP objective",
            )

        def _grad(tv: np.ndarray) -> np.ndarray:
            return preflop_map_gradient(
                bundles,
                baseline,
                tv,
                tilt_eval,
                l2=m_l2,
                log=LOG,
                log_label=f"{iter_label} | MAP gradient",
            )

        LOG.info(
            "Preflop Newton iter %d/%d | %d bundles | l2=%g | sub-steps: objective → gradient → "
            "Hessian FD (6× grad) → line search",
            completed,
            max_newton_iters,
            len(bundles),
            m_l2,
        )
        if history_hook is not None:
            history_hook(
                {
                    "kind": "preflop_newton_iter_start",
                    "newton_iter": outer,
                    "max_newton_iters": max_newton_iters,
                    "n_bundles": len(bundles),
                    "theta_pre": [float(x) for x in theta],
                }
            )

        LOG.info("%s | (1/4) MAP objective F(θ) at current θ…", iter_label)
        t_sub = time.perf_counter()
        f0 = _obj(theta)
        LOG.info("%s | (1/4) MAP objective done | F=%.6f | wall_s=%.2f", iter_label, f0, time.perf_counter() - t_sub)

        LOG.info("%s | (2/4) MAP gradient ∇F(θ) (dominant cost: E-step per bundle)…", iter_label)
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
            model = PreflopActionModel(baseline, tuple(float(x) for x in theta))
            last_q = _e_step_all_bundles_with_progress(
                bundles,
                model,
                log=LOG,
                label=f"{iter_label} | sync q(h) after early stop",
            )
            LOG.info(
                "Preflop Newton early stop | iter %d/%d | |grad|=%.2e | elapsed_s=%.2f",
                outer + 1,
                max_newton_iters,
                gn,
                time.perf_counter() - t0,
            )
            if history_hook is not None:
                history_hook(
                    {
                        "kind": "preflop_newton_step",
                        "newton_iter": outer,
                        "theta_pre": [float(x) for x in theta],
                        "grad_norm": gn,
                        "map_objective": f0,
                        "early_stop": True,
                    }
                )
            break

        LOG.info("%s | (3/4) Hessian via finite differences on ∇F (six extra MAP gradients)…", iter_label)
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
            "%s | Newton direction | |d|=%.5f | (4/4) backtracking line search on F(θ+αd)…",
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
                "Preflop Newton line search failed at iter %d; keeping theta | |grad|=%.4f",
                outer + 1,
                gn,
            )
            model = PreflopActionModel(baseline, tuple(float(x) for x in theta))
            last_q = _e_step_all_bundles_with_progress(
                bundles,
                model,
                log=LOG,
                label=f"{iter_label} | sync q(h) after line search failure",
            )
            if history_hook is not None:
                history_hook(
                    {
                        "kind": "preflop_newton_step",
                        "newton_iter": outer,
                        "theta_pre": [float(x) for x in theta],
                        "grad_norm": gn,
                        "map_objective": f0,
                        "line_search_alpha": alpha,
                        "line_search_failed": True,
                    }
                )
            break

        theta = theta_try
        LOG.info(
            "Preflop Newton step | iter %d/%d | alpha=%.4g | F: %.4f -> %.4f | theta=[%.5f, %.5f, %.5f] | |g|=%.4f | wall_s=%.2f",
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
                    "kind": "preflop_newton_step",
                    "newton_iter": outer,
                    "theta_pre": [float(x) for x in theta],
                    "grad_norm": gn,
                    "map_objective": f1,
                    "line_search_alpha": alpha,
                    "hessian_fd_eps": hessian_fd_eps,
                }
            )

        model = PreflopActionModel(baseline, tuple(float(x) for x in theta))
        last_q = _e_step_all_bundles_with_progress(
            bundles,
            model,
            log=LOG,
            label=f"{iter_label} | sync q(h) for next iter",
        )

    if not last_q:
        model = PreflopActionModel(baseline, tuple(float(x) for x in theta))
        last_q = _e_step_all_bundles_with_progress(
            bundles,
            model,
            log=LOG,
            label="Preflop Newton | final E-step (fallback)",
        )

    LOG.info(
        "Preflop Newton finished | iters_used=%d | theta_pre=[%.6f, %.6f, %.6f]",
        completed,
        float(theta[0]),
        float(theta[1]),
        float(theta[2]),
    )
    return theta, last_q
