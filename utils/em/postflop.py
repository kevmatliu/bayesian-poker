"""Postflop EM over combo keys and theta_post."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.em.common import (
    M_STEP_GRAD_NORM_TOL,
    POSTFLOP_M_BATCH_SIZE,
    normalize_log_weights,
)
from utils.prior.postflop import PostflopFeatures, PostflopPrior

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostflopThetaObservation:
    """Legacy: one candidate combo with log-prior hint + decision sequence (e.g. supervised rows)."""

    combo_key: str
    log_prior_range: float
    decisions: Tuple[Tuple[PostflopFeatures, int], ...]


@dataclass(frozen=True)
class PostflopEMTimestep:
    """One target postflop action and per-combo features at that decision point."""

    action: int
    features_by_combo: Tuple[Tuple[str, PostflopFeatures], ...]


@dataclass(frozen=True)
class PostflopEMHandBundle:
    """Postflop EM hand: 1,326 (sparse) combo prior + target decisions (mirrors PreflopEMHandBundle)."""

    decisions: Tuple[PostflopEMTimestep, ...]
    initial_combo_range: Dict[str, float]


def e_step_combo_posterior(
    observations: Sequence[PostflopThetaObservation],
    prior: PostflopPrior,
) -> Dict[str, float]:
    """q(c) ∝ exp(log pi(c) + sum_t log P_theta(a_t | x_t(c)))."""
    log_w: Dict[str, float] = {}
    for obs in observations:
        logp = obs.log_prior_range
        for feat, action in obs.decisions:
            probs = prior.action_probs(feat)
            logp += math.log(max(probs.get(action, 0.0), 1e-300))
        log_w[obs.combo_key] = logp
    return normalize_log_weights(log_w)


def e_step_postflop_bundle(
    bundle: PostflopEMHandBundle,
    prior: PostflopPrior,
) -> Dict[str, float]:
    """``q(c) ∝ pi_0(c) * prod_t P_theta(a_t | x_t(c))``.

    Batched across combos using :meth:`PostflopPrior.action_probs_matrix`
    (Method D): for each timestep we evaluate every combo's likelihood in
    one matmul instead of one Python softmax per combo.
    """
    from utils.prior.postflop import feature_vector

    initial = bundle.initial_combo_range
    log_q: Dict[str, float] = {
        combo: math.log(p0) for combo, p0 in initial.items() if p0 > 0.0
    }
    if not log_q:
        raise ValueError("E-step: empty initial combo range")
    alive = set(log_q.keys())

    for step in bundle.decisions:
        feat_map = dict(step.features_by_combo)
        # Combos that don't appear at this step (e.g. now blocked by a
        # new community card) drop out of ``alive``.
        live = [c for c in alive if c in feat_map]
        if not live:
            raise ValueError(
                "E-step: every combo dropped out before all decisions consumed"
            )
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
        # Drop combos that fell off this step.
        alive = set(live)

    log_q = {c: v for c, v in log_q.items() if c in alive}
    if not log_q:
        raise ValueError("E-step: no combo survived to the final decision")
    return normalize_log_weights(log_q)


def postflop_theta_gradient(
    prior_template: PostflopPrior,
    hands: Sequence[Sequence[PostflopThetaObservation]],
    q_by_hand: Sequence[Mapping[str, float]],
    *,
    l2: float = 0.25,
) -> np.ndarray:
    grad = np.zeros(3, dtype=float)
    theta_vec = prior_template.theta_vec

    for hand_obs, qmap in zip(hands, q_by_hand):
        for obs in hand_obs:
            w = float(qmap.get(obs.combo_key, 0.0))
            if w <= 0.0:
                continue
            for feat, action in obs.decisions:
                utilities = prior_template.action_utility_vectors(feat)
                p_theta = prior_template.action_probs(feat)
                legal = prior_template.legal_actions(feat)
                expected_u = np.zeros(3, dtype=float)
                for b in legal:
                    expected_u += float(p_theta[b]) * utilities[b]
                grad += w * (utilities[action] - expected_u)

    grad -= l2 * theta_vec
    return grad


def postflop_theta_gradient_bundles(
    prior_template: PostflopPrior,
    bundles: Sequence[PostflopEMHandBundle],
    q_by_hand: Sequence[Mapping[str, float]],
    *,
    l2: float = 0.25,
    bundle_indices: Optional[Sequence[int]] = None,
    apply_l2: bool = True,
) -> np.ndarray:
    """Batched M-step gradient.

    For each timestep we materialise the live combos as a single
    ``(N, PHI_DIM)`` feature matrix and call
    :meth:`PostflopPrior.action_probs_matrix` once. The gradient
    contribution simplifies to ``w * (e_action - p_theta)`` after
    cancelling the ``-E_p[u]`` drift term across actions, so we can
    accumulate it with a single weighted sum per row (Method D).

    When ``bundle_indices`` is set, only those hand-bundle indices contribute
    (for minibatch M-step); use ``apply_l2=False`` and apply L2 outside when
    scaling stochastic gradients.
    """
    from utils.prior.postflop import feature_vector

    grad = np.zeros(3, dtype=float)
    theta_vec = prior_template.theta_vec
    eye = np.eye(3, dtype=float)

    if bundle_indices is None:
        index_iter = range(len(bundles))
    else:
        index_iter = bundle_indices

    for bi in index_iter:
        bundle = bundles[int(bi)]
        qmap = q_by_hand[int(bi)]
        if not qmap:
            continue
        for step in bundle.decisions:
            feat_map = dict(step.features_by_combo)
            action = int(step.action)
            live: List[Tuple[str, float, PostflopFeatures]] = []
            for combo, w in qmap.items():
                wf = float(w)
                if wf <= 0.0:
                    continue
                feat = feat_map.get(combo)
                if feat is None:
                    continue
                live.append((combo, wf, feat))
            if not live:
                continue
            phi = np.stack([feature_vector(t[2]) for t in live], axis=0)
            facing = np.fromiter(
                (t[2].facing_bet for t in live),
                dtype=bool,
                count=len(live),
            )
            weights = np.asarray([t[1] for t in live], dtype=float)
            p_theta = prior_template.action_probs_matrix(phi, facing)  # (N, 3)
            # u(action) - E_p[u] = e_action - p_theta
            diff = eye[action][None, :] - p_theta  # (N, 3)
            grad += (weights[:, None] * diff).sum(axis=0)

    if apply_l2:
        grad -= l2 * theta_vec
    return grad


def m_step_theta_post_gradient_ascent(
    prior_template: PostflopPrior,
    hands: Sequence[Sequence[PostflopThetaObservation]],
    q_by_hand: Sequence[Mapping[str, float]],
    *,
    theta_init: Sequence[float],
    l2: float = 0.25,
    lr: float = 0.05,
    steps: int = 200,
    clip: float = 3.0,
    center_each_step: bool = True,
    grad_norm_tol: float = M_STEP_GRAD_NORM_TOL,
) -> Tuple[np.ndarray, int]:
    theta = np.asarray(theta_init, dtype=float).copy()
    used = steps

    for step_i in range(steps):
        live = PostflopPrior(
            theta_post=tuple(float(x) for x in theta),
            floor=prior_template.floor,
            beta_facing=prior_template.beta_facing_matrix,
            beta_no_bet=prior_template.beta_no_bet_matrix,
        )
        g = postflop_theta_gradient(live, hands, q_by_hand, l2=l2)
        gn = float(np.linalg.norm(g))
        if gn < grad_norm_tol:
            used = step_i + 1
            LOG.info(
                "Postflop EM M-step (observation) early stop | iter %d/%d | |grad|=%.2e < %.2e",
                step_i + 1,
                steps,
                gn,
                grad_norm_tol,
            )
            break
        theta += lr * g
        if center_each_step:
            theta -= float(np.mean(theta))
        theta = np.clip(theta, -clip, clip)

    return theta, used


def m_step_theta_post_gradient_ascent_bundles(
    prior_template: PostflopPrior,
    bundles: Sequence[PostflopEMHandBundle],
    q_by_hand: Sequence[Mapping[str, float]],
    *,
    theta_init: Sequence[float],
    l2: float = 0.25,
    lr: float = 0.05,
    steps: int = 200,
    clip: float = 3.0,
    center_each_step: bool = True,
    outer_1based: int = 1,
    num_outer_iterations: int = 1,
    grad_norm_tol: float = M_STEP_GRAD_NORM_TOL,
    m_batch_size: int = POSTFLOP_M_BATCH_SIZE,
    m_step_seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    theta = np.asarray(theta_init, dtype=float).copy()
    log_every = max(1, steps // 10)
    t_m0 = time.perf_counter()
    used = steps

    n_hands = len(bundles)
    rng = np.random.default_rng(m_step_seed)
    if m_batch_size <= 0 or m_batch_size >= n_hands:
        bundle_batch_cap = n_hands
        use_minibatch = False
    else:
        bundle_batch_cap = int(m_batch_size)
        use_minibatch = True

    for step_i in range(steps):
        live = PostflopPrior(
            theta_post=tuple(float(x) for x in theta),
            floor=prior_template.floor,
            beta_facing=prior_template.beta_facing_matrix,
            beta_no_bet=prior_template.beta_no_bet_matrix,
        )
        if use_minibatch:
            bsz = min(bundle_batch_cap, n_hands)
            batch_ix = rng.choice(n_hands, size=bsz, replace=False)
            scale = n_hands / float(bsz)
            g_data = postflop_theta_gradient_bundles(
                live,
                bundles,
                q_by_hand,
                l2=l2,
                bundle_indices=batch_ix,
                apply_l2=False,
            )
            g = scale * g_data - l2 * live.theta_vec
        else:
            g = postflop_theta_gradient_bundles(live, bundles, q_by_hand, l2=l2)
        gn = float(np.linalg.norm(g))
        if gn < grad_norm_tol:
            used = step_i + 1
            LOG.info(
                "Postflop EM M-step early stop | outer %d/%d | grad %d/%d | |grad|=%.2e < %.2e | "
                "theta=[%.5f, %.5f, %.5f] | elapsed_s=%.2f",
                outer_1based,
                num_outer_iterations,
                step_i + 1,
                steps,
                gn,
                grad_norm_tol,
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
                time.perf_counter() - t_m0,
            )
            break
        theta += lr * g
        if center_each_step:
            theta -= float(np.mean(theta))
        theta = np.clip(theta, -clip, clip)
        if step_i == 0 or step_i == steps - 1 or (step_i + 1) % log_every == 0:
            LOG.info(
                "Postflop EM M-step | outer %d/%d | grad %d/%d | "
                "theta=[%.5f, %.5f, %.5f] | |grad|=%.5f | elapsed_s=%.2f",
                outer_1based,
                num_outer_iterations,
                step_i + 1,
                steps,
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
                gn,
                time.perf_counter() - t_m0,
            )

    return theta, used


def run_postflop_theta_em(
    bundles_by_hand: Sequence[PostflopEMHandBundle],
    *,
    prior_floor: float = 1e-6,
    theta_init: Sequence[float] | None = None,
    beta_facing: np.ndarray | None = None,
    beta_no_bet: np.ndarray | None = None,
    num_em_iters: int = 10,
    m_lr: float = 0.05,
    m_steps: int = 200,
    m_l2: float = 0.25,
    clip: float = 3.0,
    m_batch_size: int = POSTFLOP_M_BATCH_SIZE,
    m_step_seed: Optional[int] = None,
    m_grad_norm_tol: float = M_STEP_GRAD_NORM_TOL,
    history_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    hand_meta: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Outer EM for postflop tendency ``theta_post`` with frozen ``beta`` matrices.

    For each outer iteration: E-step per hand-bundle via
    :func:`e_step_postflop_bundle` (batched combo likelihoods), then M-step via
    :func:`m_step_theta_post_gradient_ascent_bundles`. ``history_hook`` can
    record per-hand ESS / max ``q`` for diagnostics; large runs subsample
    ``postflop_e_step`` events to limit disk IO.

    Returns:
        Tuple of ``(theta_post array shape (3,), list of per-hand posterior dicts
        from the final outer iteration)``.
    """
    theta = (
        np.zeros(3, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).copy()
    )

    last_per_hand: List[Dict[str, float]] = []
    bundles_list = list(bundles_by_hand)
    n_h = len(bundles_list)

    for outer in range(num_em_iters):
        t0 = time.perf_counter()
        LOG.info(
            "Postflop EM outer %d/%d | %d hand-bundles | m_steps=%d lr=%g l2=%g",
            outer + 1,
            num_em_iters,
            n_h,
            m_steps,
            m_lr,
            m_l2,
        )
        if history_hook is not None:
            history_hook(
                {
                    "kind": "postflop_em_outer_start",
                    "em_outer": outer,
                    "num_em_iters": num_em_iters,
                    "n_hands": n_h,
                    "m_steps": m_steps,
                    "m_lr": m_lr,
                    "m_l2": m_l2,
                    "m_batch_size": int(m_batch_size),
                    "theta_post": [float(x) for x in theta],
                }
            )

        prior = PostflopPrior(
            theta_post=tuple(float(x) for x in theta),
            floor=prior_floor,
            beta_facing=beta_facing,
            beta_no_bet=beta_no_bet,
        )

        last_per_hand = []
        prog = max(1, n_h // 20) if n_h > 20 else 1
        pf_stride = 1
        if history_hook is not None and n_h > 2000:
            pf_stride = max(1, n_h // 2000)
            LOG.info(
                "Postflop EM | jsonl subsample: postflop_e_step every %d hand(s) (n_hands=%d)",
                pf_stride,
                n_h,
            )
        t_e0 = time.perf_counter()
        for hi, bundle in enumerate(bundles_list):
            q = e_step_postflop_bundle(bundle, prior)
            last_per_hand.append(dict(q))
            if history_hook is not None and (
                hi == 0 or hi == n_h - 1 or hi % pf_stride == 0
            ):
                meta: Dict[str, Any] = {}
                if hand_meta is not None and hi < len(hand_meta):
                    meta = dict(hand_meta[hi])
                qv = list(q.values())
                ess = (sum(qv) ** 2) / sum(v * v for v in qv) if qv else 0.0
                history_hook(
                    {
                        "kind": "postflop_e_step",
                        "em_outer": outer,
                        "em_timestep": outer,
                        "hand_index_in_batch": hi,
                        "e_step_subsample_stride": pf_stride,
                        "theta_post": [float(x) for x in theta],
                        "max_q": float(max(q.values())) if q else None,
                        "ess": float(ess),
                        **meta,
                    }
                )
            if hi == 0 or hi == n_h - 1 or (hi + 1) % prog == 0:
                LOG.info(
                    "Postflop EM E-step | outer %d/%d | hand %d/%d | elapsed_s=%.2f",
                    outer + 1,
                    num_em_iters,
                    hi + 1,
                    n_h,
                    time.perf_counter() - t_e0,
                )

        max_ess = 0.0
        if last_per_hand:
            for qm in last_per_hand:
                qv = list(qm.values())
                if qv:
                    max_ess = max(max_ess, (sum(qv) ** 2) / sum(v * v for v in qv))
        LOG.info(
            "Postflop EM E-step done | outer %d/%d | wall_s=%.2f | max_ess≈%.2f",
            outer + 1,
            num_em_iters,
            time.perf_counter() - t_e0,
            max_ess,
        )
        if history_hook is not None:
            history_hook(
                {
                    "kind": "postflop_e_step_finished",
                    "em_outer": outer,
                    "n_hands": n_h,
                    "e_step_subsample_stride": pf_stride,
                    "e_step_wall_s": round(time.perf_counter() - t_e0, 4),
                    "max_ess": float(max_ess),
                    "theta_post": [float(x) for x in theta],
                }
            )

        base_prior = PostflopPrior(
            floor=prior_floor,
            beta_facing=beta_facing,
            beta_no_bet=beta_no_bet,
        )

        nh_m = len(bundles_list)
        mb_eff = (
            "full"
            if m_batch_size <= 0 or m_batch_size >= nh_m
            else f"{min(m_batch_size, nh_m)}/{nh_m} hands/step"
        )
        LOG.info(
            "Postflop EM M-step start | outer %d/%d | %d gradient steps | batch=%s",
            outer + 1,
            num_em_iters,
            m_steps,
            mb_eff,
        )
        theta, post_m_used = m_step_theta_post_gradient_ascent_bundles(
            base_prior,
            bundles_list,
            last_per_hand,
            theta_init=theta,
            l2=m_l2,
            lr=m_lr,
            steps=m_steps,
            clip=clip,
            center_each_step=True,
            outer_1based=outer + 1,
            num_outer_iterations=num_em_iters,
            grad_norm_tol=m_grad_norm_tol,
            m_batch_size=m_batch_size,
            m_step_seed=m_step_seed,
        )

        if history_hook is not None:
            history_hook(
                {
                    "kind": "postflop_m_step",
                    "em_outer": outer,
                    "em_timestep": outer,
                    "theta_post": [float(x) for x in theta],
                    "n_hands": len(bundles_list),
                    "m_gradient_steps_used": post_m_used,
                    "m_gradient_steps_cap": m_steps,
                    "m_batch_size": int(m_batch_size),
                    "outer_wall_s": round(time.perf_counter() - t0, 4),
                }
            )

        LOG.info(
            "Postflop EM outer %d/%d finished | theta_post=[%.6f, %.6f, %.6f] | "
            "M-step grad iters used=%d/%d | outer_wall_s=%.2f",
            outer + 1,
            num_em_iters,
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            post_m_used,
            m_steps,
            time.perf_counter() - t0,
        )

    return theta, last_per_hand


def single_hand_em_gradient_sample(
    observations: Sequence[PostflopThetaObservation],
    *,
    prior_floor: float = 0.0,
) -> np.ndarray:
    """Debug helper: M-step gradient for ``theta_post`` assuming **uniform** ``q`` over combos.

    Wraps :func:`postflop_theta_gradient` with synthetic equal weights. Not
    used in production EM (which uses :func:`e_step_combo_posterior` /
    :func:`e_step_postflop_bundle`), but handy for sanity checks.
    """
    hands = [list(observations)]
    n = len(observations)
    if n == 0:
        return np.zeros(3, dtype=float)
    w = 1.0 / n
    qmaps = [{obs.combo_key: w for obs in observations}]
    prior = PostflopPrior(floor=prior_floor)
    return postflop_theta_gradient(prior, hands, qmaps, l2=0.0)
