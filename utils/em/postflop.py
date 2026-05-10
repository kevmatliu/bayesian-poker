"""Postflop EM over combo keys and theta_post."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.em.common import M_STEP_GRAD_NORM_TOL, normalize_log_weights
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
    """q(c) ∝ pi_0(c) * prod_t P_theta(a_t | x_t(c)); pi_0 is the exact combo probability vector."""
    log_q: Dict[str, float] = {}
    for combo, p0 in bundle.initial_combo_range.items():
        if p0 <= 0.0:
            continue
        logp = math.log(p0)
        ok = True
        for step in bundle.decisions:
            feat_map = dict(step.features_by_combo)
            feat = feat_map.get(combo)
            if feat is None:
                ok = False
                break
            probs = prior.action_probs(feat)
            logp += math.log(max(probs.get(step.action, 0.0), 1e-300))
        if not ok:
            continue
        log_q[combo] = logp
    if not log_q:
        raise ValueError("E-step: no combo with positive prior mass and full feature coverage")
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
) -> np.ndarray:
    grad = np.zeros(3, dtype=float)
    theta_vec = prior_template.theta_vec

    for bundle, qmap in zip(bundles, q_by_hand):
        for step in bundle.decisions:
            feat_map = dict(step.features_by_combo)
            action = step.action
            for combo, w in qmap.items():
                wf = float(w)
                if wf <= 0.0:
                    continue
                feat = feat_map.get(combo)
                if feat is None:
                    continue
                utilities = prior_template.action_utility_vectors(feat)
                p_theta = prior_template.action_probs(feat)
                legal = prior_template.legal_actions(feat)
                expected_u = np.zeros(3, dtype=float)
                for b in legal:
                    expected_u += float(p_theta[b]) * utilities[b]
                grad += wf * (utilities[action] - expected_u)

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
) -> Tuple[np.ndarray, int]:
    theta = np.asarray(theta_init, dtype=float).copy()
    log_every = max(1, steps // 10)
    t_m0 = time.perf_counter()
    used = steps

    for step_i in range(steps):
        live = PostflopPrior(
            theta_post=tuple(float(x) for x in theta),
            floor=prior_template.floor,
            beta_facing=prior_template.beta_facing_matrix,
            beta_no_bet=prior_template.beta_no_bet_matrix,
        )
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
    m_grad_norm_tol: float = M_STEP_GRAD_NORM_TOL,
    history_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    hand_meta: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
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

        LOG.info(
            "Postflop EM M-step start | outer %d/%d | %d gradient steps",
            outer + 1,
            num_em_iters,
            m_steps,
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
    hands = [list(observations)]
    n = len(observations)
    if n == 0:
        return np.zeros(3, dtype=float)
    w = 1.0 / n
    qmaps = [{obs.combo_key: w for obs in observations}]
    prior = PostflopPrior(floor=prior_floor)
    return postflop_theta_gradient(prior, hands, qmaps, l2=0.0)
