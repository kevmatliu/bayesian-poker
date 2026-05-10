"""Postflop EM over combo keys and theta_post."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.em.common import normalize_log_weights
from utils.prior.postflop import PostflopFeatures, PostflopPrior


@dataclass(frozen=True)
class PostflopThetaObservation:
    """One candidate combo in a hand: prior mass hint + sequence of decisions."""

    combo_key: str
    log_prior_range: float
    decisions: Tuple[Tuple[PostflopFeatures, int], ...]


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
) -> np.ndarray:
    theta = np.asarray(theta_init, dtype=float).copy()

    for _ in range(steps):
        live = PostflopPrior(
            theta_post=tuple(float(x) for x in theta),
            floor=prior_template.floor,
            beta_facing=prior_template.beta_facing_matrix,
            beta_no_bet=prior_template.beta_no_bet_matrix,
        )
        g = postflop_theta_gradient(live, hands, q_by_hand, l2=l2)
        theta += lr * g
        if center_each_step:
            theta -= float(np.mean(theta))
        theta = np.clip(theta, -clip, clip)

    return theta


def run_postflop_theta_em(
    observations_by_hand: Sequence[List[PostflopThetaObservation]],
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
    history_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    hand_meta: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    theta = (
        np.zeros(3, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).copy()
    )

    last_per_hand: List[Dict[str, float]] = []

    for outer in range(num_em_iters):
        prior = PostflopPrior(
            theta_post=tuple(float(x) for x in theta),
            floor=prior_floor,
            beta_facing=beta_facing,
            beta_no_bet=beta_no_bet,
        )

        last_per_hand = []
        for hi, hand_obs in enumerate(observations_by_hand):
            q = e_step_combo_posterior(hand_obs, prior)
            last_per_hand.append(dict(q))
            if history_hook is not None:
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
                        "theta_post": [float(x) for x in theta],
                        "max_q": float(max(q.values())) if q else None,
                        "ess": float(ess),
                        **meta,
                    }
                )

        base_prior = PostflopPrior(
            floor=prior_floor,
            beta_facing=beta_facing,
            beta_no_bet=beta_no_bet,
        )

        theta = m_step_theta_post_gradient_ascent(
            base_prior,
            observations_by_hand,
            last_per_hand,
            theta_init=theta,
            l2=m_l2,
            lr=m_lr,
            steps=m_steps,
            clip=clip,
            center_each_step=True,
        )

        if history_hook is not None:
            history_hook(
                {
                    "kind": "postflop_m_step",
                    "em_outer": outer,
                    "em_timestep": outer,
                    "theta_post": [float(x) for x in theta],
                    "n_hands": len(observations_by_hand),
                }
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
