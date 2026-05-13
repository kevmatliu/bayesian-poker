"""Postflop EM: posterior over hole-card combos and ``theta_post``.

E-step: :func:`e_step_postflop_bundle` updates ``q(c)`` with batched
:meth:`~utils.action.postflop.PostflopActionModel.action_probs_matrix` per street.
M-step: :func:`m_step_theta_post_gradient_ascent_bundles` does tilt-parameter ascent
with optional hand-level minibatches (same scaling contract as preflop).


Learning the theta_post tilt.

E-step: e_step_preflop_bundle which updates q(c) with batched PostflopActionModel.action_probs_matrix per street.

M_step: m_step_theta_post_gradient_ascent_bundles, using a gradient ascent-modified 
version of EM due to the high-cost of the computational methods to compute the M-step. 

Instead of calculating M exactly, we take a gradient step toward the EM objective.
"""

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
    effective_sample_size,
    max_effective_sample_size,
    minibatch_plan,
    normalize_log_weights,
)
from utils.action.postflop import (
    PostflopActionModel,
    PostflopFeatures,
    feature_vector,
)
from utils.prior.postflop import PostflopPrior

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostflopThetaObservation:
    """One candidate combo with log-prior hint + decision sequence (e.g. supervised rows)."""

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
    """All EM inputs for one Pluribus hand when inferring the target's latent combo, c.

    What: Same bundling idea as Preflop, but the latent state is a concrete two-card **combo key** 
    (1,326 universe, usually stored sparsely). Starting support uses one observer's known hole cards.
    Each :class:`PostflopEMTimestep` records one target postflop action plus the
    per-combo feature rows at that decision point, so the likelihood
    P(a_t | x_t(c), \theta) can be evaluated for every combo still
    alive on the board.

    Rationale: Coupling the decisions with the initial range, allowing us to stack
    the gradients more easily.
    """

    decisions: Tuple[PostflopEMTimestep, ...]
    initial_combo_range: Dict[str, float]


def e_step_combo_posterior(
    observations: Sequence[PostflopThetaObservation],
    prior: PostflopActionModel,
) -> Dict[str, float]:
    """q(c) ∝ exp(log pi(c) + sum_t log P_theta(a_t | x_t(c)))."""
    log_w: Dict[str, float] = {}                                   # unnormalized log weights per hole-card combo
    for obs in observations:                                       # typically all 1326 or a sparse subset
        logp = obs.log_prior_range                                 # log pi_0(c) from range construction
        for feat, action in obs.decisions:                         # supervised-like trajectory for this combo
            probs = prior.action_probs(feat)
            logp += math.log(max(probs.get(action, 0.0), 1e-300))  # avoid log(0) underflow (pls don't penalize :( )
        log_w[obs.combo_key] = logp
    return normalize_log_weights(log_w)                            # softmax over combos → q(c)


def e_step_postflop_bundle(
    bundle: PostflopEMHandBundle,
    prior: PostflopActionModel,
) -> Dict[str, float]:
    """``q(c) propto pi_0(c) * prod_t P_theta(a_t | x_t(c))``.

    Batched across combos using `PostflopActionModel.action_probs_matrix`.
    For each timestep we evaluate every combo's likelihood in one matmul instead of one Python softmax per combo.
    """
    initial = bundle.initial_combo_range                                     # sparse prior support over combo strings
    log_q: Dict[str, float] = {
        combo: math.log(p0) for combo, p0 in initial.items() if p0 > 0.0
    }                                                                        # start log q at log pi_0 only
    if not log_q:
        raise ValueError("E-step: empty initial combo range")
    alive = set(log_q.keys())                                                # combos still consistent with board / blockers so far

    for step in bundle.decisions:                                            # each street decision for the target seat
        feat_map = dict(step.features_by_combo)                              
        # keeping track of the ``live`` combos that survive each street. 
        live = [c for c in alive if c in feat_map]                           # intersect prior support with this timestep
        if not live:
            raise ValueError(
                "E-step: every combo dropped out before all decisions consumed"
            )
        phi = np.stack([feature_vector(feat_map[c]) for c in live], axis=0)  # (N, PHI_DIM) batch
        facing = np.fromiter(
            (feat_map[c].facing_bet for c in live),
            dtype=bool,
            count=len(live),
        )                                                                    # row mask: facing-bet vs no-bet softmax head
        log_probs = prior.action_log_probs_matrix(phi, facing)               # batched log softmax rows
        log_pa = log_probs[:, int(step.action)]                              # column for observed action only
        for combo, lp in zip(live, log_pa):
            log_q[combo] += float(lp)                                        # accumulate log P(a_t | x_t(c), theta)
        # Drop combos that fell off this step.
        alive = set(live)                                                    # shrink support to combos that existed at this node

    log_q = {c: v for c, v in log_q.items() if c in alive}                   # discard dead keys for safety
    if not log_q:
        raise ValueError("E-step: no combo survived to the final decision")
    return normalize_log_weights(log_q)


def postflop_theta_gradient(
    prior_template: PostflopActionModel,
    hands: Sequence[Sequence[PostflopThetaObservation]],
    q_by_hand: Sequence[Mapping[str, float]],
    *,
    l2: float = 0.25,
) -> np.ndarray:
    """
    Gradient calculation for theta_post EM M-step.

    Note that we regularize the gradient with an L2 penalty, effectively giving a Gaussian prior on theta_post.
    Helps to stabilize the gradient ascent updates.
    """    

    grad = np.zeros(3, dtype=float)                                      # tilt gradient in R^3
    theta_vec = prior_template.theta_vec                                 # current theta for L2 anchor

    for hand_obs, qmap in zip(hands, q_by_hand):                         # one list of combo-rows per hand
        for obs in hand_obs:                                             # each combo's feature trajectory
            w = float(qmap.get(obs.combo_key, 0.0))                      # posterior weight q(c)
            if w <= 0.0:
                continue
            for feat, action in obs.decisions:                           # contrastive term per timestep
                utilities = prior_template.action_utility_vectors(feat)  # centered behavior vectors
                p_theta = prior_template.action_probs(feat)              # model softmax at theta
                legal = prior_template.legal_actions(feat)               # {0,1,2} subset depending on node
                expected_u = np.zeros(3, dtype=float)                    # E_p[u] under model
                for b in legal:
                    expected_u += float(p_theta[b]) * utilities[b]
                grad += w * (utilities[action] - expected_u)             # policy-gradient style update

    grad -= l2 * theta_vec                                               # 
    return grad


def postflop_theta_gradient_bundles(
    prior_template: PostflopActionModel,
    bundles: Sequence[PostflopEMHandBundle],
    q_by_hand: Sequence[Mapping[str, float]],
    *,
    l2: float = 0.25,
    bundle_indices: Optional[Sequence[int]] = None,     # for minibatch M-step; if None, use all bundles for full-batch gradient
    apply_l2: bool = True,
) -> np.ndarray:
    """Batched M-step gradient.

    For each timestep, we group live combos as a single ``(N, PHI_DIM)`` feature matrix and 
    call `action_probs_matrix` once. The gradient contribution simplifies to 
    ``w * (e_action - p_theta)`` after cancelling the ``-E_p[u]`` drift term across actions, 
    so we can accumulate it with a single weighted sum per row.
    """
    grad = np.zeros(3, dtype=float)                                    # accumulates full-data or minibatch gradient
    theta_vec = prior_template.theta_vec
    eye = np.eye(3, dtype=float)                                       # one-hot rows for closed-form utility difference

    if bundle_indices is None:
        index_iter = range(len(bundles))                               # all hands
    else:
        index_iter = bundle_indices                                    # minibatch subset only

    for bi in index_iter:
        bundle = bundles[int(bi)]
        qmap = q_by_hand[int(bi)]
        if not qmap:
            continue
        for step in bundle.decisions:                                  # one matrix multiply covers all live combos
            feat_map = dict(step.features_by_combo)
            action = int(step.action)                                  # global action index (fold/call/raise)
            live: List[Tuple[str, float, PostflopFeatures]] = []
            for combo, w in qmap.items():
                wf = float(w)
                if wf <= 0.0:
                    continue
                feat = feat_map.get(combo)                             # None if combo blocked this street
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
            weights = np.asarray([t[1] for t in live], dtype=float)    # q(c) per row
            p_theta = prior_template.action_probs_matrix(phi, facing)  # (N, 3) softmax rows
            # u(action) - E_p[u] = e_action - p_theta
            diff = eye[action][None, :] - p_theta                      # (N, 3) contrast per combo
            grad += (weights[:, None] * diff).sum(axis=0)              # sum weighted rows → dQ/dtheta

    if apply_l2:
        grad -= l2 * theta_vec                                         # skipped inside minibatch; caller adds scaled L2
    return grad


def m_step_theta_post_gradient_ascent(
    beta_source: PostflopPrior,
    hands: Sequence[Sequence[PostflopThetaObservation]],
    q_by_hand: Sequence[Mapping[str, float]],
    *,
    theta_init: Sequence[float],
    l2: float = 0.25,
    lr: float = 0.05,
    steps: int = 200,
    clip: float = 10000.,
    center_each_step: bool = True,
    grad_norm_tol: float = M_STEP_GRAD_NORM_TOL,
) -> Tuple[np.ndarray, int]:
    """
    M-step for theta_post using gradient ascent on the expected complete-data log-likelihood.

    We use gradient ascent instead of exact maximization due to the complexity of the M-step objective, which involves a sum over all hands and combos.
    The gradient is computed in `postflop_theta_gradient`, which implements a policy-gradient style update based on the current posterior weights q(c) and the model's action probabilities.
    """

    theta = np.asarray(theta_init, dtype=float).copy()
    used = steps                                                    # actual iterations if early-stopped

    for step_i in range(steps):
        live = PostflopActionModel(
            beta_source,
            tuple(float(x) for x in theta),
        )                                                           # rebuild so softmax uses fresh theta each step
        g = postflop_theta_gradient(live, hands, q_by_hand, l2=l2)  # full-batch grad
        gn = float(np.linalg.norm(g))                               # Frobenius norm for stopping
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
        theta += lr * g                                             # ascent on expected complete-data log-likelihood
        if center_each_step:
            theta -= float(np.mean(theta))                          # remove gauge / improve conditioning

    return theta, used


def m_step_theta_post_gradient_ascent_bundles(
    beta_source: PostflopPrior,
    bundles: Sequence[PostflopEMHandBundle],
    q_by_hand: Sequence[Mapping[str, float]],
    *,
    theta_init: Sequence[float],
    l2: float = 0.25,
    lr: float = 0.05,
    steps: int = 200,
    center_each_step: bool = True,
    outer_1based: int = 1,
    num_outer_iterations: int = 1,
    grad_norm_tol: float = M_STEP_GRAD_NORM_TOL,
    m_batch_size: int = POSTFLOP_M_BATCH_SIZE,      # mini-batch size for stochastic gradient ascent; 0 or negative means full batch
    m_step_seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Mini-batch method for M-step for theta_post using gradient ascent on the expected complete-data log-likelihood
    """

    theta = np.asarray(theta_init, dtype=float).copy()
    log_every = max(1, steps // 10)                                              # ~10 progress logs per full run
    t_m0 = time.perf_counter()
    used = steps

    n_hands = len(bundles)
    rng = np.random.default_rng(m_step_seed)
    use_minibatch = m_batch_size > 0 and m_batch_size < n_hands                  # hand-level SGD if true

    for step_i in range(steps):
        live = PostflopActionModel(
            beta_source,
            tuple(float(x) for x in theta),
        )
        if use_minibatch:
            batch_ix, scale, _ = minibatch_plan(n_hands, m_batch_size, rng)      # scale = n/batch
            g_data = postflop_theta_gradient_bundles(
                live,
                bundles,
                q_by_hand,
                l2=l2,
                bundle_indices=batch_ix,
                apply_l2=False,
            )                                                                    # L2 applied manually next line for correct scaling
            g = scale * g_data - l2 * live.theta_vec                             # unbiased noisy grad + full L2
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

        if step_i == 0 or step_i == steps - 1 or (step_i + 1) % log_every == 0:  # periodic trace logging
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

    We classify the EM steps into ``outer`` iterations and ``inner`` gradient steps.
    The E-step is exact and batched per hand-bundle, but the M-step is approximate via gradient ascent.
    The M-step runs to completion each EM iteration, but it may early-stop based on the gradient norm.

    For each outer iteration: E-step updates ``q(c)`` for each hand via
    - e_step_postflop_bundle, 
    - M_step updates through m_step_theta_post_gradient_ascent_bundle
    
    history_hook records the progress of the EM run. 

    Returns:
        Tuple of ``(theta_post array shape (3,), list of per-hand posterior dicts
        from the final outer iteration)``.
    """
    theta = (
        np.zeros(3, dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).copy()
    )                                                                                 # default zero tilt unless warm-started

    last_per_hand: List[Dict[str, float]] = []
    bundles_list = list(bundles_by_hand)                                              # materialize for indexing + repeated passes
    n_h = len(bundles_list)

    for outer in range(num_em_iters):                                                 # outer EM: alternate E on q(c), M on theta
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

        base_prior = PostflopPrior(
            floor=prior_floor,
            beta_facing=beta_facing,
            beta_no_bet=beta_no_bet,
        )                                                                             # frozen population logits for this run
        prior = PostflopActionModel(
            base_prior,
            tuple(float(x) for x in theta),
        )                                                                             # softmax head includes both beta and theta

        last_per_hand = []
        prog = max(1, n_h // 20) if n_h > 20 else 1
        pf_stride = 1
        if history_hook is not None and n_h > 2000:
            pf_stride = max(1, n_h // 2000)                                           # logging the jsonl events on huge batches
            LOG.info(
                "Postflop EM | jsonl subsample: postflop_e_step every %d hand(s) (n_hands=%d)",
                pf_stride,
                n_h,
            )

        t_e0 = time.perf_counter()                      # measuring wall-time
        for hi, bundle in enumerate(bundles_list):
            q = e_step_postflop_bundle(bundle, prior)                                 # batched combo posterior
            last_per_hand.append(dict(q))                                             # copy so later M-step sees fixed E-step output

            if history_hook is not None and (
                hi == 0 or hi == n_h - 1 or hi % pf_stride == 0
            ):                                                                        # always log first/last; stride interior
                meta: Dict[str, Any] = {}
                if hand_meta is not None and hi < len(hand_meta):
                    meta = dict(hand_meta[hi])
                qv = list(q.values())
                ess = effective_sample_size(qv)                                       # scalar concentration per hand
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

        max_ess = max_effective_sample_size(last_per_hand) if last_per_hand else 0.0  # worst-case spread across hands
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

        nh_m = len(bundles_list)
        mb_eff = (
            "full"
            if m_batch_size <= 0 or m_batch_size >= nh_m
            else f"{min(m_batch_size, nh_m)}/{nh_m} hands/step"
        )                                                                             # minibatch label for logs
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
        )                                                                             # in-place ascent on theta_post with optional minibatches

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
    """
    Debug helper: M-step gradient for ``theta_post`` assuming uniform ``q`` over combos.
    """
    hands = [list(observations)]                                                    # wrap as one synthetic "hand" for shared gradient code
    n = len(observations)
    if n == 0:
        return np.zeros(3, dtype=float)
    w = 1.0 / n                                                                     # uniform q(c) by construction
    qmaps = [{obs.combo_key: w for obs in observations}]
    model = PostflopActionModel(PostflopPrior(floor=prior_floor), (0.0, 0.0, 0.0))  # zero tilt: pure baseline pull
    return postflop_theta_gradient(model, hands, qmaps, l2=0.0)                     # no shrinkage in debug helper
