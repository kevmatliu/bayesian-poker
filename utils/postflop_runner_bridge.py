"""Build ``PostflopFeatures`` from ``parse.State`` for runner / EM integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# (made, draw, rich ndarray, equity) — reused across actions on the same board + hole.
PostflopStrengthBundle = Tuple[float, float, np.ndarray, float]

from utils.em import PostflopEMHandBundle, PostflopEMTimestep, PostflopThetaObservation
from utils.filter.helpers import initial_class_prior, normalize
from utils.filter.postflop import ComboRangeFilter, combo_key, parse_combo_key
from utils.prior.postflop import CALL, FOLD, PostflopFeatures, RAISE
from utils.strength.common import Card, parse_cards
from utils.strength.fast_eval import (
    card_to_index,
    combo_key_from_indices,
    made_percentile_at_combo_key,
    made_percentile_by_combo_key,
    rollout_equity_at_combo_key,
    rollout_equity_by_combo_key,
)
from utils.strength.postflop import (
    board_texture,
    draw_strength_from_hand,
    hand_feature_vector,
)

POSTFLOP_STREETS = ("flop", "turn", "river")


def postflop_target_decisions_for_hand(hand, target: str) -> List[Tuple[str, int, Tuple]]:
    """(street, action_index, raw_action_tuple) for each postflop action by ``target``."""
    decisions: List[Tuple[str, int, Tuple]] = []
    for street in POSTFLOP_STREETS:
        actions = hand.actions.get(street, {})
        for action_index in sorted(actions):
            actor = actions[action_index][0]
            if actor != target:
                continue
            decisions.append((street, action_index, actions[action_index]))
    return decisions


def raw_action_bucket_to_postflop(bucket: int) -> int:
    """Collapse preflop-style buckets (0–4) to FOLD/CALL/RAISE."""
    if bucket == 0:
        return FOLD
    if bucket == 1:
        return CALL
    return RAISE


def _street_commitments(betting_history: List) -> Dict[str, float]:
    committed: Dict[str, float] = {}
    for player, _, amt in betting_history:
        committed[player] = float(amt)
    return committed


def _board_wetness_from_mapper(texture: dict) -> float:
    score = 0.0
    if texture.get("monotone"):
        score += 0.35
    if texture.get("two_tone"):
        score += 0.15
    if texture.get("paired"):
        score += 0.15
    if texture.get("connected"):
        score += 0.15
    if texture.get("very_connected"):
        score += 0.1
    return min(1.0, score)


def _acting_order_active(player_order: List[str], alive: List[Tuple[str, bool]]) -> List[str]:
    alive_set = {p for p, ok in alive if ok}
    return [p for p in player_order if p in alive_set]


def _in_position_last_actor(player_order: List[str], alive: List[Tuple[str, bool]], target: str) -> bool:
    act = _acting_order_active(player_order, alive)
    return bool(act) and act[-1] == target


@dataclass(frozen=True)
class StateContext:
    """State-dependent features that are constant across hole-card combos."""

    bet_frac_pot: float
    pot_odds: float
    in_position: bool
    multiway: bool
    spr: float
    street: str
    board_wetness: float
    facing_bet: bool


def state_context_from_state(state, target: str, street: str) -> Optional[StateContext]:
    """Compute the combo-independent context for ``target`` acting next."""
    board = state.community_cards or ""
    if len(board) < 6:  # need at least the 3-card flop
        return None

    hist = state.betting_history or []
    committed = _street_commitments(hist)
    max_c = max(committed.values()) if committed else 0.0
    tgt_c = committed.get(target, 0.0)
    to_call = max(0.0, max_c - tgt_c)
    facing_bet = to_call > 1e-6

    pot = float(state.pot_size)
    stacks = state.current_stacks or {}
    stack_t = float(stacks.get(target, 0.0))
    spr = stack_t / max(pot, 1e-6)

    alive = state.players_in_hand or []
    n_active = sum(1 for _, ok in alive if ok)
    multiway = n_active > 2

    in_pos = _in_position_last_actor(list(state.player_order), alive, target)

    # Board texture only depends on community cards; avoid full ``poker_hand_mapper``
    # (best-hand + percentiles) which was dominating bundle collection.
    try:
        b_cards = parse_cards([board[i : i + 2] for i in range(0, len(board), 2)])
        if len(b_cards) < 3:
            return None
        tex = board_texture(b_cards)
    except (ValueError, Exception):
        tex = {}
    wet = _board_wetness_from_mapper(tex)

    bet_frac = (to_call / max(pot, 1e-6)) if facing_bet else 0.0
    pot_odds = (to_call / max(pot + to_call, 1e-6)) if facing_bet else 0.0

    return StateContext(
        bet_frac_pot=bet_frac,
        pot_odds=pot_odds,
        in_position=in_pos,
        multiway=multiway,
        spr=spr,
        street=street,
        board_wetness=wet,
        facing_bet=facing_bet,
    )


def features_from_context(
    context: StateContext,
    made: float,
    draw: float,
    *,
    rich: Optional[np.ndarray] = None,
    equity: float = -1.0,
) -> PostflopFeatures:
    """Combine state context with a combo's made/draw scores.

    ``rich`` (Method A) carries the board-relative categorical
    indicators produced by :func:`utils.strength.postflop.hand_feature_vector`,
    and ``equity`` (Method E) is the expected made percentile over future
    runouts. Both default to neutral values so legacy callers that only
    know ``(made, draw)`` still produce a valid feature row.
    """
    return PostflopFeatures(
        made=float(made),
        draw=float(draw),
        bet_frac_pot=context.bet_frac_pot,
        pot_odds=context.pot_odds,
        in_position=context.in_position,
        multiway=context.multiway,
        spr=context.spr,
        street=context.street,
        board_wetness=context.board_wetness,
        facing_bet=context.facing_bet,
        rich=rich,
        equity=float(equity),
    )


def compute_postflop_strength_bundle(
    board: str,
    hole_cards: str,
    *,
    equity_mc_samples: Optional[int] = 32,
) -> Optional[PostflopStrengthBundle]:
    """Made / draw / rich / equity for a fixed board string + hole (no pot or position).

    Used by :func:`collect_postflop_observations_known_hole_cards` so repeated
    decisions on the same board only pay for one flop rollout / percentile pass.
    """
    if not hole_cards or len(hole_cards) < 4 or not board or len(board) < 6:
        return None
    try:
        board_cards = parse_cards([board[i : i + 2] for i in range(0, len(board), 2)])
        hole = parse_cards([hole_cards[0:2], hole_cards[2:4]])
    except (ValueError, Exception):
        return None
    if len(board_cards) < 3 or len(hole) != 2:
        return None

    board_tuple = tuple(sorted(card_to_index(c) for c in board_cards))
    combo_key = combo_key_from_indices(card_to_index(hole[0]), card_to_index(hole[1]))

    made_p = made_percentile_at_combo_key(board_tuple, combo_key)
    made = 0.5 if made_p is None else float(made_p)
    draw = float(draw_strength_from_hand(hole, board_cards))
    rich = hand_feature_vector(hole, board_cards)

    mc = equity_mc_samples if len(board_tuple) == 3 else None
    eq_p = rollout_equity_at_combo_key(board_tuple, combo_key, mc_samples=mc)
    equity = float(made if eq_p is None else eq_p)
    return (made, draw, rich, equity)


def postflop_features_from_state(
    state,
    target: str,
    street: str,
    hole_cards: str,
    *,
    equity_mc_samples: Optional[int] = 32,
    strength_bundle: Optional[PostflopStrengthBundle] = None,
) -> Optional[PostflopFeatures]:
    """Compute features for *target* acting next in ``state`` with known hole cards.

    Uses cached per-board tensors plus O(1) combo lookups — avoids rebuilding
    full percentile / equity dicts on every street action (see
    :func:`utils.strength.fast_eval.made_percentile_at_combo_key`).

    Pass ``strength_bundle`` from :func:`compute_postflop_strength_bundle` when the
    board + hole are unchanged so pot / position context can update without
    recomputing rollouts.
    """
    if not hole_cards or len(hole_cards) < 4:
        return None
    context = state_context_from_state(state, target, street)
    if context is None:
        return None

    if strength_bundle is not None:
        made, draw, rich, equity = strength_bundle
    else:
        board = state.community_cards or ""
        bundle = compute_postflop_strength_bundle(
            board, hole_cards, equity_mc_samples=equity_mc_samples
        )
        if bundle is None:
            return None
        made, draw, rich, equity = bundle

    return PostflopFeatures(
        made=made,
        draw=draw,
        bet_frac_pot=context.bet_frac_pot,
        pot_odds=context.pot_odds,
        in_position=context.in_position,
        multiway=context.multiway,
        spr=context.spr,
        street=street,
        board_wetness=context.board_wetness,
        facing_bet=context.facing_bet,
        rich=rich,
        equity=equity,
    )


# Type alias for the strength cache: maps board string to a dict of
# ``combo_key → (made, draw, rich, equity)`` rows. Cached by the runner
# across consecutive actions that share the same community cards.
ComboStrengthEntry = Tuple[float, float, np.ndarray, float]
ComboStrengthCache = Dict[str, Dict[str, ComboStrengthEntry]]


def _build_combo_strength_table(
    board: str,
    *,
    equity_mc_samples: Optional[int] = 32,
) -> Dict[str, ComboStrengthEntry]:
    """All-at-once ``(made, draw, rich, equity)`` for every live combo on ``board``.

    Uses the cached per-board percentile and rollout-equity tables
    (Methods 1 & E) so each combo costs O(1) lookups instead of an
    opponent-enumeration sweep. The ``rich`` block (Method A) is
    recomputed locally from the parsed hole cards (~1 µs per combo).
    """
    board_cards = parse_cards([board[i : i + 2] for i in range(0, len(board), 2)])
    if len(board_cards) < 3:
        return {}
    board_idx = tuple(card_to_index(c) for c in board_cards)
    made_table = made_percentile_by_combo_key(board_idx)
    if len(board_idx) >= 5:
        equity_table = made_table
    else:
        equity_table = rollout_equity_by_combo_key(
            board_idx,
            mc_samples=equity_mc_samples if len(board_idx) == 3 else None,
        )

    out: Dict[str, ComboStrengthEntry] = {}
    for key, made in made_table.items():
        ca, cb = parse_combo_key(key)
        hole = [ca, cb]
        draw = draw_strength_from_hand(hole, board_cards)
        rich = hand_feature_vector(hole, board_cards)
        equity = float(equity_table.get(key, made))
        out[key] = (float(made), float(draw), rich, equity)
    return out


def precompute_combo_strengths(
    combos: Iterable[str],
    board: str,
    *,
    cache: Optional[ComboStrengthCache] = None,
    equity_mc_samples: Optional[int] = 32,
) -> Dict[str, ComboStrengthEntry]:
    """Compute per-combo ``(made, draw, rich, equity)`` for ``combos`` on ``board``.

    The all-1326 board table is built once (vectorized) and the subset
    matching ``combos`` is sliced out — orders of magnitude faster than
    looping :func:`poker_hand_mapper` per combo. The optional ``cache``
    is keyed by board string and stores the full per-board table so
    successive action updates on the same street are free.
    """
    if cache is not None and board in cache:
        full = cache[board]
    else:
        full = _build_combo_strength_table(
            board, equity_mc_samples=equity_mc_samples
        )
        if cache is not None:
            cache[board] = full

    out: Dict[str, ComboStrengthEntry] = {}
    for combo in combos:
        entry = full.get(combo)
        if entry is None:
            continue
        out[combo] = entry
    return out


def combo_features_for_state(
    state,
    target: str,
    street: str,
    combos: Iterable[str],
    *,
    strength_cache: Optional[ComboStrengthCache] = None,
    equity_mc_samples: Optional[int] = 32,
) -> Tuple[Optional[StateContext], Dict[str, PostflopFeatures]]:
    """Return ``(context, {combo_key: PostflopFeatures})`` for an action point.

    Combos blocked by the board (or otherwise un-evaluable) are silently
    dropped from the output map. ``strength_cache`` is keyed by the
    board string and is only valid for as long as the community cards
    don't change — clear it on the turn / river deal.
    """
    context = state_context_from_state(state, target, street)
    if context is None:
        return None, {}

    board = state.community_cards or ""
    board_cards = parse_cards([board[i : i + 2] for i in range(0, len(board), 2)])
    board_set = set(board_cards)

    eligible: List[str] = []
    for combo in combos:
        try:
            ca, cb = parse_combo_key(combo)
        except ValueError:
            continue
        if ca in board_set or cb in board_set:
            continue
        eligible.append(combo)

    strengths = precompute_combo_strengths(
        eligible, board, cache=strength_cache,
        equity_mc_samples=equity_mc_samples,
    )
    feats: Dict[str, PostflopFeatures] = {}
    for combo, (made, draw, rich, equity) in strengths.items():
        feats[combo] = features_from_context(
            context, made, draw, rich=rich, equity=equity
        )
    return context, feats


def collect_postflop_em_bundle_for_hand(
    hand,
    observer: str,
    target: str,
    hand_index: int,
    *,
    target_actions: Optional[List[Tuple[str, int, Tuple]]] = None,
) -> Optional[PostflopEMHandBundle]:
    """EM bundle: same prior construction as preflop EM (initial_class_prior → 1,326 combos).

    Uses observer dead cards + board progression for support only (no Bayesian filtering on
    actions). Target hole cards are not required.

    Pass ``target_actions`` when the caller already computed
    :func:`postflop_target_decisions_for_hand` for this (hand, target) — e.g. once per hand
    instead of once per observer.
    """
    if observer == target:
        return None
    if target_actions is None:
        target_actions = postflop_target_decisions_for_hand(hand, target)
    if not target_actions:
        return None

    observer_hole = hand.hole_cards.get(observer, "") or ""
    initial_169 = normalize(initial_class_prior(dead_cards=observer_hole))

    timesteps: List[PostflopEMTimestep] = []
    strength_cache: ComboStrengthCache = {}
    last_board: Optional[str] = None
    combos: Dict[str, float] = {}
    initialized = False
    initial_snapshot: Optional[Dict[str, float]] = None

    for street, action_index, raw_action in target_actions:
        states = hand.states.get(street, [])
        if action_index >= len(states):
            continue
        state = states[action_index]
        board = state.community_cards or ""
        if len(board) < 6:
            continue

        if not initialized or board != last_board:
            if not initialized:
                try:
                    combos = ComboRangeFilter.explode_preflop_to_combos(
                        initial_169,
                        observer_hole,
                        board,
                    )
                except ValueError:
                    return None
                initial_snapshot = dict(combos)
                initialized = True
            else:
                try:
                    combos = ComboRangeFilter.narrow_combo_distribution(
                        combos,
                        observer_hole_cards=observer_hole,
                        board_cards=board,
                    )
                except ValueError:
                    return None
            last_board = board
            strength_cache.clear()

        _, feats = combo_features_for_state(
            state,
            target,
            street,
            combos.keys(),
            strength_cache=strength_cache,
        )
        if not feats:
            continue

        raw_bucket = int(raw_action[1][0])
        post_action = raw_action_bucket_to_postflop(raw_bucket)
        sparse_pairs = tuple((c, feats[c]) for c in combos if combos[c] > 0.0 and c in feats)
        if not sparse_pairs:
            continue
        timesteps.append(PostflopEMTimestep(action=post_action, features_by_combo=sparse_pairs))

    if not timesteps or initial_snapshot is None:
        return None
    return PostflopEMHandBundle(
        decisions=tuple(timesteps),
        initial_combo_range=normalize(dict(initial_snapshot)),
    )


def collect_postflop_observations_known_hole_cards(
    hand,
    target: str,
    hand_index: int,
    *,
    equity_mc_samples: int = 8,
) -> Optional[PostflopThetaObservation]:
    """Single combo observation for one hand if ``target`` hole cards are known.

    ``equity_mc_samples``: flop rollout Monte Carlo size when building Method E
    equity (default **8** for bulk training speed; use **32** for closer match to
    interactive runner defaults).
    """
    hole = hand.hole_cards.get(target, "") or ""
    if len(hole) < 4:
        return None

    combo_key = f"h{hand_index}|{hole}"
    decisions: List[Tuple[PostflopFeatures, int]] = []
    # Same (community_cards, hole) often repeats across many postflop actions.
    strength_by_board_hole: Dict[Tuple[str, str], PostflopStrengthBundle] = {}

    for street in POSTFLOP_STREETS:
        acts = hand.actions.get(street, {})
        sts = hand.states.get(street, [])
        for ai in sorted(acts):
            actor, _, _amt = acts[ai]
            if actor != target:
                continue
            if ai >= len(sts):
                continue
            st = sts[ai]
            bucket = acts[ai][1][0]
            board_str = st.community_cards or ""
            if len(board_str) < 6:
                continue
            bh_key = (board_str, hole)
            if bh_key not in strength_by_board_hole:
                b = compute_postflop_strength_bundle(
                    board_str, hole, equity_mc_samples=equity_mc_samples
                )
                if b is None:
                    continue
                strength_by_board_hole[bh_key] = b
            feat = postflop_features_from_state(
                st, target, street, hole, strength_bundle=strength_by_board_hole[bh_key]
            )
            if feat is None:
                continue
            a = raw_action_bucket_to_postflop(bucket)
            decisions.append((feat, a))

    if not decisions:
        return None

    return PostflopThetaObservation(
        combo_key=combo_key,
        log_prior_range=0.0,
        decisions=tuple(decisions),
    )


def collect_session_postflop_bundles_by_pair(
    hands: List,
    observers: List[str],
    targets: List[str],
) -> Dict[Tuple[str, str], List[PostflopEMHandBundle]]:
    """For EM: one :class:`PostflopEMHandBundle` per hand with target postflop actions."""
    out: Dict[Tuple[str, str], List[PostflopEMHandBundle]] = {}
    for observer in observers:
        for target in targets:
            out[(observer, target)] = []

    for hi, hand in enumerate(hands):
        ta_by_target = {t: postflop_target_decisions_for_hand(hand, t) for t in targets}
        for observer in observers:
            for target in targets:
                if observer == target:
                    continue
                bundle = collect_postflop_em_bundle_for_hand(
                    hand,
                    observer,
                    target,
                    hi,
                    target_actions=ta_by_target.get(target),
                )
                if bundle is not None:
                    out[(observer, target)].append(bundle)

    return out
