"""Build ``PostflopFeatures`` from ``parse.State`` for runner / EM integration."""

from __future__ import annotations

from typing import Dict, List, Tuple

from utils.em import PostflopThetaObservation
from utils.prior.postflop import CALL, FOLD, PostflopFeatures, RAISE
from utils.strength.postflop import poker_hand_mapper

POSTFLOP_STREETS = ("flop", "turn", "river")


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


def postflop_features_from_state(
    state,
    target: str,
    street: str,
    hole_cards: str,
) -> Optional[PostflopFeatures]:
    """Compute features for *target* acting next in ``state`` with known hole cards."""
    board = state.community_cards or ""
    if len(board) < 6:  # need flop minimum 3 cards -> 6 chars
        return None
    if not hole_cards or len(hole_cards) < 4:
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

    try:
        info = poker_hand_mapper(hole_cards, board)
    except (ValueError, Exception):
        return None

    tex = info.get("board_texture") or {}
    wet = _board_wetness_from_mapper(tex)
    m = float(info.get("made", 0.5))
    d = float(info.get("draw", 0.0))

    bet_frac = (to_call / max(pot, 1e-6)) if facing_bet else 0.0
    pot_odds = (to_call / max(pot + to_call, 1e-6)) if facing_bet else 0.0

    return PostflopFeatures(
        made=m,
        draw=d,
        bet_frac_pot=bet_frac,
        pot_odds=pot_odds,
        in_position=in_pos,
        multiway=multiway,
        spr=spr,
        street=street,
        board_wetness=wet,
        facing_bet=facing_bet,
    )


def collect_postflop_observations_known_hole_cards(
    hand,
    target: str,
    hand_index: int,
) -> Optional[PostflopThetaObservation]:
    """Single combo observation for one hand if ``target`` hole cards are known."""
    hole = hand.hole_cards.get(target, "") or ""
    if len(hole) < 4:
        return None

    combo_key = f"h{hand_index}|{hole}"
    decisions: List[Tuple[PostflopFeatures, int]] = []

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
            feat = postflop_features_from_state(st, target, street, hole)
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


def collect_session_postflop_hands_by_pair(
    hands: List,
    observers: List[str],
    targets: List[str],
) -> Dict[Tuple[str, str], List[List[PostflopThetaObservation]]]:
    """For EM: one inner list per hand (single combo); unknown holes skipped."""
    out: Dict[Tuple[str, str], List[List[PostflopThetaObservation]]] = {}
    for observer in observers:
        for target in targets:
            out[(observer, target)] = []

    for hi, hand in enumerate(hands):
        for observer in observers:
            for target in targets:
                obs = collect_postflop_observations_known_hole_cards(hand, target, hi)
                if obs is not None:
                    out[(observer, target)].append([obs])

    return out
