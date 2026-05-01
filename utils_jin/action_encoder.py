"""Action re-bucketing using the pre-action state.

The legacy parser already stores action buckets, but those buckets are computed
with the pot after the action has been applied.  For a model state, we usually
want pot-relative bet sizes based on the public pot before the action.  This
module recomputes action buckets from the pre-action ``State`` object and the
legacy action tuple.

Sizing convention:

* Pre-flop raises are bucketed by big-blind multiples.  An open to 2.25bb is
  strategically normal, even though it is bigger than the blind pot.
* Post-flop bets/raises are bucketed by contribution divided by pot before the
  action.
"""

from __future__ import annotations

from dataclasses import dataclass

FOLD = 0
CHECK_CALL = 1
SMALL_BET_RAISE = 2
MEDIUM_BET_RAISE = 3
LARGE_BET_RAISE = 4

ACTION_LABELS = {
    FOLD: "fold",
    CHECK_CALL: "check_call",
    SMALL_BET_RAISE: "small_bet_raise",
    MEDIUM_BET_RAISE: "medium_bet_raise",
    LARGE_BET_RAISE: "large_bet_raise",
}


@dataclass(frozen=True)
class ActionEncoding:
    """A model-facing action with enough metadata for debugging."""

    bucket: int
    label: str
    amount_to: float
    contribution: float
    pot_fraction: float | None
    legacy_bucket: int


def _blind_commitment(hand, player: str) -> float:
    """Infer pre-flop blind commitment from the legacy ``Hand`` object."""

    # The existing parser assumes p1 is the small blind and p2 is the big blind.
    if player == "p1":
        return float(getattr(hand, "_sb", 0.0))
    if player == "p2":
        return float(getattr(hand, "_bb", 0.0))
    return 0.0


def committed_before_action(state, hand, street: str, player: str) -> float:
    """Return chips already committed by ``player`` on the current street.

    The legacy parser stores betting history entries as
    ``(player, (action_bucket, action_level), amount_to)``.  The latest
    ``amount_to`` for the acting player is their current street commitment.
    Pre-flop blinds are not in betting history, so we add them explicitly.
    """

    latest = None
    for actor, _action_info, amount_to in state.betting_history:
        if actor == player:
            latest = float(amount_to)

    if latest is not None:
        return latest
    if street == "pre-flop":
        return _blind_commitment(hand, player)
    return 0.0


def encode_action(state, hand, street: str, legacy_action: tuple) -> ActionEncoding:
    """Convert a legacy action tuple to the model action bucket.

    Parameters
    ----------
    state:
        Pre-action legacy ``State`` object.
    hand:
        Legacy ``Hand`` object.  Used only to recover blind commitments.
    street:
        Street name from the parser.
    legacy_action:
        Tuple stored by ``Hand.actions[street][t]``:
        ``(player, (legacy_bucket, action_level), amount_to)``.
    """

    player, action_info, amount_to_raw = legacy_action
    legacy_bucket = int(action_info[0])
    amount_to = float(amount_to_raw)

    if legacy_bucket == FOLD:
        return ActionEncoding(
            bucket=FOLD,
            label=ACTION_LABELS[FOLD],
            amount_to=0.0,
            contribution=0.0,
            pot_fraction=None,
            legacy_bucket=legacy_bucket,
        )

    committed = committed_before_action(state, hand, street, player)
    contribution = max(0.0, amount_to - committed)

    if legacy_bucket == CHECK_CALL:
        # ``cc`` means check when contribution is zero and call otherwise.
        return ActionEncoding(
            bucket=CHECK_CALL,
            label=ACTION_LABELS[CHECK_CALL],
            amount_to=amount_to,
            contribution=contribution,
            pot_fraction=None,
            legacy_bucket=legacy_bucket,
        )

    pot_before = float(state.pot_size)
    pot_fraction = None if pot_before <= 0 else contribution / pot_before

    if street == "pre-flop":
        # Pre-flop sizing is conventionally described in big blinds.  Using
        # pot fraction makes ordinary 2.25bb opens look huge because the blind
        # pot is only 1.5bb.
        big_blind = float(getattr(hand, "_bb", 0.0))
        bb_multiple = None if big_blind <= 0 else amount_to / big_blind
        if bb_multiple is None or bb_multiple <= 3.0:
            bucket = SMALL_BET_RAISE
        elif bb_multiple <= 8.0:
            bucket = MEDIUM_BET_RAISE
        else:
            bucket = LARGE_BET_RAISE
    elif pot_fraction is None or pot_fraction < 0.5:
        bucket = SMALL_BET_RAISE
    elif pot_fraction < 1.0:
        bucket = MEDIUM_BET_RAISE
    else:
        bucket = LARGE_BET_RAISE

    return ActionEncoding(
        bucket=bucket,
        label=ACTION_LABELS[bucket],
        amount_to=amount_to,
        contribution=contribution,
        pot_fraction=pot_fraction,
        legacy_bucket=legacy_bucket,
    )
