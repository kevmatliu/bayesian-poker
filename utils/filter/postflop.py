"""Post-flop Bayesian filter over the 1,326 specific 2-card combos."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from utils.filter.common import FilterStep, effective_sample_size, normalize
from utils.action.postflop import PostflopActionModel, PostflopFeatures, PostflopPrior
from utils.action.preflop import StateKey
from utils.parse import Card, all_52_cards, parse_card, parse_cards
from utils.strength.preflop import all_169_classes, get_equivalence_class

CardsLike = Union[str, Iterable[Card], Iterable[str], None]


def _coerce_cards(cards: CardsLike) -> List[Card]:
    """Normalize string / iterable input into a list of :class:`Card`."""
    if cards is None:
        return []
    if isinstance(cards, str):
        s = cards.strip()
        if not s:
            return []
        if len(s) % 2 != 0:
            raise ValueError(f"Invalid cards string: {cards!r}")
        return parse_cards([s[i : i + 2] for i in range(0, len(s), 2)])

    out: List[Card] = []
    for item in cards:
        if isinstance(item, Card):
            out.append(item)
        elif isinstance(item, str):
            out.append(parse_card(item))
        else:
            raise TypeError(f"Unsupported card element: {item!r}")
    return out


def combo_key(card_a: Card, card_b: Card) -> str:
    """Canonical 4-character combo key with the higher-ranked card first.

    Ties on rank are broken by suit (C < D < H < S). Suits are emitted in
    lowercase to match the ``pokerkit`` / ``Hand.hole_cards`` convention
    (``"AhKh"``), giving a single canonical key for a 2-card holding.
    """
    a, b = sorted([card_a, card_b], key=lambda c: (-c.value, c.suit))
    return f"{a.rank}{a.suit.lower()}{b.rank}{b.suit.lower()}"


def parse_combo_key(key: str) -> Tuple[Card, Card]:
    """Inverse of :func:`combo_key`. O(1) lookup against the canonical table."""
    if len(key) != 4:
        raise ValueError(f"Combo key must be 4 chars (e.g. 'AhKh'), got {key!r}")
    cached = _COMBO_KEY_TO_CARDS.get(key) if "_COMBO_KEY_TO_CARDS" in globals() else None
    if cached is not None:
        return cached
    return parse_card(key[0:2]), parse_card(key[2:4])


def all_combo_keys() -> List[str]:
    """All 1,326 unordered 2-card combinations as canonical keys."""
    deck = all_52_cards()
    return [combo_key(a, b) for a, b in combinations(deck, 2)]


def _build_class_to_combos() -> Dict[str, List[Tuple[str, Card, Card]]]:
    """Static map class -> list of ``(combo_key, card_a, card_b)`` for all 169 classes.

    Pairs always have 6 concrete combos, suited classes have 4, offsuit classes
    have 12 — total 1,326. Building this once (at import time) lets
    :func:`ComboRangeFilter.explode_preflop_to_combos` iterate only the live
    members of each class instead of bucketing every 1,326 combo per call.
    """
    table: Dict[str, List[Tuple[str, Card, Card]]] = {
        cls: [] for cls in all_169_classes()
    }
    for ca, cb in combinations(all_52_cards(), 2):
        cls = get_equivalence_class([ca, cb])
        table[cls].append((combo_key(ca, cb), ca, cb))
    return table


_CLASS_TO_COMBOS: Dict[str, List[Tuple[str, Card, Card]]] = _build_class_to_combos()

_COMBO_KEY_TO_CARDS: Dict[str, Tuple[Card, Card]] = {
    key: (ca, cb)
    for members in _CLASS_TO_COMBOS.values()
    for key, ca, cb in members
}

_COMBO_KEY_TO_CLASS: Dict[str, str] = {
    key: cls
    for cls, members in _CLASS_TO_COMBOS.items()
    for key, _ca, _cb in members
}


class ComboRangeFilter:
    """Bayesian filter over the 1,326 concrete 2-card combos.

    ``Q_t(c) ∝ Q_{t-1}(c) * P(a_t | s_t, c)``

    where ``c`` is a specific 2-card holding and the action likelihood is
    obtained from the post-flop multinomial-logit prior conditioned on the
    combo-derived made/draw features and the current betting state.

    Likelihoods use ``prior_model``'s frozen ``beta`` matrices; ``theta_post``
    on this filter is applied via :class:`PostflopActionModel` (same tilt as
    :meth:`PostflopActionModel.action_probs_matrix_given_theta` with that
    ``theta_post``). Pass a baseline :class:`PostflopPrior` only; tendency is
    controlled through ``theta_post`` here.
    """

    def __init__(
        self,
        observer_name: str,
        target_name: str,
        observer_hole_cards: CardsLike = "",
        board_cards: CardsLike = "",
        prior_model: Optional[PostflopPrior] = None,
        *,
        theta_post: Sequence[float] | None = None,
        initial_combo_dist: Optional[Mapping[str, float]] = None,
    ):
        self.observer_name = observer_name
        self.target_name = target_name
        self.observer_cards: List[Card] = _coerce_cards(observer_hole_cards)
        self.board: List[Card] = _coerce_cards(board_cards)
        self.prior_model = prior_model or PostflopPrior()
        if theta_post is None:
            self._theta_post = (0.0, 0.0, 0.0)
        else:
            t = tuple(float(x) for x in theta_post)
            if len(t) != 3:
                raise ValueError("theta_post must have length 3 (fold, passive, aggression tilts).")
            self._theta_post = t
        self._action_model = PostflopActionModel(self.prior_model, self._theta_post)
        self.combos: Dict[str, float] = (
            normalize(dict(initial_combo_dist)) if initial_combo_dist else {}
        )
        self.steps: List[FilterStep] = []

    @staticmethod
    def explode_preflop_to_combos(
        preflop_range: Mapping[str, float],
        observer_cards: CardsLike,
        board: CardsLike,
    ) -> Dict[str, float]:
        """Bridge the 169-class pre-flop range to a 1,326-combo distribution.

        Each hand class (e.g. ``AKs``) is mapped to its concrete combos
        (e.g. ``AhKh``, ``AdKd``). Combos that conflict with the observer's
        hole cards or any visible board card receive **zero** mass. The
        remaining valid combos for that class share the class probability
        uniformly. The returned distribution is normalized to sum to 1.

        Each hand class has a fixed combo count (pair → 6, suited → 4,
        offsuit → 12). The static ``class -> combos`` table is built once at
        import time so this routine only iterates the (at most 1,326) cached
        members of the requested classes — no 52-card combinatorics per call.
        """
        observer = _coerce_cards(observer_cards)
        board_cards = _coerce_cards(board)
        dead = set(observer) | set(board_cards)

        combo_dist: Dict[str, float] = {}
        for hand_class, prob in preflop_range.items():
            mass = float(prob)
            if mass <= 0.0:
                continue
            members = _CLASS_TO_COMBOS.get(hand_class)
            if not members:
                continue
            live = [(key, ca, cb) for key, ca, cb in members if ca not in dead and cb not in dead]
            if not live:
                continue
            share = mass / len(live)
            for key, _ca, _cb in live:
                combo_dist[key] = combo_dist.get(key, 0.0) + share

        if not combo_dist:
            raise ValueError(
                "explode_preflop_to_combos: no valid combos remain after "
                f"removing observer hole cards / board (observer={observer}, "
                f"board={board_cards})."
            )
        return normalize(combo_dist)

    def explode_from_preflop(
        self,
        preflop_range: Mapping[str, float],
        board: Optional[CardsLike] = None,
        observer_hole_cards: Optional[CardsLike] = None,
    ) -> Dict[str, float]:
        """Initialise ``self.combos`` from a pre-flop class distribution.

        Optionally update the observer hole cards or board first so the
        bridge always reflects current public/private knowledge.
        """
        if observer_hole_cards is not None:
            self.observer_cards = _coerce_cards(observer_hole_cards)
        if board is not None:
            self.board = _coerce_cards(board)
        self.combos = self.explode_preflop_to_combos(
            preflop_range,
            self.observer_cards,
            self.board,
        )
        return self.combos

    def set_board(
        self,
        board: CardsLike,
        renormalize: bool = True,
    ) -> Dict[str, float]:
        """Update the board (e.g. turn / river dealt) and zero new conflicts.

        Combos that include a card now visible on the board are dropped.
        Surviving combos are re-normalised when ``renormalize`` is True.
        """
        new_board = _coerce_cards(board)
        for c in self.observer_cards:
            if c in new_board:
                raise ValueError(
                    f"set_board: observer card {c!r} appears on the board."
                )
        self.board = new_board
        if not self.combos:
            return self.combos

        dead = set(self.observer_cards) | set(self.board)
        survivors: Dict[str, float] = {}
        for combo, prob in self.combos.items():
            ca, cb = parse_combo_key(combo)
            if ca in dead or cb in dead:
                continue
            survivors[combo] = prob

        if not survivors:
            raise ValueError(
                "set_board: every combo is now blocked by observer cards or board."
            )
        self.combos = normalize(survivors) if renormalize else survivors
        return self.combos

    @staticmethod
    def narrow_combo_distribution(
        combos: Mapping[str, float],
        *,
        observer_hole_cards: CardsLike,
        board_cards: CardsLike,
        renormalize: bool = True,
    ) -> Dict[str, float]:
        """Drop combos conflicting with observer holes or board; optionally renormalize."""
        if not combos:
            return {}
        observer = _coerce_cards(observer_hole_cards)
        board_cards_list = _coerce_cards(board_cards)
        for c in observer:
            if c in board_cards_list:
                raise ValueError(
                    f"narrow_combo_distribution: observer card {c!r} appears on the board."
                )
        dead = set(observer) | set(board_cards_list)
        survivors: Dict[str, float] = {}
        for combo, prob in combos.items():
            if prob <= 0.0:
                continue
            ca, cb = parse_combo_key(combo)
            if ca in dead or cb in dead:
                continue
            survivors[combo] = prob
        if not survivors:
            raise ValueError(
                "narrow_combo_distribution: every combo is blocked by observer cards or board."
            )
        return normalize(survivors) if renormalize else survivors

    def update(
        self,
        action_bucket: int,
        feature_by_combo: Mapping[str, PostflopFeatures],
        state_key: StateKey | str = "",
    ) -> Dict[str, float]:
        """Apply ``Q_{t+1}(c) ∝ Q_t(c) * P(a_t | s_t, c)``.

        ``feature_by_combo`` maps each canonical combo key to the
        :class:`PostflopFeatures` derived from that combo together with the
        current betting state and board. Combos with no entry are zeroed
        out. Internally we batch every live combo into a single
        ``(N, PHI_DIM)`` feature matrix and let
        :meth:`PostflopPrior.action_probs_matrix_given_theta` apply the filter's
        ``theta_post`` tilt with one matmul + softmax per batch (Method D).
        """
        if not self.combos:
            raise ValueError(
                "ComboRangeFilter has no combos; call explode_from_preflop "
                "or pass an initial_combo_dist before updating."
            )

        live_combos: List[str] = []
        live_probs: List[float] = []
        live_feats: List[PostflopFeatures] = []
        for combo, prob in self.combos.items():
            if prob <= 0.0:
                continue
            feat = feature_by_combo.get(combo)
            if feat is None:
                continue
            live_combos.append(combo)
            live_probs.append(float(prob))
            live_feats.append(feat)

        if live_combos:
            from utils.action.postflop import feature_vector

            feature_matrix = np.stack([feature_vector(f) for f in live_feats], axis=0)
            facing = np.fromiter(
                (f.facing_bet for f in live_feats),
                dtype=bool,
                count=len(live_feats),
            )
            probs_matrix = self._action_model.action_probs_matrix(feature_matrix, facing)
            action_col = probs_matrix[:, int(action_bucket)]
            prior_arr = np.asarray(live_probs, dtype=float)
            unnorm_arr = prior_arr * action_col
            unnorm = {
                c: float(v)
                for c, v in zip(live_combos, unnorm_arr)
                if v > 0.0
            }
        else:
            unnorm = {}

        evidence = sum(unnorm.values())
        if evidence <= 0.0:
            raise ValueError(
                "ComboRangeFilter produced zero evidence at "
                f"action={action_bucket} (observer={self.observer_name}, "
                f"target={self.target_name})."
            )

        self.combos = {c: v / evidence for c, v in unnorm.items()}

        state_key_str = (
            state_key.as_string() if isinstance(state_key, StateKey) else str(state_key)
        )
        top_combo, top_prob = self.top_k(1)[0]
        self.steps.append(
            FilterStep(
                state_key=state_key_str,
                action_bucket=action_bucket,
                evidence=evidence,
                ess=effective_sample_size(self.combos),
                top_class=top_combo,
                top_prob=top_prob,
                layer="combo",
            )
        )
        return self.combos

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.combos.items(), key=lambda x: x[1], reverse=True)[:k]

    def class_marginal(self) -> Dict[str, float]:
        """Aggregate the combo distribution back onto the 169 hand classes."""
        out: Dict[str, float] = {cls: 0.0 for cls in all_169_classes()}
        for combo, prob in self.combos.items():
            cls = _COMBO_KEY_TO_CLASS.get(combo)
            if cls is None:
                ca, cb = parse_combo_key(combo)
                cls = get_equivalence_class([ca, cb])
            out[cls] = out.get(cls, 0.0) + prob
        return out

    def true_combo_probability(self, combo: str) -> float:
        if len(combo) != 4:
            raise ValueError(f"true_combo_probability expects a 4-char key, got {combo!r}")
        a, b = parse_combo_key(combo)
        return self.combos.get(combo_key(a, b), 0.0)

    def log_likelihood(self) -> float:
        return sum(math.log(step.evidence) for step in self.steps if step.evidence > 0)
