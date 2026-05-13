"""Shared utilities for filter modules (priors over classes / combos, normalization)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from utils.strength.preflop import all_169_classes


_SUIT_CHARS = "SHDC"        # single-letter suits in deterministic iteration order


@dataclass
class FilterStep:
    state_key: str              # human-readable spot key (position, action history, etc.)
    action_bucket: int          # discrete action index observed at this update
    evidence: float             # sum of unnormalized weights (marginal likelihood)
    ess: float                  # effective sample size after update
    top_class: str              # highest-probability class / bucket
    top_prob: float             # probability mass at the mode
    layer: str                  # "preflop" | "postflop"


def _dead_card_set(dead_cards: str = "") -> set[str]:
    return {
        dead_cards[i : i + 2].upper()          # normalize case for comparisons
        for i in range(0, len(dead_cards), 2)  # step by two chars per card
        if dead_cards[i : i + 2]               # skip empty slices when string length is odd
    }


def available_combo_count(hand_class: str, dead_cards: str = "") -> int:
    dead = _dead_card_set(dead_cards)                                                               # cards removed from the deck before counting

    if len(hand_class) == 2:                                                                        # pocket pair class ``RR``
        rank = hand_class[0]                                                                        # duplicated rank char
        available_cards = [f"{rank}{suit}" for suit in _SUIT_CHARS if f"{rank}{suit}" not in dead]  # live cards of that rank
        n = len(available_cards)                                                                   
        return (n * (n - 1)) // 2                                                                   # C(n,2) unordered hole pairs

    rank1, rank2 = hand_class[0], hand_class[1]                                                     # distinct ranks for non-pair classes
    suited = hand_class.endswith("s")                                                               # ``s`` suffix vs ``o`` offsuit

    if suited:                                                                                      # both hole cards must share the same suit
        return sum(
            1                                                                                       # count one live suited combo for this suit wheel entry
            for suit in _SUIT_CHARS
            if f"{rank1}{suit}" not in dead and f"{rank2}{suit}" not in dead                        # neither blocker present
        )

    return sum(
        1                                                                                           # count distinct ordered pairs then implicitly treat as unordered in caller context
        for suit1 in _SUIT_CHARS
        for suit2 in _SUIT_CHARS
        if suit1 != suit2                                                                           # offsuit requires different suits
        and f"{rank1}{suit1}" not in dead                                                           # first card not removed
        and f"{rank2}{suit2}" not in dead                                                           # second card not removed
    )


def initial_class_prior(dead_cards: str = "") -> Dict[str, float]:
    """Uniform-over-combos prior conditioned on any known dead cards."""
    classes = all_169_classes()                                                                     # fixed ordering for vector alignment
    counts = {hand_class: available_combo_count(hand_class, dead_cards) for hand_class in classes}  # support size per class
    total = sum(counts.values())                                                                    
    if total <= 0:                                                                                  
        raise ValueError("Cannot build an initial prior with zero available combos.")
    return {hand_class: count / total for hand_class, count in counts.items()}                      # combo-uniform --> class prior


def normalize(d: Dict) -> Dict:
    total = sum(d.values())                      
    if total <= 0:                               
        raise ValueError("Cannot normalize a zero distribution.")
    return {k: v / total for k, v in d.items()}  # project onto probability simplex


def effective_sample_size(distribution: Dict) -> float:
    """ESS = 1 / sum(p_i^2)."""
    denom = sum(p * p for p in distribution.values()) 
    return 1.0 / denom if denom > 0 else 0.0 

