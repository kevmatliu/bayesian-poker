"""Dataclasses and small config types shared by runner entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

PREFLOP_PRIOR_FLOOR = 0.01


@dataclass(frozen=True)
class EMPreflopRunConfig:
    enabled: bool = False
    outer_iters: int = 5
    m_steps: int = 100
    m_lr: float = 0.05
    m_l2: float = 0.25


@dataclass(frozen=True)
class EMPreflopResult:
    enabled: bool = False
    theta_pre_by_pair: Dict[str, List[float]] = field(default_factory=dict)
    outer_iterations: int = 0
    m_step_steps: int = 0
    m_learning_rate: float = 0.0
    m_l2: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class EMPostflopRunConfig:
    enabled: bool = False
    outer_iters: int = 10
    m_steps: int = 200
    m_lr: float = 0.05
    m_l2: float = 0.25
    prior_floor: float = 1e-6


@dataclass(frozen=True)
class EMPostflopResult:
    enabled: bool = False
    theta_post_by_pair: Dict[str, List[float]] = field(default_factory=dict)
    outer_iterations: int = 0
    m_step_steps: int = 0
    m_learning_rate: float = 0.0
    m_l2: float = 0.0
    hands_with_target_cards_per_pair: Dict[str, int] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class PreflopDecision:
    hand_index: int
    observer: str
    target: str
    action_index: int
    state_key: str
    action_bucket: int
    amount: int


@dataclass(frozen=True)
class PostflopDecision:
    hand_index: int
    observer: str
    target: str
    street: str
    action_index: int
    raw_action_bucket: int
    postflop_action: int
    amount: int


@dataclass
class HandFilterResult:
    hand_index: int
    observer: str
    target: str
    observer_hole_cards: str
    phi: float
    decisions: List[PreflopDecision] = field(default_factory=list)
    top_range: List[Dict[str, float]] = field(default_factory=list)
    final_range: Dict[str, float] = field(default_factory=dict)
    log_likelihood: float = 0.0
    postflop_decisions: List[PostflopDecision] = field(default_factory=list)
    top_combos: List[Dict[str, float]] = field(default_factory=list)
    final_combo_marginal: Dict[str, float] = field(default_factory=dict)
    postflop_log_likelihood: float = 0.0


@dataclass
class RunnerResult:
    session_path: str
    observers: List[str]
    targets: List[str]
    phi: float
    hand_results: List[HandFilterResult]
    em: EMPreflopResult
    postflop: EMPostflopResult
