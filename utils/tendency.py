"""Shared types for action priors, tendencies, and inference phase."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from utils.prior.postflop import PostflopPrior
from utils.prior.preflop import PreflopPrior


class InferencePhase(Enum):
    """Which street-type model is active."""

    PREFLOP = "preflop"
    POSTFLOP = "postflop"


@dataclass
class TendencyTheta:
    """Session / player deviation vectors (fold, passive, aggressive) per phase."""

    preflop: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    postflop: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ActionPrior:
    """Tags preflop vs postflop distribution family (multinomial-logit baselines + tilt)."""

    phase: InferencePhase
    model: object

    @classmethod
    def preflop(cls, prior: Optional[PreflopPrior] = None) -> ActionPrior:
        return cls(InferencePhase.PREFLOP, prior or PreflopPrior())

    @classmethod
    def postflop(cls, prior: Optional[PostflopPrior] = None) -> ActionPrior:
        return cls(InferencePhase.POSTFLOP, prior or PostflopPrior())
