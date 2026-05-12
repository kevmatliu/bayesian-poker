"""Shared types for action models, tendencies, and inference phase."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from utils.action.postflop import PostflopActionModel, PostflopPrior
from utils.action.preflop import PreflopActionModel, PreflopPrior


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
    """Tags preflop vs postflop distribution family (baseline ``*Prior`` + ``theta``)."""

    phase: InferencePhase
    model: object

    @classmethod
    def preflop(
        cls,
        prior: Optional[PreflopPrior] = None,
        *,
        theta_pre: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> ActionPrior:
        p = prior or PreflopPrior()
        return cls(InferencePhase.PREFLOP, PreflopActionModel(p, theta_pre))

    @classmethod
    def postflop(
        cls,
        prior: Optional[PostflopPrior] = None,
        *,
        theta_post: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> ActionPrior:
        p = prior or PostflopPrior()
        return cls(InferencePhase.POSTFLOP, PostflopActionModel(p, theta_post))
