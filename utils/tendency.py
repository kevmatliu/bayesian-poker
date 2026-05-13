"""Shared types for action models, tendencies, and inference phase."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from utils.action.postflop import PostflopActionModel, PostflopPrior
from utils.action.preflop import PreflopActionModel, PreflopPrior


class InferencePhase(Enum):
    """Which street-type model is active."""

    PREFLOP = "preflop"  # use abstract hand classes and preflop action heads
    POSTFLOP = "postflop"  # use combo/features and postflop action heads


@dataclass
class TendencyTheta:
    """Session / player deviation vectors (fold, passive, aggressive) per phase."""

    preflop: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # logits tilt (fold, passive, agg) pre
    postflop: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # same tuple shape postflop


@dataclass
class ActionPrior:
    """Tags preflop vs postflop distribution family (baseline ``*Prior`` + ``theta``)."""

    phase: InferencePhase  # which model family is wrapped
    model: object  # concrete ``*ActionModel`` implementing log-prob / sample hooks

    @classmethod
    def preflop(
        cls,
        prior: Optional[PreflopPrior] = None,                                 # population baseline; default constructed if absent
        *,
        theta_pre: Tuple[float, float, float] = (0.0, 0.0, 0.0),              # tendency offsets folded into model
    ) -> ActionPrior:
        p = prior or PreflopPrior()                                           # instantiate sensible ChIPS-style default weights
        return cls(InferencePhase.PREFLOP, PreflopActionModel(p, theta_pre))  # bundle phase + model

    @classmethod
    def postflop(
        cls,
        prior: Optional[PostflopPrior] = None,                                   # postflop baseline logits over F/C/R
        *,
        theta_post: Tuple[float, float, float] = (0.0, 0.0, 0.0),                # same three-way tilt as preflop
    ) -> ActionPrior:
        p = prior or PostflopPrior()                                             # default postflop population prior
        return cls(InferencePhase.POSTFLOP, PostflopActionModel(p, theta_post))  # tagged postflop model
