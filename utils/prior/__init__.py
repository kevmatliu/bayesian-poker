"""Population baseline *training* (multinomial logistic regression).

Frozen weight objects live in :mod:`utils.prior.preflop` / :mod:`utils.prior.postflop`.
Tilted action distributions are :class:`utils.action.preflop.PreflopActionModel` /
:class:`utils.action.postflop.PostflopActionModel`.
"""

from utils.prior.training import (
    train_multinomial_2_class,
    train_multinomial_3_class,
)

__all__ = [
    "train_multinomial_2_class",
    "train_multinomial_3_class",
]
