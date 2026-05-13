"""Population baseline training (multinomial logistic regression).

Frozen weight objects live in prior.preflop or prior.postflop
"""

from utils.prior.training import (
    train_multinomial_2_class,
    train_multinomial_3_class,
)

__all__ = [
    "train_multinomial_2_class",
    "train_multinomial_3_class",
]
