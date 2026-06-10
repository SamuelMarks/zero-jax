"""Module docstring."""

from typing import Any
import ml_switcheroo

"Module docstring."
from .activation import (
    gelu,
    softmax,
    logsumexp,
    one_hot,
    sigmoid,
    hard_sigmoid,
    log_softmax,
    selu,
    celu,
    elu,
    silu,
    swish,
    hard_tanh,
    relu6,
    relu,
    log_sigmoid,
)
from . import initializers

__all__ = [
    "gelu",
    "softmax",
    "logsumexp",
    "one_hot",
    "initializers",
    "sigmoid",
    "log_softmax",
    "selu",
    "celu",
    "elu",
    "silu",
    "swish",
    "hard_tanh",
    "hard_sigmoid",
    "relu6",
    "relu",
    "log_sigmoid",
]
