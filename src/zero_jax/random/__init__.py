"""Pseudo-random number generation (PRNG) utilities."""

from __future__ import annotations

from typing import Any
import ml_switcheroo

from .prng import (
    split,
    fold_in,
    PRNGKey,
    uniform,
    normal,
    randint,
    bernoulli,
    categorical,
    permutation,
    choice,
)

__all__ = [
    "split",
    "fold_in",
    "PRNGKey",
    "uniform",
    "normal",
    "randint",
    "bernoulli",
    "categorical",
    "permutation",
    "choice",
]
