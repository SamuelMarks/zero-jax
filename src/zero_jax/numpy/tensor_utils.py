"""Tensor conversion utilities for zero_jax.numpy."""

from __future__ import annotations

import ml_switcheroo_compiler.ops as compiler_ops
from ml_switcheroo_compiler.core.tensor import Tensor

from zero_jax._types import ArrayLikeSpec


def to_array(x: ArrayLikeSpec) -> Tensor:
    """Convert an array-like input to an ``ml-switcheroo-compiler`` ``Tensor``.

    Args:
        x: Input array-like value.

    Returns:
        Tensor: The converted compiler tensor.
    """
    return compiler_ops.array(x)


__all__ = ["to_array"]
