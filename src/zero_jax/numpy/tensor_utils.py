"""Module documentation."""

from typing import Any

import ml_switcheroo_compiler as compiler
import ml_switcheroo_compiler.ops as compiler_ops
from ml_switcheroo_compiler.core.dtype import DType


def to_array(x):
    """JAX API implementation for to_array.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return compiler_ops.array(x)
