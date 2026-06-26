"""Module documentation."""

from typing import Any
import ml_switcheroo_compiler.ops as ops
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def fft(a: Any, n: Any = None, axis: int = -1, norm: Any = None) -> Any:
    """JAX API implementation for fft.

    Args:
        a: Argument a.
        n: Argument n.
        axis: Argument axis.
        norm: Argument norm.

    Returns:
        Any: The result.
    """
    return _wrap(ops.fft(_to_tensor(a)))


def rfft(a: Any, n: Any = None, axis: int = -1, norm: Any = None) -> Any:
    """JAX API implementation for rfft.

    Args:
        a: Argument a.
        n: Argument n.
        axis: Argument axis.
        norm: Argument norm.

    Returns:
        Any: The result.
    """
    return _wrap(ops.rfft(_to_tensor(a)))


__all__ = ["fft", "rfft"]
