"""Module documentation."""

from typing import Any
import ml_switcheroo_compiler.ops as ops
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def cholesky(a: Any) -> Any:
    """JAX API implementation for cholesky.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    return _wrap(ops.linalg.cholesky(_to_tensor(a)))


def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True) -> Any:
    """JAX API implementation for svd.

    Args:
        a: Argument a.
        full_matrices: Argument full_matrices.
        compute_uv: Argument compute_uv.

    Returns:
        Any: The result.
    """
    res = ops.linalg.svd(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)  # pragma: no cover


__all__ = ["cholesky", "svd"]
