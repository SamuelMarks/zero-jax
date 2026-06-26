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


def det(a: Any) -> Any:
    """JAX API implementation for det.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    return _wrap(ops.linalg.det(_to_tensor(a)))


def eigh(a: Any, UPLO: str = "L", symmetrize_input: bool = True) -> Any:
    """JAX API implementation for eigh.

    Args:
        a: Argument a.
        UPLO: Argument UPLO.
        symmetrize_input: Argument symmetrize_input.

    Returns:
        Any: The result.
    """
    res = ops.linalg.eigh(_to_tensor(a))
    return tuple(_wrap(t) for t in res)


def eigvalsh(a: Any, UPLO: str = "L") -> Any:
    """JAX API implementation for eigvalsh.

    Args:
        a: Argument a.
        UPLO: Argument UPLO.

    Returns:
        Any: The result.
    """
    return _wrap(ops.linalg.eigvalsh(_to_tensor(a)))


def inv(a: Any) -> Any:
    """JAX API implementation for inv.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    return _wrap(ops.linalg.inv(_to_tensor(a)))


def matrix_power(a: Any, n: int) -> Any:
    """JAX API implementation for matrix_power.

    Args:
        a: Argument a.
        n: Argument n.

    Returns:
        Any: The result.
    """
    return _wrap(ops.linalg.matrix_power(_to_tensor(a), n))


def pinv(a: Any, rcond: Any = None, hermitian: bool = False) -> Any:
    """JAX API implementation for pinv.

    Args:
        a: Argument a.
        rcond: Argument rcond.
        hermitian: Argument hermitian.

    Returns:
        Any: The result.
    """
    return _wrap(ops.linalg.pinv(_to_tensor(a)))


def qr(a: Any, mode: str = "reduced") -> Any:
    """JAX API implementation for qr.

    Args:
        a: Argument a.
        mode: Argument mode.

    Returns:
        Any: The result.
    """
    res = ops.linalg.qr(_to_tensor(a))
    return tuple(_wrap(t) for t in res)


def slogdet(a: Any) -> Any:
    """JAX API implementation for slogdet.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    res = ops.linalg.slogdet(_to_tensor(a))
    return tuple(_wrap(t) for t in res)


def solve(a: Any, b: Any) -> Any:
    """JAX API implementation for solve.

    Args:
        a: Argument a.
        b: Argument b.

    Returns:
        Any: The result.
    """
    return _wrap(ops.linalg.solve(_to_tensor(a), _to_tensor(b)))


def svd(
    a: Any, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False
) -> Any:
    """JAX API implementation for svd.

    Args:
        a: Argument a.
        full_matrices: Argument full_matrices.
        compute_uv: Argument compute_uv.
        hermitian: Argument hermitian.

    Returns:
        Any: The result.
    """
    res = ops.linalg.svd(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)  # pragma: no cover


__all__ = [
    "cholesky",
    "det",
    "eigh",
    "eigvalsh",
    "inv",
    "matrix_power",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "svd",
]
