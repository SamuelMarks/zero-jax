"""Polynomial operations for jax.numpy."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy._ndarray import _to_tensor, _wrap


def poly(seq_of_zeros: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the coefficients of a polynomial.

    Args:
        seq_of_zeros: Input parameter ``seq_of_zeros``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.poly(_to_tensor(seq_of_zeros)))


def polyadd(a1: ArrayLikeSpec, a2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Adds two polynomials.

    Args:
        a1: Input parameter ``a1``.
        a2: Input parameter ``a2``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.polyadd(_to_tensor(a1), _to_tensor(a2)))


def polyder(p: ArrayLikeSpec, m: int = 1) -> ArrayLikeSpec:
    """Computes the derivative of the specified order of a polynomial.

    Args:
        p: Input parameter ``p``.
        m: Input parameter ``m``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.polyder(_to_tensor(p), m))


def polydiv(u: ArrayLikeSpec, v: ArrayLikeSpec) -> ArrayLikeSpec:
    """Divides one polynomial by another.

    Args:
        u: Input parameter ``u``.
        v: Input parameter ``v``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.polydiv(_to_tensor(u), _to_tensor(v)))


def polyfit(
    x: ArrayLikeSpec,
    y: ArrayLikeSpec,
    deg: int,
    rcond: ArrayLikeSpec = None,
    full: bool = False,
    w: ArrayLikeSpec = None,
    cov: bool = False,
) -> ArrayLikeSpec:
    """Least squares polynomial fit.

    Args:
        x: Input parameter ``x``.
        y: Input parameter ``y``.
        deg: Input parameter ``deg``.
        rcond: Input parameter ``rcond``.
        full: Input parameter ``full``.
        w: Input parameter ``w``.
        cov: Input parameter ``cov``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.polyfit(_to_tensor(x), _to_tensor(y), deg, rcond=rcond, full=full, w=w, cov=cov))


def polyint(p: ArrayLikeSpec, m: int = 1, k: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Computes the antiderivative of a polynomial.

    Args:
        p: Input parameter ``p``.
        m: Input parameter ``m``.
        k: Input parameter ``k``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.polyint(_to_tensor(p), m, k=k))


def polymul(a1: ArrayLikeSpec, a2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Multiplies two polynomials.

    Args:
        a1: Input parameter ``a1``.
        a2: Input parameter ``a2``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.polymul(_to_tensor(a1), _to_tensor(a2)))


def polysub(a1: ArrayLikeSpec, a2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Subtracts one polynomial from another.

    Args:
        a1: Input parameter ``a1``.
        a2: Input parameter ``a2``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.polysub(_to_tensor(a1), _to_tensor(a2)))


def polyval(p: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Evaluates a polynomial at specific values.

    Args:
        p: Input parameter ``p``.
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.polyval(_to_tensor(p), _to_tensor(x)))


def roots(p: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the roots of a polynomial.

    Args:
        p: Input parameter ``p``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.roots(_to_tensor(p)))
