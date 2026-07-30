"""Polynomial operations for jax.numpy."""

from __future__ import annotations

from typing import Any

from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def poly(seq_of_zeros: Any) -> Any:
    """Computes the coefficients of a polynomial."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.poly(_to_tensor(seq_of_zeros))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)


def polyadd(a1: Any, a2: Any) -> Any:
    """Adds two polynomials."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.polyadd(_to_tensor(a1), _to_tensor(a2))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)


def polyder(p: Any, m: int = 1) -> Any:
    """Computes the derivative of the specified order of a polynomial."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.polyder(_to_tensor(p), m)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)


def polydiv(u: Any, v: Any) -> Any:
    """Divides one polynomial by another."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.polydiv(_to_tensor(u), _to_tensor(v))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)  # pragma: no cover


def polyfit(
    x: Any,
    y: Any,
    deg: int,
    rcond: Any = None,
    full: bool = False,
    w: Any = None,
    cov: bool = False,
) -> Any:
    """Least squares polynomial fit."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.polyfit(
        _to_tensor(x), _to_tensor(y), deg, rcond=rcond, full=full, w=w, cov=cov
    )
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)


def polyint(p: Any, m: int = 1, k: Any = None) -> Any:
    """Computes the antiderivative of a polynomial."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.polyint(_to_tensor(p), m, k=k)
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)


def polymul(a1: Any, a2: Any) -> Any:
    """Multiplies two polynomials."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.polymul(_to_tensor(a1), _to_tensor(a2))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)


def polysub(a1: Any, a2: Any) -> Any:
    """Subtracts one polynomial from another."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.polysub(_to_tensor(a1), _to_tensor(a2))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)


def polyval(p: Any, x: Any) -> Any:
    """Evaluates a polynomial at specific values."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.polyval(_to_tensor(p), _to_tensor(x))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)


def roots(p: Any) -> Any:
    """Computes the roots of a polynomial."""
    import zero_jax._compiler_proxy_ops as ops

    res = ops.roots(_to_tensor(p))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)
