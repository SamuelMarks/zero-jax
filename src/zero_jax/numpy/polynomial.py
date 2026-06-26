"""Polynomial operations for jax.numpy."""

from __future__ import annotations

from typing import Any

np = __import__("numpy")


def poly(seq_of_zeros: Any) -> Any:
    """Computes the coefficients of a polynomial."""
    from .lax_numpy import array

    return array(np.poly(seq_of_zeros))


def polyadd(a1: Any, a2: Any) -> Any:
    """Adds two polynomials."""
    from .lax_numpy import array

    return array(np.polyadd(a1, a2))


def polyder(p: Any, m: int = 1) -> Any:
    """Computes the derivative of the specified order of a polynomial."""
    from .lax_numpy import array

    return array(np.polyder(p, m))


def polydiv(u: Any, v: Any) -> Any:
    """Divides one polynomial by another."""
    from .lax_numpy import array

    q, r = np.polydiv(u, v)
    return array(q), array(r)


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
    from .lax_numpy import array

    res = np.polyfit(x, y, deg, rcond=rcond, full=full, w=w, cov=cov)
    if isinstance(res, tuple):
        return tuple(
            array(r) if isinstance(r, np.ndarray) else r for r in res
        )  # pragma: no cover
    return array(res)


def polyint(p: Any, m: int = 1, k: Any = None) -> Any:
    """Computes the antiderivative of a polynomial."""
    from .lax_numpy import array

    return array(np.polyint(p, m, k))


def polymul(a1: Any, a2: Any) -> Any:
    """Multiplies two polynomials."""
    from .lax_numpy import array

    return array(np.polymul(a1, a2))


def polysub(a1: Any, a2: Any) -> Any:
    """Subtracts one polynomial from another."""
    from .lax_numpy import array

    return array(np.polysub(a1, a2))


def polyval(p: Any, x: Any) -> Any:
    """Evaluates a polynomial at specific values."""
    from .lax_numpy import array

    return array(np.polyval(p, x))


def roots(p: Any) -> Any:
    """Computes the roots of a polynomial."""
    from .lax_numpy import array

    return array(np.roots(p))
