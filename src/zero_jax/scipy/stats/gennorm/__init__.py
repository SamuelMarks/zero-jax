"""Frontend API routing for jax.scipy.stats.gennorm."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import abs as _abs
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, power, sign
from zero_jax.scipy.special import gammainc, gammaln


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Generalized normal log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``beta``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        b = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        z = _abs(x - loc) / scale
        return log(b / 2.0) - log(scale) - gammaln(1.0 / b) - power(z, b)
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Generalized normal probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``beta``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)


def cdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Generalized normal cumulative distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``beta``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        b = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        z = (x - loc) / scale
        return 0.5 + 0.5 * sign(z) * gammainc(1.0 / b, power(_abs(z), b))
    return _ops.cdf(*args, **kwargs)
