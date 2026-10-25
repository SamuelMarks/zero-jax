"""Frontend API routing for jax.scipy.stats.t."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, log1p
from zero_jax.scipy.special import gammaln


def logpdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Student's T log probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``df``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        x = asarray(args[0], dtype=float32)
        df = asarray(args[1], dtype=float32)
        loc = asarray(args[2] if len(args) > 2 else kwargs.get("loc", 0.0), dtype=float32)
        scale = asarray(args[3] if len(args) > 3 else kwargs.get("scale", 1.0), dtype=float32)
        z = (x - loc) / scale
        return gammaln((df + 1.0) * 0.5) - gammaln(df * 0.5) - 0.5 * log(df * math.pi) - log(scale) - 0.5 * (df + 1.0) * log1p((z * z) / df)
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Student's T probability distribution function.

    Args:
        *args: Input parameter ``args`` (``x``, ``df``, ``loc=0.0``, ``scale=1.0``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return exp(logpdf(*args, **kwargs))
    return _ops.pdf(*args, **kwargs)
