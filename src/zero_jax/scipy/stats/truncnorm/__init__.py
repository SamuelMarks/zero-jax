"""Frontend API routing for jax.scipy.stats.truncnorm."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, maximum, where
from zero_jax.scipy.stats import norm as _norm


def logpdf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Truncated normal log probability distribution function.

    Args:
        x: Input quantiles.
        a: Lower truncation bound in standard normal units.
        b: Upper truncation bound in standard normal units.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    a_arr = asarray(a, dtype=float32)
    b_arr = asarray(b, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    z = (x_arr - loc_arr) / scale_arr
    norm_const = maximum(_norm.cdf(b_arr) - _norm.cdf(a_arr), 1e-12)
    val = _norm.logpdf(z) - log(scale_arr) - log(norm_const)
    return where((z >= a_arr) & (z <= b_arr), val, float("-inf"))


def pdf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Truncated normal probability distribution function.

    Args:
        x: Input quantiles.
        a: Lower truncation bound in standard normal units.
        b: Upper truncation bound in standard normal units.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, a, b, loc, scale))


def cdf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Truncated normal cumulative distribution function.

    Args:
        x: Input quantiles.
        a: Lower truncation bound in standard normal units.
        b: Upper truncation bound in standard normal units.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    x_arr = asarray(x, dtype=float32)
    a_arr = asarray(a, dtype=float32)
    b_arr = asarray(b, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    z = (x_arr - loc_arr) / scale_arr
    ca = _norm.cdf(a_arr)
    cb = _norm.cdf(b_arr)
    val = (_norm.cdf(z) - ca) / maximum(cb - ca, 1e-12)
    return where(z < a_arr, 0.0, where(z > b_arr, 1.0, val))


def logcdf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Truncated normal log cumulative distribution function.

    Args:
        x: Input quantiles.
        a: Lower truncation bound in standard normal units.
        b: Upper truncation bound in standard normal units.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log cumulative distribution.
    """
    return log(cdf(x, a, b, loc, scale))


def sf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Truncated normal distribution survival function.

    Args:
        x: Input quantiles.
        a: Lower truncation bound in standard normal units.
        b: Upper truncation bound in standard normal units.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated survival function.
    """
    return 1.0 - cdf(x, a, b, loc, scale)


def logsf(x: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Truncated normal distribution log survival function.

    Args:
        x: Input quantiles.
        a: Lower truncation bound in standard normal units.
        b: Upper truncation bound in standard normal units.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log survival function.
    """
    return log(sf(x, a, b, loc, scale))


__all__ = [
    "cdf",
    "logcdf",
    "logpdf",
    "logsf",
    "pdf",
    "sf",
]
