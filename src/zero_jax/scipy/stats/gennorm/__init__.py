"""Frontend API routing for jax.scipy.stats.gennorm."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import abs as _abs
from zero_jax.numpy.lax_numpy import asarray, exp, float32, log, power, sign
from zero_jax.scipy.special import gammainc, gammaln


def logpdf(x: ArrayLikeSpec, beta: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Generalized normal log probability distribution function.

    Args:
        x: Input quantiles.
        beta: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    b_arr = asarray(beta, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    z = _abs(x_arr - loc_arr) / scale_arr
    return log(b_arr / 2.0) - log(scale_arr) - gammaln(1.0 / b_arr) - power(z, b_arr)


def pdf(x: ArrayLikeSpec, beta: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Generalized normal probability distribution function.

    Args:
        x: Input quantiles.
        beta: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    return exp(logpdf(x, beta, loc, scale))


def cdf(x: ArrayLikeSpec, beta: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Generalized normal cumulative distribution function.

    Args:
        x: Input quantiles.
        beta: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    x_arr = asarray(x, dtype=float32)
    b_arr = asarray(beta, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    z = (x_arr - loc_arr) / scale_arr
    return 0.5 + 0.5 * sign(z) * gammainc(1.0 / b_arr, power(_abs(z), b_arr))


__all__ = [
    "cdf",
    "logpdf",
    "pdf",
]
