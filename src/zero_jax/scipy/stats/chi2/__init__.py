"""Frontend API routing for jax.scipy.stats.chi2."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import asarray, float32
from zero_jax.scipy.stats import gamma as _gamma


def logpdf(x: ArrayLikeSpec, df: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Chi-square log probability distribution function.

    Args:
        x: Input quantiles.
        df: Degrees of freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log probability density.
    """
    x_arr = asarray(x, dtype=float32)
    df_arr = asarray(df, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return _gamma.logpdf(x_arr, df_arr * 0.5, loc_arr, scale_arr * 2.0)


def pdf(x: ArrayLikeSpec, df: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Chi-square probability distribution function.

    Args:
        x: Input quantiles.
        df: Degrees of freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated probability density.
    """
    x_arr = asarray(x, dtype=float32)
    df_arr = asarray(df, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return _gamma.pdf(x_arr, df_arr * 0.5, loc_arr, scale_arr * 2.0)


def cdf(x: ArrayLikeSpec, df: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Chi-square cumulative distribution function.

    Args:
        x: Input quantiles.
        df: Degrees of freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated cumulative distribution.
    """
    x_arr = asarray(x, dtype=float32)
    df_arr = asarray(df, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return _gamma.cdf(x_arr, df_arr * 0.5, loc_arr, scale_arr * 2.0)


def logcdf(x: ArrayLikeSpec, df: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Chi-square log cumulative distribution function.

    Args:
        x: Input quantiles.
        df: Degrees of freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log cumulative distribution.
    """
    x_arr = asarray(x, dtype=float32)
    df_arr = asarray(df, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return _gamma.logcdf(x_arr, df_arr * 0.5, loc_arr, scale_arr * 2.0)


def sf(x: ArrayLikeSpec, df: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Chi-square survival function.

    Args:
        x: Input quantiles.
        df: Degrees of freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated survival function.
    """
    x_arr = asarray(x, dtype=float32)
    df_arr = asarray(df, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return _gamma.sf(x_arr, df_arr * 0.5, loc_arr, scale_arr * 2.0)


def logsf(x: ArrayLikeSpec, df: ArrayLikeSpec, loc: ArrayLikeSpec = 0.0, scale: ArrayLikeSpec = 1.0) -> ArrayLikeSpec:
    """Chi-square log survival function.

    Args:
        x: Input quantiles.
        df: Degrees of freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        ArrayLikeSpec: Evaluated log survival function.
    """
    x_arr = asarray(x, dtype=float32)
    df_arr = asarray(df, dtype=float32)
    loc_arr = asarray(loc, dtype=float32)
    scale_arr = asarray(scale, dtype=float32)
    return _gamma.logsf(x_arr, df_arr * 0.5, loc_arr, scale_arr * 2.0)


__all__ = [
    "cdf",
    "logcdf",
    "logpdf",
    "logsf",
    "pdf",
    "sf",
]
