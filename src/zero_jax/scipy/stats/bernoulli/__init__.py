"""Frontend API routing for jax.scipy.stats.bernoulli."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


def cdf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli cumulative distribution function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.cdf(*args, **kwargs)


def logpmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli log probability mass function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli probability mass function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.pmf(*args, **kwargs)


def ppf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bernoulli percent point function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.ppf(*args, **kwargs)
