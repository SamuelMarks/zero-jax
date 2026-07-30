"""Frontend API routing for jax.scipy.stats.poisson."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Poisson cumulative distribution function."""
    return _ops.cdf(*args, **kwargs)


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Poisson log probability mass function."""
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Poisson probability mass function."""
    return _ops.pmf(*args, **kwargs)
