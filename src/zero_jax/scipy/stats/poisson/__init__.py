"""Frontend API routing for jax.scipy.stats.poisson."""

from typing import Any


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Poisson cumulative distribution function."""
    raise NotImplementedError("cdf not yet implemented in zero-jax")


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Poisson log probability mass function."""
    raise NotImplementedError("logpmf not yet implemented in zero-jax")


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Poisson probability mass function."""
    raise NotImplementedError("pmf not yet implemented in zero-jax")
