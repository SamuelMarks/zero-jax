"""Frontend API routing for jax.scipy.stats.bernoulli."""

from typing import Any


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Bernoulli cumulative distribution function."""
    raise NotImplementedError("cdf not yet implemented in zero-jax")


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Bernoulli log probability mass function."""
    raise NotImplementedError("logpmf not yet implemented in zero-jax")


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Bernoulli probability mass function."""
    raise NotImplementedError("pmf not yet implemented in zero-jax")


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Bernoulli percent point function."""
    raise NotImplementedError("ppf not yet implemented in zero-jax")
