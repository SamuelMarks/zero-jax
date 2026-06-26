"""Frontend API routing for jax.scipy.stats.bernoulli."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Bernoulli cumulative distribution function."""
    return getattr(_ops, "cdf")(*args, **kwargs)


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Bernoulli log probability mass function."""
    return getattr(_ops, "logpmf")(*args, **kwargs)


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Bernoulli probability mass function."""
    return getattr(_ops, "pmf")(*args, **kwargs)


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Bernoulli percent point function."""
    return getattr(_ops, "ppf")(*args, **kwargs)
