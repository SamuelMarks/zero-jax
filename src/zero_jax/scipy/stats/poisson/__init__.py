"""Frontend API routing for jax.scipy.stats.poisson."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Poisson cumulative distribution function."""
    return getattr(_ops, "cdf")(*args, **kwargs)


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Poisson log probability mass function."""
    return getattr(_ops, "logpmf")(*args, **kwargs)


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Poisson probability mass function."""
    return getattr(_ops, "pmf")(*args, **kwargs)
