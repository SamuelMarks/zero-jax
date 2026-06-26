"""Frontend API routing for jax.scipy.stats.geom."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Geometric log probability mass function."""
    return getattr(_ops, "logpmf")(*args, **kwargs)


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Geometric probability mass function."""
    return getattr(_ops, "pmf")(*args, **kwargs)
