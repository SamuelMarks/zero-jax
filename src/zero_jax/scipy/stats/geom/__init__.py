"""Frontend API routing for jax.scipy.stats.geom."""

from typing import Any


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Geometric log probability mass function."""
    raise NotImplementedError("logpmf not yet implemented in zero-jax")


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Geometric probability mass function."""
    raise NotImplementedError("pmf not yet implemented in zero-jax")
