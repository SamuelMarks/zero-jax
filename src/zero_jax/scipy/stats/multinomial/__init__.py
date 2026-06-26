"""Frontend API routing for jax.scipy.stats.multinomial."""

from typing import Any


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Multinomial log probability mass function."""
    raise NotImplementedError("logpmf not yet implemented in zero-jax")


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Multinomial probability mass function."""
    raise NotImplementedError("pmf not yet implemented in zero-jax")
