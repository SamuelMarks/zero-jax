"""Frontend API routing for jax.scipy.stats.t."""

from typing import Any


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Student's T log probability distribution function."""
    raise NotImplementedError("logpdf not yet implemented in zero-jax")


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Student's T probability distribution function."""
    raise NotImplementedError("pdf not yet implemented in zero-jax")
