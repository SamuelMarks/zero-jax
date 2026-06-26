"""Frontend API routing for jax.scipy.stats.laplace."""

from typing import Any


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Laplace cumulative distribution function."""
    raise NotImplementedError("cdf not yet implemented in zero-jax")


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Laplace log probability distribution function."""
    raise NotImplementedError("logpdf not yet implemented in zero-jax")


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Laplace probability distribution function."""
    raise NotImplementedError("pdf not yet implemented in zero-jax")
