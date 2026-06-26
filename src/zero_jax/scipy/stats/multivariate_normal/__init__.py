"""Frontend API routing for jax.scipy.stats.multivariate_normal."""

from typing import Any


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Multivariate normal log probability distribution function."""
    raise NotImplementedError("logpdf not yet implemented in zero-jax")


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Multivariate normal probability distribution function."""
    raise NotImplementedError("pdf not yet implemented in zero-jax")
