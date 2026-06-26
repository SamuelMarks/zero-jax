"""Frontend API routing for jax.scipy.stats.gennorm."""

from typing import Any


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Generalized normal cumulative distribution function."""
    raise NotImplementedError("cdf not yet implemented in zero-jax")


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Generalized normal log probability distribution function."""
    raise NotImplementedError("logpdf not yet implemented in zero-jax")


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Generalized normal probability distribution function."""
    raise NotImplementedError("pdf not yet implemented in zero-jax")
