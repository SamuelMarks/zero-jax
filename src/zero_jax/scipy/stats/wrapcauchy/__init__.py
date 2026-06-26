"""Frontend API routing for jax.scipy.stats.wrapcauchy."""

from typing import Any


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Wrapped Cauchy log probability distribution function."""
    raise NotImplementedError("logpdf not yet implemented in zero-jax")


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Wrapped Cauchy probability distribution function."""
    raise NotImplementedError("pdf not yet implemented in zero-jax")
