"""Frontend API routing for jax.scipy.stats.uniform."""

from typing import Any


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Uniform cumulative distribution function."""
    raise NotImplementedError("cdf not yet implemented in zero-jax")


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Uniform log probability distribution function."""
    raise NotImplementedError("logpdf not yet implemented in zero-jax")


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Uniform probability distribution function."""
    raise NotImplementedError("pdf not yet implemented in zero-jax")


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Uniform distribution percent point function."""
    raise NotImplementedError("ppf not yet implemented in zero-jax")
