"""Frontend API routing for jax.scipy.stats.laplace."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Laplace cumulative distribution function."""
    return getattr(_ops, "cdf")(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Laplace log probability distribution function."""
    return getattr(_ops, "logpdf")(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Laplace probability distribution function."""
    return getattr(_ops, "pdf")(*args, **kwargs)
