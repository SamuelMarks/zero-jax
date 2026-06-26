"""Frontend API routing for jax.scipy.stats.beta."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Beta cumulative distribution function"""
    return getattr(_ops, "cdf")(*args, **kwargs)


def logcdf(*args: Any, **kwargs: Any) -> Any:
    """Beta log cumulative distribution function."""
    return getattr(_ops, "logcdf")(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Beta log probability distribution function."""
    return getattr(_ops, "logpdf")(*args, **kwargs)


def logsf(*args: Any, **kwargs: Any) -> Any:
    """Beta distribution log survival function."""
    return getattr(_ops, "logsf")(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Beta probability distribution function."""
    return getattr(_ops, "pdf")(*args, **kwargs)


def sf(*args: Any, **kwargs: Any) -> Any:
    """Beta distribution survival function."""
    return getattr(_ops, "sf")(*args, **kwargs)
