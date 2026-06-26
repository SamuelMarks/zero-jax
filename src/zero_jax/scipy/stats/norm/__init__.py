"""Frontend API routing for jax.scipy.stats.norm."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Normal cumulative distribution function."""
    return getattr(_ops, "cdf")(*args, **kwargs)


def isf(*args: Any, **kwargs: Any) -> Any:
    """Normal distribution inverse survival function."""
    return getattr(_ops, "isf")(*args, **kwargs)


def logcdf(*args: Any, **kwargs: Any) -> Any:
    """Normal log cumulative distribution function."""
    return getattr(_ops, "logcdf")(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Normal log probability distribution function."""
    return getattr(_ops, "logpdf")(*args, **kwargs)


def logsf(*args: Any, **kwargs: Any) -> Any:
    """Normal distribution log survival function."""
    return getattr(_ops, "logsf")(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Normal probability distribution function."""
    return getattr(_ops, "pdf")(*args, **kwargs)


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Normal distribution percent point function."""
    return getattr(_ops, "ppf")(*args, **kwargs)


def sf(*args: Any, **kwargs: Any) -> Any:
    """Normal distribution survival function."""
    return getattr(_ops, "sf")(*args, **kwargs)
