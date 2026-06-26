"""Frontend API routing for jax.scipy.stats.cauchy."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy cumulative distribution function."""
    return getattr(_ops, "cdf")(*args, **kwargs)


def isf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy distribution inverse survival function."""
    return getattr(_ops, "isf")(*args, **kwargs)


def logcdf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy log cumulative distribution function."""
    return getattr(_ops, "logcdf")(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy log probability distribution function."""
    return getattr(_ops, "logpdf")(*args, **kwargs)


def logsf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy distribution log survival function."""
    return getattr(_ops, "logsf")(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy probability distribution function."""
    return getattr(_ops, "pdf")(*args, **kwargs)


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy distribution percent point function."""
    return getattr(_ops, "ppf")(*args, **kwargs)


def sf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy distribution log survival function."""
    return getattr(_ops, "sf")(*args, **kwargs)
