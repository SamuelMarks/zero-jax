"""Frontend API routing for jax.scipy.stats.logistic."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Logistic cumulative distribution function."""
    return getattr(_ops, "cdf")(*args, **kwargs)


def isf(*args: Any, **kwargs: Any) -> Any:
    """Logistic distribution inverse survival function."""
    return getattr(_ops, "isf")(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Logistic log probability distribution function."""
    return getattr(_ops, "logpdf")(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Logistic probability distribution function."""
    return getattr(_ops, "pdf")(*args, **kwargs)


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Logistic distribution percent point function."""
    return getattr(_ops, "ppf")(*args, **kwargs)


def sf(*args: Any, **kwargs: Any) -> Any:
    """Logistic distribution survival function."""
    return getattr(_ops, "sf")(*args, **kwargs)
