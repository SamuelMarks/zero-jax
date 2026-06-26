"""Frontend API routing for jax.scipy.stats.expon."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Exponential log probability distribution function."""
    return getattr(_ops, "logpdf")(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Exponential probability distribution function."""
    return getattr(_ops, "pdf")(*args, **kwargs)
