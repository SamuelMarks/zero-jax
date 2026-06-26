"""Frontend API routing for jax.scipy.stats.wrapcauchy."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Wrapped Cauchy log probability distribution function."""
    return getattr(_ops, "logpdf")(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Wrapped Cauchy probability distribution function."""
    return getattr(_ops, "pdf")(*args, **kwargs)
