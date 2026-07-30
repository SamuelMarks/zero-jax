"""Frontend API routing for jax.scipy.stats.expon."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Exponential log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Exponential probability distribution function."""
    return _ops.pdf(*args, **kwargs)
