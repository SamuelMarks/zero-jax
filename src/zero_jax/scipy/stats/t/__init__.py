"""Frontend API routing for jax.scipy.stats.t."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Student's T log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Student's T probability distribution function."""
    return _ops.pdf(*args, **kwargs)
