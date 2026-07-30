"""Frontend API routing for jax.scipy.stats.pareto."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Pareto log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Pareto probability distribution function."""
    return _ops.pdf(*args, **kwargs)
