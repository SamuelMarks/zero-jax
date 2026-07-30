"""Frontend API routing for jax.scipy.stats.geom."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Geometric log probability mass function."""
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Geometric probability mass function."""
    return _ops.pmf(*args, **kwargs)
