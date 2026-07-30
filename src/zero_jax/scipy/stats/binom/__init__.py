"""Frontend API routing for jax.scipy.stats.binom."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def logpmf(*args: Any, **kwargs: Any) -> Any:
    """Binomial log probability mass function."""
    return _ops.logpmf(*args, **kwargs)


def pmf(*args: Any, **kwargs: Any) -> Any:
    """Binomial probability mass function."""
    return _ops.pmf(*args, **kwargs)
