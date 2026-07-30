"""Frontend API routing for jax.scipy.stats.wrapcauchy."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Wrapped Cauchy log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Wrapped Cauchy probability distribution function."""
    return _ops.pdf(*args, **kwargs)
