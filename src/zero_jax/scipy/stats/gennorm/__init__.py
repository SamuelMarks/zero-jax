"""Frontend API routing for jax.scipy.stats.gennorm."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Generalized normal cumulative distribution function."""
    return _ops.cdf(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Generalized normal log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Generalized normal probability distribution function."""
    return _ops.pdf(*args, **kwargs)
