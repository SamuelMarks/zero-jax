"""Frontend API routing for jax.scipy.stats.uniform."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Uniform cumulative distribution function."""
    return _ops.cdf(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Uniform log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Uniform probability distribution function."""
    return _ops.pdf(*args, **kwargs)


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Uniform distribution percent point function."""
    return _ops.ppf(*args, **kwargs)
