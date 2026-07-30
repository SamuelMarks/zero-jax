"""Frontend API routing for jax.scipy.stats.gamma."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Gamma cumulative distribution function."""
    return _ops.cdf(*args, **kwargs)


def logcdf(*args: Any, **kwargs: Any) -> Any:
    """Gamma log cumulative distribution function."""
    return _ops.logcdf(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Gamma log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def logsf(*args: Any, **kwargs: Any) -> Any:
    """Gamma log survival function."""
    return _ops.logsf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Gamma probability distribution function."""
    return _ops.pdf(*args, **kwargs)


def sf(*args: Any, **kwargs: Any) -> Any:
    """Gamma survival function."""
    return _ops.sf(*args, **kwargs)
