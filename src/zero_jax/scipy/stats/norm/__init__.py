"""Frontend API routing for jax.scipy.stats.norm."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Normal cumulative distribution function."""
    return _ops.cdf(*args, **kwargs)


def isf(*args: Any, **kwargs: Any) -> Any:
    """Normal distribution inverse survival function."""
    return _ops.isf(*args, **kwargs)


def logcdf(*args: Any, **kwargs: Any) -> Any:
    """Normal log cumulative distribution function."""
    return _ops.logcdf(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Normal log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def logsf(*args: Any, **kwargs: Any) -> Any:
    """Normal distribution log survival function."""
    return _ops.logsf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Normal probability distribution function."""
    return _ops.pdf(*args, **kwargs)


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Normal distribution percent point function."""
    return _ops.ppf(*args, **kwargs)


def sf(*args: Any, **kwargs: Any) -> Any:
    """Normal distribution survival function."""
    return _ops.sf(*args, **kwargs)
