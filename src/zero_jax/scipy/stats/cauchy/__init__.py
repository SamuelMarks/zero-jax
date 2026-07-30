"""Frontend API routing for jax.scipy.stats.cauchy."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy cumulative distribution function."""
    return _ops.cdf(*args, **kwargs)


def isf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy distribution inverse survival function."""
    return _ops.isf(*args, **kwargs)


def logcdf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy log cumulative distribution function."""
    return _ops.logcdf(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def logsf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy distribution log survival function."""
    return _ops.logsf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy probability distribution function."""
    return _ops.pdf(*args, **kwargs)


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy distribution percent point function."""
    return _ops.ppf(*args, **kwargs)


def sf(*args: Any, **kwargs: Any) -> Any:
    """Cauchy distribution log survival function."""
    return _ops.sf(*args, **kwargs)
