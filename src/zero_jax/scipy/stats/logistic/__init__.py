"""Frontend API routing for jax.scipy.stats.logistic."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def cdf(*args: Any, **kwargs: Any) -> Any:
    """Logistic cumulative distribution function."""
    return _ops.cdf(*args, **kwargs)


def isf(*args: Any, **kwargs: Any) -> Any:
    """Logistic distribution inverse survival function."""
    return _ops.isf(*args, **kwargs)


def logpdf(*args: Any, **kwargs: Any) -> Any:
    """Logistic log probability distribution function."""
    return _ops.logpdf(*args, **kwargs)


def pdf(*args: Any, **kwargs: Any) -> Any:
    """Logistic probability distribution function."""
    return _ops.pdf(*args, **kwargs)


def ppf(*args: Any, **kwargs: Any) -> Any:
    """Logistic distribution percent point function."""
    return _ops.ppf(*args, **kwargs)


def sf(*args: Any, **kwargs: Any) -> Any:
    """Logistic distribution survival function."""
    return _ops.sf(*args, **kwargs)
