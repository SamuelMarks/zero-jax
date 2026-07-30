"""Frontend API routing for jax.scipy.stats."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops

from . import bernoulli, beta, betabinom, binom, cauchy, chi2, dirichlet, expon, gamma


class gaussian_kde:
    """Gaussian Kernel Density Estimator"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


from . import gennorm, geom, laplace, logistic


def mode(*args: Any, **kwargs: Any) -> Any:
    """Compute the mode (most common value) along an axis of an array."""
    return _ops.mode(*args, **kwargs)


from . import multinomial, multivariate_normal, nbinom, norm, pareto, poisson


def rankdata(*args: Any, **kwargs: Any) -> Any:
    """Compute the rank of data along an array axis."""
    return _ops.rankdata(*args, **kwargs)


def sem(*args: Any, **kwargs: Any) -> Any:
    """Compute the standard error of the mean."""
    return _ops.sem(*args, **kwargs)


from . import t, truncnorm, uniform, vonmises, wrapcauchy
