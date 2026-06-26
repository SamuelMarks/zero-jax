"""Frontend API routing for jax.scipy.stats."""

from typing import Any
from . import bernoulli
from . import beta
from . import betabinom
from . import binom
from . import cauchy
from . import chi2
from . import dirichlet
from . import expon
from . import gamma


class gaussian_kde:
    """Gaussian Kernel Density Estimator"""

    pass


from . import gennorm
from . import geom
from . import laplace
from . import logistic


def mode(*args: Any, **kwargs: Any) -> Any:
    """Compute the mode (most common value) along an axis of an array."""
    raise NotImplementedError("mode not yet implemented in zero-jax")


from . import multinomial
from . import multivariate_normal
from . import nbinom
from . import norm
from . import pareto
from . import poisson


def rankdata(*args: Any, **kwargs: Any) -> Any:
    """Compute the rank of data along an array axis."""
    raise NotImplementedError("rankdata not yet implemented in zero-jax")


def sem(*args: Any, **kwargs: Any) -> Any:
    """Compute the standard error of the mean."""
    raise NotImplementedError("sem not yet implemented in zero-jax")


from . import t
from . import truncnorm
from . import uniform
from . import vonmises
from . import wrapcauchy
