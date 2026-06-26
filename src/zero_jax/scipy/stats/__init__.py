"""Frontend API routing for jax.scipy.stats."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops
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
    return getattr(_ops, "mode")(*args, **kwargs)


from . import multinomial
from . import multivariate_normal
from . import nbinom
from . import norm
from . import pareto
from . import poisson


def rankdata(*args: Any, **kwargs: Any) -> Any:
    """Compute the rank of data along an array axis."""
    return getattr(_ops, "rankdata")(*args, **kwargs)


def sem(*args: Any, **kwargs: Any) -> Any:
    """Compute the standard error of the mean."""
    return getattr(_ops, "sem")(*args, **kwargs)


from . import t
from . import truncnorm
from . import uniform
from . import vonmises
from . import wrapcauchy
