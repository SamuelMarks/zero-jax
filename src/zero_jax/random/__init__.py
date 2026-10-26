"""Frontend API routing for zero-jax.random."""

from __future__ import annotations

from .missing_funcs import (
    Key,
    ball,
    beta,
    binomial,
    bits,
    cauchy,
    chisquare,
    clone,
    dirichlet,
    double_sided_maxwell,
    exponential,
    f,
    gamma,
    generalized_normal,
    geometric,
    gumbel,
    key,
    key_data,
    key_dtype,
    key_impl,
    laplace,
    loggamma,
    logistic,
    lognormal,
    maxwell,
    multinomial,
    multivariate_normal,
    orthogonal,
    pareto,
    poisson,
    rademacher,
    rayleigh,
    t,
    triangular,
    wald,
    weibull_min,
    wrap_key_data,
)
from .prng import (
    PRNGKey,
    bernoulli,
    categorical,
    choice,
    fold_in,
    normal,
    permutation,
    randint,
    split,
    truncated_normal,
    uniform,
)


def random_gamma_p(name: str = "random_gamma") -> str:
    """Return the primitive identifier for ``random_gamma_p``.

    Args:
        name: Primitive name.

    Returns:
        str: Primitive identifier string.
    """
    return name


__all__ = [
    "Key",
    "PRNGKey",
    "ball",
    "bernoulli",
    "beta",
    "binomial",
    "bits",
    "categorical",
    "cauchy",
    "chisquare",
    "choice",
    "clone",
    "dirichlet",
    "double_sided_maxwell",
    "exponential",
    "f",
    "fold_in",
    "gamma",
    "generalized_normal",
    "geometric",
    "gumbel",
    "key",
    "key_data",
    "key_dtype",
    "key_impl",
    "laplace",
    "loggamma",
    "logistic",
    "lognormal",
    "maxwell",
    "multinomial",
    "multivariate_normal",
    "normal",
    "orthogonal",
    "pareto",
    "permutation",
    "poisson",
    "rademacher",
    "randint",
    "random_gamma_p",
    "rayleigh",
    "split",
    "t",
    "triangular",
    "truncated_normal",
    "uniform",
    "wald",
    "weibull_min",
    "wrap_key_data",
]
