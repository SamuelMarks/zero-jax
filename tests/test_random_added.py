import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import zero_jax.random as jrandom
import zero_jax.numpy as jnp


def test_random_added():
    k = jrandom.key(0)

    assert jrandom.ball(k, 2).shape is not None
    assert jrandom.beta(k, 1.0, 1.0).shape is not None
    assert jrandom.binomial(k, 10, 0.5).shape is not None
    assert jrandom.bits(k).shape is not None
    assert jrandom.cauchy(k).shape is not None
    assert jrandom.chisquare(k, 1.0).shape is not None
    assert jrandom.clone(k).shape is not None
    assert jrandom.dirichlet(k, np.array([0.5, 0.5])).shape is not None
    assert jrandom.double_sided_maxwell(k, 0.0, 1.0).shape is not None
    assert jrandom.exponential(k).shape is not None
    assert jrandom.f(k, 1.0, 1.0).shape is not None
    assert jrandom.gamma(k, 1.0).shape is not None
    assert jrandom.generalized_normal(k, 1.0).shape is not None
    assert jrandom.geometric(k, 0.5).shape is not None
    assert jrandom.gumbel(k).shape is not None
    assert jrandom.key_data(k).shape is not None
    assert jrandom.key_impl(k).shape is not None
    assert jrandom.laplace(k).shape is not None
    assert jrandom.loggamma(k, 1.0).shape is not None
    assert jrandom.logistic(k).shape is not None
    assert jrandom.lognormal(k).shape is not None
    assert jrandom.maxwell(k).shape is not None
    assert (
        jrandom.multivariate_normal(k, np.array([0.0, 0.0]), np.eye(2)).shape
        is not None
    )
    assert jrandom.orthogonal(k, 2).shape is not None
    assert jrandom.pareto(k, 1.0).shape is not None
    assert jrandom.poisson(k, 1.0).shape is not None
    assert jrandom.rademacher(k).shape is not None
    assert jrandom.rayleigh(k, 1.0).shape is not None
    assert jrandom.t(k, 1.0).shape is not None
    assert jrandom.triangular(k, -1.0, 0.0, 1.0).shape is not None
    assert jrandom.wald(k, 1.0, 1.0).shape is not None
    assert jrandom.weibull_min(k, 1.0, 1.0).shape is not None
    assert jrandom.wrap_key_data(k).shape is not None
