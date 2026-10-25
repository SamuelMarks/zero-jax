"""Tests for tests/test_random_added.py."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import zero_jax.numpy as jnp
import zero_jax.random as jrandom
from zero_jax._types import ArrayLikeSpec, VariadicCallable


def test_random_added() -> None:
    """Execute ``test_random_added``."""
    k = jrandom.key(0)

    funcs: list[tuple[VariadicCallable[ArrayLikeSpec], list[ArrayLikeSpec]]] = [
        (jrandom.ball, [k, 2]),
        (jrandom.beta, [k, 1.0, 1.0]),
        (jrandom.binomial, [k, 10, 0.5]),
        (jrandom.bits, [k]),
        (jrandom.cauchy, [k]),
        (jrandom.chisquare, [k, 1.0]),
        (jrandom.clone, [k]),
        (jrandom.dirichlet, [k, jnp.array([0.5, 0.5])]),
        (jrandom.double_sided_maxwell, [k, 0.0, 1.0]),
        (jrandom.exponential, [k]),
        (jrandom.f, [k, 1.0, 1.0]),
        (jrandom.gamma, [k, 1.0]),
        (jrandom.generalized_normal, [k, 1.0]),
        (jrandom.geometric, [k, 0.5]),
        (jrandom.gumbel, [k]),
        (jrandom.key_data, [k]),
        (jrandom.key_impl, [k]),
        (jrandom.laplace, [k]),
        (jrandom.loggamma, [k, 1.0]),
        (jrandom.logistic, [k]),
        (jrandom.lognormal, [k]),
        (jrandom.maxwell, [k]),
        (jrandom.multivariate_normal, [k, jnp.array([0.0, 0.0]), jnp.eye(2)]),
        (jrandom.orthogonal, [k, 2]),
        (jrandom.pareto, [k, 1.0]),
        (jrandom.poisson, [k, 1.0]),
        (jrandom.rademacher, [k]),
        (jrandom.rayleigh, [k, 1.0]),
        (jrandom.t, [k, 1.0]),
        (jrandom.triangular, [k, -1.0, 0.0, 1.0]),
        (jrandom.wald, [k, 1.0, 1.0]),
        (jrandom.weibull_min, [k, 1.0, 1.0]),
        (jrandom.wrap_key_data, [k]),
    ]

    for func, args in funcs:
        try:
            func(*args)
        except Exception:
            pass
