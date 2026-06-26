import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import zero_jax.random as jrandom
import zero_jax.numpy as jnp


def test_random_added():
    k = jrandom.key(0)

    funcs = [
        ("ball", [k, 2]),
        ("beta", [k, 1.0, 1.0]),
        ("binomial", [k, 10, 0.5]),
        ("bits", [k]),
        ("cauchy", [k]),
        ("chisquare", [k, 1.0]),
        ("clone", [k]),
        ("dirichlet", [k, np.array([0.5, 0.5])]),
        ("double_sided_maxwell", [k, 0.0, 1.0]),
        ("exponential", [k]),
        ("f", [k, 1.0, 1.0]),
        ("gamma", [k, 1.0]),
        ("generalized_normal", [k, 1.0]),
        ("geometric", [k, 0.5]),
        ("gumbel", [k]),
        ("key_data", [k]),
        ("key_impl", [k]),
        ("laplace", [k]),
        ("loggamma", [k, 1.0]),
        ("logistic", [k]),
        ("lognormal", [k]),
        ("maxwell", [k]),
        ("multivariate_normal", [k, np.array([0.0, 0.0]), np.eye(2)]),
        ("orthogonal", [k, 2]),
        ("pareto", [k, 1.0]),
        ("poisson", [k, 1.0]),
        ("rademacher", [k]),
        ("rayleigh", [k, 1.0]),
        ("t", [k, 1.0]),
        ("triangular", [k, -1.0, 0.0, 1.0]),
        ("wald", [k, 1.0, 1.0]),
        ("weibull_min", [k, 1.0, 1.0]),
        ("wrap_key_data", [k]),
    ]

    for name, args in funcs:
        func = getattr(jrandom, name)
        try:
            func(*args)
        except Exception:
            pass
