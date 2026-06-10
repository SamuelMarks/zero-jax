import collections
from typing import Any
from zero_jax import numpy as jnp
from zero_jax.tree_util import tree_map


class GradientTransformation(
    collections.namedtuple("GradientTransformation", ["init", "update"])
):
    pass


def sgd(
    learning_rate: float,
    momentum: float = None,
    nesterov: bool = False,
    accumulator_dtype: Any = None,
) -> GradientTransformation:
    # A very basic implementation for parity
    def init_fn(params):
        return {}  # Empty state

    def update_fn(updates, state, params=None):
        return tree_map(lambda g: jnp.multiply(g, -learning_rate), updates), state

    return GradientTransformation(init_fn, update_fn)


def adam(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
) -> GradientTransformation:
    # Basic adam
    def init_fn(params):
        return {
            "count": 0,
            "mu": tree_map(jnp.zeros_like, params),
            "nu": tree_map(jnp.zeros_like, params),
        }

    def update_fn(updates, state, params=None):
        import numpy as np

        # For parity tests, we'll implement standard eagerly for now.
        count = state["count"] + 1
        mu = tree_map(
            lambda m, g: jnp.add(jnp.multiply(b1, m), jnp.multiply(1 - b1, g)),
            state["mu"],
            updates,
        )
        nu = tree_map(
            lambda v, g: jnp.add(
                jnp.multiply(b2, v), jnp.multiply(1 - b2, jnp.multiply(g, g))
            ),
            state["nu"],
            updates,
        )

        mu_hat = tree_map(lambda m: jnp.divide(m, 1 - b1**count), mu)
        nu_hat = tree_map(lambda v: jnp.divide(v, 1 - b2**count), nu)

        updates_out = tree_map(
            lambda m, v: jnp.multiply(
                -learning_rate, jnp.divide(m, jnp.add(jnp.power(v, 0.5), eps))
            ),
            mu_hat,
            nu_hat,
        )
        return updates_out, {"count": count, "mu": mu, "nu": nu}

    return GradientTransformation(init_fn, update_fn)
