"""Utilities for generating deterministic zero_jax arrays for fuzz testing."""

from __future__ import annotations

from collections.abc import Sequence

import zero_jax.numpy as jnp
import zero_jax.random as jrandom
from zero_jax._types import DTypeSpec, ShapeTuple
from zero_jax.numpy.lax_numpy import ndarray


def generate_random_arrays(
    key: int,
    shapes: Sequence[ShapeTuple],
    dtypes: Sequence[DTypeSpec],
) -> list[ndarray]:
    """Generate deterministic ``zero_jax.numpy.ndarray`` instances for the given shapes and dtypes.

    Args:
        key: Integer random seed.
        shapes: Sequence of array shape tuples.
        dtypes: Sequence of target dtypes.

    Returns:
        list[ndarray]: List of randomly generated ``zero_jax`` arrays.

    Raises:
        ValueError: If an unsupported dtype is requested.
    """
    prng_key = jrandom.PRNGKey(key)
    arrays: list[ndarray] = []
    for idx, (shape, dtype) in enumerate(zip(shapes, dtypes)):
        subkey = jrandom.fold_in(prng_key, idx)
        if dtype in (jnp.float32, jnp.float64, float, "float32", "float64"):
            arr = jrandom.normal(subkey, shape=shape, dtype=dtype)
            arrays.append(jnp.asarray(arr, dtype=dtype))
        elif dtype in (jnp.int32, jnp.int64, int, "int32", "int64"):
            arr = jrandom.randint(subkey, shape=shape, minval=-100, maxval=100, dtype=dtype)
            arrays.append(jnp.asarray(arr, dtype=dtype))
        elif dtype in (jnp.bool_, bool, "bool"):
            arr = jrandom.bernoulli(subkey, p=0.5, shape=shape)
            arrays.append(jnp.asarray(arr, dtype=jnp.bool_))
        else:
            raise ValueError(f"Unsupported dtype for fuzzing: {dtype}")
    return arrays
