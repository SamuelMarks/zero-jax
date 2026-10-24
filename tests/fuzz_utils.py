"""Utilities for generating random NumPy arrays for fuzz testing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from zero_jax._types import DTypeSpec, ShapeTuple


def generate_random_arrays(
    key: int,
    shapes: Sequence[ShapeTuple],
    dtypes: Sequence[DTypeSpec],
) -> list[np.ndarray]:
    """Generate random NumPy arrays for the given shapes and dtypes.

    Args:
        key: Integer random seed.
        shapes: Sequence of array shape tuples.
        dtypes: Sequence of target NumPy dtypes.

    Returns:
        list[np.ndarray]: List of randomly generated NumPy arrays.

    Raises:
        ValueError: If an unsupported dtype is requested.
    """
    np.random.seed(key)
    arrays: list[np.ndarray] = []
    for shape, dtype in zip(shapes, dtypes):
        if dtype in (np.float32, np.float64):
            arrays.append(np.random.randn(*shape).astype(dtype))
        elif dtype in (np.int32, np.int64):
            arrays.append(np.random.randint(-100, 100, size=shape).astype(dtype))
        elif dtype is bool:
            arrays.append(np.random.choice([True, False], size=shape))
        else:
            raise ValueError(f"Unsupported dtype for fuzzing: {dtype}")
    return arrays
