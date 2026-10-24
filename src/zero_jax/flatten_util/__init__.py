"""Flatten util submodule for zero_jax."""

from __future__ import annotations

import math
from collections.abc import Callable

import zero_jax.numpy as jnp
from zero_jax._types import ArrayProtocol, PyTree
from zero_jax.numpy.lax_numpy import ndarray
from zero_jax.tree_util import tree_flatten, tree_unflatten


def ravel_pytree(pytree: PyTree) -> tuple[ndarray, Callable[[ndarray], PyTree]]:
    """Flatten a PyTree of arrays into a single 1D array.

    Args:
        pytree: A PyTree of arrays and/or scalars to flatten.

    Returns:
        tuple[ndarray, Callable[[ndarray], PyTree]]: A tuple ``(flat_array, unflatten_fn)``.
    """
    leaves, treedef = tree_flatten(pytree)
    if not leaves:
        flat = jnp.array([])

        def unflatten_fn_empty(flat_array: ndarray) -> PyTree:
            """Reconstruct an empty PyTree from ``flat_array``.

            Args:
                flat_array: Empty 1D array.

            Returns:
                PyTree: Reconstructed empty PyTree.
            """
            return tree_unflatten(treedef, [])

        return flat, unflatten_fn_empty

    shapes = [jnp.shape(leaf) for leaf in leaves]
    sizes = [math.prod(s) for s in shapes]

    dtypes = [leaf.dtype if isinstance(leaf, ArrayProtocol) else jnp.array(leaf).dtype for leaf in leaves]

    flat_leaves = [jnp.ravel(leaf) for leaf in leaves]
    flat = jnp.concatenate(flat_leaves)

    def unflatten_fn(flat_array: ndarray) -> PyTree:
        """Reconstruct the original PyTree structure from a 1D ``flat_array``.

        Args:
            flat_array: 1D array containing all flattened leaf elements.

        Returns:
            PyTree: Reconstructed PyTree matching the original structure and dtypes.
        """
        unflattened_leaves: list[ndarray] = []
        start = 0
        for shape, size, original_dtype in zip(shapes, sizes, dtypes):
            end = start + size
            leaf_flat = flat_array[start:end]
            leaf = jnp.reshape(leaf_flat, shape)
            leaf = jnp.astype(leaf, original_dtype)
            unflattened_leaves.append(leaf)
            start = end
        return tree_unflatten(treedef, unflattened_leaves)

    return flat, unflatten_fn


__all__ = ["ravel_pytree"]
