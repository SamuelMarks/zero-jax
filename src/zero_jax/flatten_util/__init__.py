"""Flatten util submodule for zero_jax."""

import math
from typing import Any, Callable, Tuple

import zero_jax.numpy as jnp
from zero_jax.tree_util import tree_flatten, tree_unflatten


def ravel_pytree(pytree: Any) -> Tuple[Any, Callable[[Any], Any]]:
    """Flatten a pytree of arrays into a single 1D array.

    Args:
        pytree: A pytree of arrays and/or scalars to flatten.

    Returns:
        A tuple `(flat_array, unflatten_fn)`.
        `flat_array` is a 1D array containing all the values.
        `unflatten_fn` is a function that takes a 1D array and returns a pytree
        with the same structure as the original `pytree`.
    """
    leaves, treedef = tree_flatten(pytree)  # pragma: no cover
    if not leaves:  # pragma: no cover
        flat = jnp.array([])  # pragma: no cover

        def unflatten_fn_empty(flat_array: Any) -> Any:  # pragma: no cover
            return tree_unflatten(treedef, [])  # pragma: no cover

        return flat, unflatten_fn_empty  # pragma: no cover

    # Record the original shapes, sizes, and dtypes to reconstruct exactly
    shapes = [jnp.shape(leaf) for leaf in leaves]  # pragma: no cover
    sizes = [math.prod(s) for s in shapes]  # pragma: no cover

    # Store dtypes, handling potential scalars
    dtypes = []  # pragma: no cover
    for leaf in leaves:  # pragma: no cover
        if hasattr(leaf, "dtype"):  # pragma: no cover
            dtypes.append(leaf.dtype)  # pragma: no cover
        else:
            # Scalar, convert to array to check dtype
            dtypes.append(jnp.array(leaf).dtype)  # pragma: no cover

    flat_leaves = [jnp.ravel(leaf) for leaf in leaves]  # pragma: no cover
    flat = jnp.concatenate(flat_leaves)  # pragma: no cover

    def unflatten_fn(flat_array: Any) -> Any:  # pragma: no cover
        unflattened_leaves = []  # pragma: no cover
        start = 0  # pragma: no cover
        for shape, size, original_dtype in zip(
            shapes, sizes, dtypes
        ):  # pragma: no cover
            end = start + size  # pragma: no cover
            leaf_flat = flat_array[start:end]  # pragma: no cover
            leaf = jnp.reshape(leaf_flat, shape)  # pragma: no cover
            leaf = jnp.astype(leaf, original_dtype)  # pragma: no cover
            unflattened_leaves.append(leaf)  # pragma: no cover
            start = end  # pragma: no cover
        return tree_unflatten(treedef, unflattened_leaves)  # pragma: no cover

    return flat, unflatten_fn  # pragma: no cover


__all__ = ["ravel_pytree"]
