"""Constants and constant objects for jax.numpy."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec, SliceSpec
from zero_jax.numpy._creation import arange, asarray, meshgrid, stack
from zero_jax.numpy._manipulation import atleast_1d, atleast_2d, concatenate, hstack
from zero_jax.numpy._ndarray import ndarray

e = math.e
euler_gamma = 0.5772156649015329

newaxis = None


def _slice_to_array(s: SliceSpec) -> ndarray:
    """Convert a ``slice(start, stop, step)`` into a 1-D ``ndarray``.

    Args:
        s: Slice specification.

    Returns:
        ndarray: Generated 1-D coordinate array.
    """
    start = 0 if s.start is None else s.start
    stop = 0 if s.stop is None else s.stop
    step = 1 if s.step is None else s.step
    return arange(start, stop, step)


class _IndexExp:
    """Frontend class for ``_IndexExp``."""

    def __getitem__(self, item: ArrayLikeSpec) -> ArrayLikeSpec:
        """Return an index tuple from ``item``.

        Args:
            item: Index expression.

        Returns:
            ArrayLikeSpec: Normalized index tuple.
        """
        if not isinstance(item, tuple):
            return (item,)
        return item


class _S:
    """Frontend class for ``_S``."""

    def __getitem__(self, item: ArrayLikeSpec) -> ArrayLikeSpec:
        """Return ``item`` unchanged as a slice/index expression.

        Args:
            item: Slice or index expression.

        Returns:
            ArrayLikeSpec: Unmodified ``item``.
        """
        return item


index_exp = _IndexExp()
s_ = _S()


class _MGrid:
    """Dense multi-dimensional meshgrid indexer ``jnp.mgrid``."""

    def __getitem__(self, item: SliceSpec | tuple[SliceSpec, ...]) -> ndarray:
        """Construct a dense meshgrid from slice specifications.

        Args:
            item: Single ``slice`` or tuple of ``slice``s.

        Returns:
            ndarray: Dense coordinate grid.
        """
        if isinstance(item, slice):
            return _slice_to_array(item)
        axes = [_slice_to_array(s) for s in item]
        grids = meshgrid(*axes, sparse=False, indexing="ij")
        return stack(list(grids), axis=0)


class _OGrid:
    """Open (sparse) multi-dimensional meshgrid indexer ``jnp.ogrid``."""

    def __getitem__(self, item: SliceSpec | tuple[SliceSpec, ...]) -> ndarray | tuple[ndarray, ...]:
        """Construct an open meshgrid from slice specifications.

        Args:
            item: Single ``slice`` or tuple of ``slice``s.

        Returns:
            ndarray | tuple[ndarray, ...]: Sparse coordinate grid(s).
        """
        if isinstance(item, slice):
            return _slice_to_array(item)
        axes = [_slice_to_array(s) for s in item]
        return tuple(meshgrid(*axes, sparse=True, indexing="ij"))


mgrid = _MGrid()
ogrid = _OGrid()


class _RClass:
    """Row concatenation helper ``jnp.r_``."""

    def __getitem__(self, item: SliceSpec | tuple[ArrayLikeSpec, ...] | ArrayLikeSpec) -> ndarray:
        """Concatenate slices and arrays along the first axis.

        Args:
            item: Slice, array, or tuple of slices/arrays.

        Returns:
            ndarray: Concatenated 1-D or N-D array.
        """
        if isinstance(item, slice):
            return _slice_to_array(item)
        if isinstance(item, tuple):
            parts = [_slice_to_array(x) if isinstance(x, slice) else atleast_1d(asarray(x)) for x in item]
            return concatenate(parts, axis=0)
        return atleast_1d(asarray(item))


class _CClass:
    """Column stack helper ``jnp.c_``."""

    def __getitem__(self, item: SliceSpec | tuple[ArrayLikeSpec, ...] | ArrayLikeSpec) -> ndarray:
        """Stack slices and 1-D/2-D arrays as columns along the second axis.

        Args:
            item: Slice, array, or tuple of slices/arrays.

        Returns:
            ndarray: 2-D column-stacked array.
        """
        if isinstance(item, slice):
            arr = _slice_to_array(item)
            return arr[:, None]
        if isinstance(item, tuple):
            cols: list[ndarray] = []
            for x in item:
                a = _slice_to_array(x) if isinstance(x, slice) else asarray(x)
                cols.append(a[:, None] if a.ndim == 1 else atleast_2d(a))
            return hstack(cols)
        a = asarray(item)
        return a[:, None] if a.ndim == 1 else atleast_2d(a)


r_ = _RClass()
c_ = _CClass()
