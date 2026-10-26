"""LAX slicing, indexing, gather, and scatter primitives."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec, ArrayProtocol
from zero_jax.lax.primitives import dynamic_slice, dynamic_update_slice, gather, scatter, scatter_add, slice
from zero_jax.numpy import (
    arange,
    array,
    asarray,
    astype,
    broadcast_to,
    flip,
    int32,
    reshape,
    take,
    zeros_like,
)
from zero_jax.numpy import (
    empty as jnp_empty,
)
from zero_jax.numpy import (
    split as jnp_split,
)
from zero_jax.numpy import (
    stack as jnp_stack,
)
from zero_jax.numpy import (
    tile as jnp_tile,
)
from zero_jax.numpy import (
    unstack as jnp_unstack,
)
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, ndarray


def broadcast_like(arr: ArrayLikeSpec, like_arr: ArrayLikeSpec) -> ndarray:
    """Broadcast ``arr`` to the shape of ``like_arr``.

    Args:
        arr: Array to broadcast.
        like_arr: Prototype array defining the target shape.

    Returns:
        ndarray: Broadcasted array.
    """
    return broadcast_to(asarray(arr), asarray(like_arr).shape)


def broadcast_to_rank(operand: ArrayLikeSpec, rank: int) -> ndarray:
    """Broadcast ``operand`` by prepending singleton dimensions up to ``rank``.

    Args:
        operand: Input array.
        rank: Target tensor rank.

    Returns:
        ndarray: Rank-broadcasted array.
    """
    return _wrap(ops.broadcast_to_rank(_to_tensor(operand), rank))


def broadcasted_iota(dtype: ArrayLikeSpec, shape: ArrayLikeSpec, dimension: int) -> ndarray:
    """Create an array of ``shape`` and ``dtype`` with iota indices along ``dimension``.

    Args:
        dtype: Output dtype specification.
        shape: Output array shape.
        dimension: Axis along which indices increase from ``0``.

    Returns:
        ndarray: Broadcasted iota array.
    """
    return _wrap(ops.broadcasted_iota(dtype, shape, dimension))


def collapse(operand: ArrayLikeSpec, start_dimension: int, stop_dimension: int | None = None) -> ndarray:
    """Collapse dimensions of ``operand`` in ``[start_dimension, stop_dimension)`` into a single dimension.

    Args:
        operand: Input array to collapse.
        start_dimension: Inclusive start axis.
        stop_dimension: Exclusive stop axis (defaults to ``operand.ndim``).

    Returns:
        ndarray: Reshaped array with collapsed dimensions.
    """
    arr = asarray(operand)
    ndim = arr.ndim
    lo = start_dimension if start_dimension >= 0 else ndim + start_dimension
    hi = ndim if stop_dimension is None else (stop_dimension if stop_dimension >= 0 else ndim + stop_dimension)
    collapsed_size = int(math.prod(arr.shape[lo:hi])) if hi > lo else 1
    new_shape = (*arr.shape[:lo], collapsed_size, *arr.shape[hi:])
    return reshape(arr, new_shape)


def dtype(operand: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the dtype of an operand if available.

    Args:
        operand: Input array or scalar operand.

    Returns:
        ArrayLikeSpec: The dtype of ``operand`` or ``None``.
    """
    if isinstance(operand, ArrayProtocol):
        return operand.dtype
    return None


def dynamic_index_in_dim(operand: ArrayLikeSpec, index: ArrayLikeSpec, dimension: int = 0, keepdims: bool = True) -> ndarray:
    """Extract a dynamic single-element index along ``dimension``.

    Args:
        operand: Input array.
        index: Dynamic index along ``dimension``.
        dimension: Axis to index.
        keepdims: Whether to retain the indexed dimension of size 1.

    Returns:
        ndarray: Dynamically indexed slice.
    """
    return _wrap(ops.dynamic_index_in_dim(_to_tensor(operand), _to_tensor(index), dimension, keepdims))


def dynamic_slice_in_dim(operand: ArrayLikeSpec, start_index: ArrayLikeSpec, slice_size: int, dimension: int = 0) -> ndarray:
    """Extract a dynamic slice of length ``slice_size`` along ``dimension``.

    Args:
        operand: Input array.
        start_index: Dynamic starting index along ``dimension``.
        slice_size: Static slice length along ``dimension``.
        dimension: Axis to slice.

    Returns:
        ndarray: Dynamically sliced array.
    """
    return _wrap(ops.dynamic_slice_in_dim(_to_tensor(operand), _to_tensor(start_index), slice_size, dimension))


def dynamic_update_index_in_dim(operand: ArrayLikeSpec, update: ArrayLikeSpec, index: ArrayLikeSpec, dimension: int = 0) -> ndarray:
    """Update a single slice at ``index`` along ``dimension`` with ``update``.

    Args:
        operand: Target array.
        update: Replacement array.
        index: Dynamic index along ``dimension``.
        dimension: Axis to update.

    Returns:
        ndarray: Updated array.
    """
    return _wrap(ops.dynamic_update_index_in_dim(_to_tensor(operand), _to_tensor(update), _to_tensor(index), dimension))


def dynamic_update_slice_in_dim(operand: ArrayLikeSpec, update: ArrayLikeSpec, start_index: ArrayLikeSpec, dimension: int = 0) -> ndarray:
    """Update a contiguous slice starting at ``start_index`` along ``dimension`` with ``update``.

    Args:
        operand: Target array.
        update: Replacement slice array.
        start_index: Dynamic start index along ``dimension``.
        dimension: Axis to update.

    Returns:
        ndarray: Updated array.
    """
    return _wrap(ops.dynamic_update_slice_in_dim(_to_tensor(operand), _to_tensor(update), _to_tensor(start_index), dimension))


def empty(
    shape: ArrayLikeSpec,
    dtype: ArrayLikeSpec = "float32",
    *,
    out_sharding: ArrayLikeSpec = None,
) -> ndarray:
    """Return an uninitialized array of ``shape`` and ``dtype``.

    Args:
        shape: Output array shape.
        dtype: Output array dtype.
        out_sharding: Optional output sharding specification.

    Returns:
        ndarray: Allocated array.
    """
    _ = out_sharding
    return jnp_empty(shape, dtype=dtype)


def index_in_dim(operand: ArrayLikeSpec, index: ArrayLikeSpec, dimension: int = 0, keepdims: bool = True) -> ndarray:
    """Extract a static index along ``dimension``.

    Args:
        operand: Input array.
        index: Static integer index along ``dimension``.
        dimension: Axis to index.
        keepdims: Whether to keep the indexed axis of size 1.

    Returns:
        ndarray: Indexed array.
    """
    return _wrap(ops.index_in_dim(_to_tensor(operand), _to_tensor(index), dimension, keepdims))


def index_take(operand: ArrayLikeSpec, indices: ArrayLikeSpec, axes: ArrayLikeSpec) -> ndarray:
    """Take elements from ``operand`` along ``axes`` at ``indices``.

    Args:
        operand: Source array.
        indices: Index array specifying positions to gather.
        axes: Sequence of axes along which to take.

    Returns:
        ndarray: Gathered elements.
    """
    arr = asarray(operand)
    axes_flat = asarray(axes).__array__().ravel()
    axis = int(axes_flat[0]) % max(1, arr.ndim) if axes_flat.size > 0 else 0
    idx_arr = astype(asarray(indices), int32)
    return take(arr, idx_arr, axis=axis)


def iota(dtype: ArrayLikeSpec, size: int) -> ndarray:
    """Return a 1-D array of ``[0, 1, ..., size - 1]`` with ``dtype``.

    Args:
        dtype: Output dtype specification.
        size: Number of elements.

    Returns:
        ndarray: 1-D range array.
    """
    return arange(size, dtype=dtype)


def padtype_to_pads(
    shape: ArrayLikeSpec,
    window_shape: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
) -> list[tuple[int, int]]:
    """Convert a padding string (``"SAME"`` or ``"VALID"``) into explicit ``(low, high)`` padding pairs.

    Args:
        shape: Spatial dimensions of the input array.
        window_shape: Spatial dimensions of the window.
        window_strides: Strides of the window along each spatial dimension.
        padding: Padding mode string (``"SAME"``, ``"VALID"``) or sequence of pad pairs.

    Returns:
        list[tuple[int, int]]: Explicit ``(pad_low, pad_high)`` tuple for each spatial dimension.
    """
    in_dims = [max(1, int(s)) for s in asarray(shape).__array__().ravel()]
    win_dims = [max(1, int(w)) for w in asarray(window_shape).__array__().ravel()]
    str_dims = [max(1, int(st)) for st in asarray(window_strides).__array__().ravel()]
    if isinstance(padding, str):
        mode = padding.upper()
        if mode == "VALID":
            return [(0, 0) for _ in in_dims]
        pads: list[tuple[int, int]] = []
        for s, k, st in zip(in_dims, win_dims, str_dims):
            out_size = math.ceil(s / st)
            pad_total = max(0, (out_size - 1) * st + k - s)
            pad_low = pad_total // 2
            pads.append((pad_low, pad_total - pad_low))
        return pads
    pad_arr = asarray(padding).__array__()
    if pad_arr.ndim == 2 and pad_arr.shape[1] == 2:
        return [(int(row[0]), int(row[1])) for row in pad_arr]
    return [(int(v), int(v)) for v in pad_arr.ravel()]


def rev(operand: ArrayLikeSpec, dimensions: ArrayLikeSpec) -> ndarray:
    """Reverse ``operand`` along the specified ``dimensions``.

    Args:
        operand: Input array.
        dimensions: Sequence of axis indices to reverse.

    Returns:
        ndarray: Array with elements reversed along ``dimensions``.
    """
    arr = asarray(operand)
    if arr.ndim == 0:
        return arr
    dims = sorted({int(d) % arr.ndim for d in asarray(dimensions).__array__().ravel()})
    for d in dims:
        arr = flip(arr, axis=d)
    return arr


def scatter_apply(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    update_computation: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> ndarray:
    """Apply ``update_computation`` to ``operand`` at ``scatter_indices``.

    Args:
        operand: Target array.
        scatter_indices: Scatter index array.
        updates: Update values.
        update_computation: Function to apply at scattered positions.
        dimension_numbers: Scatter dimension numbers specification.
        indices_are_sorted: Whether scatter indices are sorted.
        unique_indices: Whether scatter indices are unique.

    Returns:
        ndarray: Updated array.
    """
    _ = (indices_are_sorted, unique_indices)
    return _wrap(
        ops.scatter_apply(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            update_computation,
        )
    )


def scatter_max(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> ndarray:
    """Scatter-maximum ``updates`` into ``operand`` at ``scatter_indices``.

    Args:
        operand: Target array.
        scatter_indices: Scatter index array.
        updates: Update values.
        dimension_numbers: Scatter dimension numbers specification.
        indices_are_sorted: Whether scatter indices are sorted.
        unique_indices: Whether scatter indices are unique.

    Returns:
        ndarray: Updated array.
    """
    return _wrap(
        ops.scatter_max(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


def scatter_min(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> ndarray:
    """Scatter-minimum ``updates`` into ``operand`` at ``scatter_indices``.

    Args:
        operand: Target array.
        scatter_indices: Scatter index array.
        updates: Update values.
        dimension_numbers: Scatter dimension numbers specification.
        indices_are_sorted: Whether scatter indices are sorted.
        unique_indices: Whether scatter indices are unique.

    Returns:
        ndarray: Updated array.
    """
    return _wrap(
        ops.scatter_min(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


def scatter_mul(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> ndarray:
    """Scatter-multiply ``updates`` into ``operand`` at ``scatter_indices``.

    Args:
        operand: Target array.
        scatter_indices: Scatter index array.
        updates: Update values.
        dimension_numbers: Scatter dimension numbers specification.
        indices_are_sorted: Whether scatter indices are sorted.
        unique_indices: Whether scatter indices are unique.

    Returns:
        ndarray: Updated array.
    """
    return _wrap(
        ops.scatter_mul(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


def scatter_sub(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: ArrayLikeSpec = None,
) -> ndarray:
    """Scatter-subtract ``updates`` from ``operand`` at ``scatter_indices``.

    Args:
        operand: Target array.
        scatter_indices: Scatter index array.
        updates: Values to subtract.
        dimension_numbers: :class:`ScatterDimensionNumbers` specification.
        indices_are_sorted: Whether indices are sorted.
        unique_indices: Whether indices are unique.
        mode: Out-of-bounds indexing mode.

    Returns:
        ndarray: Updated array.
    """
    _ = (dimension_numbers, indices_are_sorted, unique_indices, mode)
    op_arr = asarray(operand)
    base = op_arr.__array__().copy()
    idx_np = asarray(scatter_indices).__array__().ravel()
    upd_np = asarray(updates).__array__().ravel()
    for i, idx_val in enumerate(idx_np):
        base[int(idx_val)] -= upd_np[i % len(upd_np)]
    return array(base, dtype=op_arr.dtype)


def shape_as_value(shape: ArrayLikeSpec) -> tuple[ndarray, ...]:
    """Convert a static or dynamic ``shape`` tuple into a tuple of 0-D scalar arrays.

    Args:
        shape: Sequence of dimension sizes.

    Returns:
        tuple[ndarray, ...]: Tuple of scalar arrays representing each dimension size.
    """
    return tuple(array(int(d), dtype=int32) for d in shape)


def slice_in_dim(
    operand: ArrayLikeSpec,
    start_index: int | None,
    limit_index: int | None,
    stride: int = 1,
    dimension: int = 0,
) -> ndarray:
    """Extract a 1D slice along a single dimension.

    Args:
        operand: Input array.
        start_index: Inclusive start index (or ``None`` for ``0``).
        limit_index: Exclusive limit index (or ``None`` for ``shape[dimension]``).
        stride: Step size along ``dimension``.
        dimension: Axis to slice.

    Returns:
        ndarray: Sliced array.
    """
    t = _to_tensor(operand)
    ndim = len(t.shape)
    axis = dimension if dimension >= 0 else ndim + dimension
    starts = [0] * ndim
    stops = list(t.shape)
    strides = [1] * ndim
    starts[axis] = start_index if start_index is not None else 0
    stops[axis] = limit_index if limit_index is not None else t.shape[axis]
    strides[axis] = stride
    return _wrap(ops.strided_slice(t, starts, stops, strides))


def split(operand: ArrayLikeSpec, sizes: ArrayLikeSpec, axis: int = 0) -> tuple[ndarray, ...]:
    """Split ``operand`` along ``axis`` into sub-arrays of explicit ``sizes``.

    Args:
        operand: Input array to split.
        sizes: Sequence of section sizes along ``axis``.
        axis: Axis along which to split.

    Returns:
        tuple[ndarray, ...]: Tuple of split sub-arrays.
    """
    arr = asarray(operand)
    cuts: list[int] = []
    running = 0
    size_list = list(sizes)
    for s in size_list[:-1]:
        running += int(s)
        cuts.append(running)
    return tuple(jnp_split(arr, cuts, axis=axis))


def stack(operands: ArrayLikeSpec, axis: int = 0) -> ndarray:
    """Stack a sequence of arrays ``operands`` along a new ``axis``.

    Args:
        operands: Sequence of arrays with identical shape and dtype.
        axis: Axis in the result array along which the inputs are stacked.

    Returns:
        ndarray: Stacked array.
    """
    return jnp_stack(operands, axis=axis)


def tile(operand: ArrayLikeSpec, reps: ArrayLikeSpec) -> ndarray:
    """Construct an array by repeating ``operand`` the number of times given by ``reps``.

    Args:
        operand: Input array.
        reps: Number of repetitions along each axis.

    Returns:
        ndarray: Tiled array.
    """
    return jnp_tile(operand, reps)


def unstack(x: ArrayLikeSpec, axis: int = 0) -> tuple[ndarray, ...]:
    """Unpack ``x`` along ``axis`` into a tuple of arrays.

    Args:
        x: Input array to unstack.
        axis: Axis along which to unstack.

    Returns:
        tuple[ndarray, ...]: Tuple of unstacked arrays.
    """
    return tuple(jnp_unstack(x, axis=axis))


def zeros_like_array(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ndarray:
    """Return an array of zeros with the same shape and dtype as ``x``.

    Args:
        x: Prototype array.
        dtype: Optional override dtype.

    Returns:
        ndarray: Array of zeros matching ``x``.
    """
    return zeros_like(x, dtype=dtype)


__all__ = [
    "broadcast_like",
    "broadcast_to_rank",
    "broadcasted_iota",
    "collapse",
    "dtype",
    "dynamic_index_in_dim",
    "dynamic_slice",
    "dynamic_slice_in_dim",
    "dynamic_update_index_in_dim",
    "dynamic_update_slice",
    "dynamic_update_slice_in_dim",
    "empty",
    "gather",
    "index_in_dim",
    "index_take",
    "iota",
    "padtype_to_pads",
    "rev",
    "scatter",
    "scatter_add",
    "scatter_apply",
    "scatter_max",
    "scatter_min",
    "scatter_mul",
    "scatter_sub",
    "shape_as_value",
    "slice",
    "slice_in_dim",
    "split",
    "stack",
    "tile",
    "unstack",
    "zeros_like_array",
]
