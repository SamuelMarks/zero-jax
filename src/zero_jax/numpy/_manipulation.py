"""Array shape and structural manipulation routines for zero_jax.numpy."""

from __future__ import annotations

import builtins
from collections.abc import Callable

import ml_switcheroo_compiler as _ml_switcheroo_compiler
from ml_switcheroo_compiler import Tensor
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import TensorConfig as _TensorConfig

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import (
    ArrayLikeSpec,
    ArrayProtocol,
    ConfigValue,
    DTypeSpec,
    VariadicCallable,
)
from zero_jax.numpy._creation import asarray
from zero_jax.numpy._ndarray import _to_tensor, _wrap, ndarray
from zero_jax.numpy.tensor_utils import to_array as _to_array


def reshape(x: ArrayLikeSpec, newshape: tuple[int, ...]) -> ArrayLikeSpec:
    """Gives a new shape to an array without changing its data.

    Args:
        x (ArrayLikeSpec): Argument x.
        newshape (ArrayLikeSpec): Argument newshape.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.reshape(_to_tensor(x), newshape=newshape))


def ravel(a: ArrayLikeSpec, order: str = "C") -> ArrayLikeSpec:
    """Return a contiguous flattened array.

    Args:
        a: Input parameter ``a``.
        order: Input parameter ``order``.

    Returns:
        The operation result.
    """
    if order == "F":
        axes = list(range(a.ndim))[::-1]
        a = transpose(a, axes=axes)
    elif order not in ("C", "A", "K"):
        raise ValueError("order must be one of 'C', 'F', 'A', or 'K'")
    return reshape(a, (-1,))


def transpose(x: ArrayLikeSpec, axes: list[int] | None = None) -> ArrayLikeSpec:
    """Reverse or permute the axes of an array.

    Args:
        x (ArrayLikeSpec): Argument x.
        axes (ArrayLikeSpec): Argument axes.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    t = _to_tensor(x)
    if axes is not None:
        return _wrap(ops.permute(t, axes=axes))
    axes = list(range(len(t.shape))[::-1])
    return _wrap(ops.permute(t, axes=axes))


def swapaxes(a: ArrayLikeSpec, axis1: int, axis2: int) -> ArrayLikeSpec:
    """Interchange two axes of an array.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis1 (ArrayLikeSpec): Argument axis1.
        axis2 (ArrayLikeSpec): Argument axis2.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.swapaxes(_to_tensor(a), axis1=axis1, axis2=axis2))


def moveaxis(a: ArrayLikeSpec, source: ArrayLikeSpec, destination: ArrayLikeSpec) -> ArrayLikeSpec:
    """Move axes of an array to new positions.

    Args:
        a (ArrayLikeSpec): Argument a.
        source (ArrayLikeSpec): Argument source.
        destination (ArrayLikeSpec): Argument destination.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.moveaxis(_to_tensor(a), source=source, destination=destination))


def rollaxis(a: ArrayLikeSpec, axis: int, start: int = 0) -> ArrayLikeSpec:
    """Execute ``rollaxis``.

    Args:
        a: Input parameter ``a``.
        axis: Input parameter ``axis``.
        start: Input parameter ``start``.

    Returns:
        The operation result.
    """
    return moveaxis(a, axis, start)


def squeeze(a: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Remove axes of length one from a.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.squeeze(_to_tensor(a), axis=axis))


def expand_dims(a: ArrayLikeSpec, axis: int) -> ArrayLikeSpec:
    """Expand the shape of an array.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.expand_dims(_to_tensor(a), axis=axis))


def concatenate(arrays: list[ArrayLikeSpec], axis: int = 0) -> ArrayLikeSpec:
    """Join a sequence of arrays along an existing axis.

    Args:
        arrays (ArrayLikeSpec): Argument arrays.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    tensors = [_to_tensor(a) for a in arrays]
    return _wrap(ops.concatenate(tensors, axis=axis))


def stack(arrays: ArrayLikeSpec, axis: int = 0) -> ArrayLikeSpec:
    """Join a sequence of arrays along a new axis.

    Args:
        arrays (ArrayLikeSpec): Argument arrays.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    tensors = [_to_tensor(arr) for arr in arrays]
    return _wrap(ops.stack(tensors, axis=axis))


def vstack(tup: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stack arrays in sequence vertically (row wise).

    Args:
        tup (ArrayLikeSpec): Argument tup.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.vstack([_to_tensor(arr) for arr in tup]))


def hstack(tup: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stack arrays in sequence horizontally (column wise).

    Args:
        tup (ArrayLikeSpec): Argument tup.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.hstack([_to_tensor(arr) for arr in tup]))


def dstack(tup: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stack arrays in sequence depth wise (along third axis).

    Args:
        tup (ArrayLikeSpec): Argument tup.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.dstack([_to_tensor(arr) for arr in tup]))


def column_stack(tup: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stack 1-D arrays as columns into a 2-D array.

    Args:
        tup: Sequence of 1-D or 2-D arrays.

    Returns:
        The stacked array.
    """
    return _wrap(ops.column_stack([_to_tensor(t) for t in tup]))


def block(arrays: ArrayLikeSpec) -> ArrayLikeSpec:
    """Assemble an nd-array from nested lists of blocks.

    Args:
        arrays: An array or list of blocks.

    Returns:
        The block array.
    """

    def _to_tensor_tree(tree: ArrayLikeSpec) -> ArrayLikeSpec:
        """Recursively convert nested lists/tuples of arrays into tensors.

        Args:
            tree: Nested list, tuple, or array.

        Returns:
            ArrayLikeSpec: Nested list of converted tensors.
        """
        if isinstance(tree, (list, tuple)):
            return [_to_tensor_tree(x) for x in tree]
        return _to_tensor(tree)

    return _wrap(ops.block(_to_tensor_tree(arrays)))


def split(ary: ArrayLikeSpec, indices_or_sections: ArrayLikeSpec, axis: int = 0) -> ArrayLikeSpec:
    """Split an array into multiple sub-arrays as views into ary.

    Args:
        ary (ArrayLikeSpec): Argument ary.
        indices_or_sections (ArrayLikeSpec): Argument indices_or_sections.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.split(_to_tensor(ary), indices_or_sections, axis))


def array_split(ary: ArrayLikeSpec, indices_or_sections: ArrayLikeSpec, axis: int = 0) -> ArrayLikeSpec:
    """Split an array into multiple sub-arrays.

    Args:
        ary (ArrayLikeSpec): Argument ary.
        indices_or_sections (ArrayLikeSpec): Argument indices_or_sections.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.shape.splitting.array_split(_to_tensor(ary), indices_or_sections, axis))


def vsplit(ary: ArrayLikeSpec, indices_or_sections: ArrayLikeSpec) -> ArrayLikeSpec:
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        ary (ArrayLikeSpec): Argument ary.
        indices_or_sections (ArrayLikeSpec): Argument indices_or_sections.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.vsplit(_to_tensor(ary), indices_or_sections))


def hsplit(ary: ArrayLikeSpec, indices_or_sections: ArrayLikeSpec) -> ArrayLikeSpec:
    """Split an array into multiple sub-arrays horizontally (column-wise).

    Args:
        ary (ArrayLikeSpec): Argument ary.
        indices_or_sections (ArrayLikeSpec): Argument indices_or_sections.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.hsplit(_to_tensor(ary), indices_or_sections))


def dsplit(ary: ArrayLikeSpec, indices_or_sections: ArrayLikeSpec) -> ArrayLikeSpec:
    """Split array into multiple sub-arrays along the 3rd axis (depth).

    Args:
        ary (ArrayLikeSpec): Argument ary.
        indices_or_sections (ArrayLikeSpec): Argument indices_or_sections.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.dsplit(_to_tensor(ary), indices_or_sections))


def unstack(x: ArrayLikeSpec, axis: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for unstack.

    Args:
        x: Argument x.
        axis: Argument axis.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.unstack(_to_tensor(x), axis=axis)
    return tuple(_wrap(t) for t in res)


def tile(A: ArrayLikeSpec, reps: ArrayLikeSpec) -> ArrayLikeSpec:
    """Construct an array by repeating A the number of times given by reps.

    Args:
        A (ArrayLikeSpec): Argument A.
        reps (ArrayLikeSpec): Argument reps.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    reps_tup = (reps,) if isinstance(reps, int) else tuple(int(x.item()) if isinstance(x, ndarray) else int(x) for x in reps)
    return _wrap(ops.tile(_to_tensor(A), reps=reps_tup))


def repeat(a: ArrayLikeSpec, repeats: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Repeat elements of an array.

    Args:
        a (ArrayLikeSpec): Argument a.
        repeats (ArrayLikeSpec): Argument repeats.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.repeat(_to_tensor(a), repeats=repeats, axis=axis))


def pad(array: ArrayLikeSpec, pad_width: ArrayLikeSpec, mode: str = "constant", **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Pad an array.

    Args:
        array: Input parameter ``array``.
        pad_width: Input parameter ``pad_width``.
        mode: Input parameter ``mode``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _wrap(ops.pad(_to_tensor(array), pad_width, mode=mode, **kwargs))


def roll(a: ArrayLikeSpec, shift: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for roll.

    Args:
        a: Argument a.
        shift: Argument shift.
        axis: Argument axis.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.roll(_to_tensor(a), shift=shift, axis=axis))


def flip(m: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Reverse the order of elements in an array along the given axis.

    Args:
        m: Input array.
        axis: Axis or axes along which to flip over.

    Returns:
        A view of m with the entries of axis reversed.
    """
    return _wrap(ops.flip(_to_tensor(m), axis=axis))


def fliplr(m: ArrayLikeSpec) -> ArrayLikeSpec:
    """Reverse the order of elements along axis 1 (left/right).

    Args:
        m: Input array.

    Returns:
        A view of m with the columns reversed.
    """
    return _wrap(ops.fliplr(_to_tensor(m)))


def flipud(m: ArrayLikeSpec) -> ArrayLikeSpec:
    """Reverse the order of elements along axis 0 (up/down).

    Args:
        m: Input array.

    Returns:
        A view of m with the rows reversed.
    """
    return _wrap(ops.flipud(_to_tensor(m)))


def rot90(m: ArrayLikeSpec, k: int = 1, axes: tuple[int, int] = (0, 1)) -> ArrayLikeSpec:
    """Rotate an array by 90 degrees in the plane specified by ``axes``.

    Args:
        m: Array of two or more dimensions.
        k: Number of times the array is rotated by 90 degrees.
        axes: The array is rotated in the plane defined by the two axes.

    Returns:
        ArrayLikeSpec: Rotated view/copy of ``m``.
    """
    arr = asarray(m)
    k_mod = int(k) % 4
    ax0, ax1 = int(axes[0]), int(axes[1])
    if k_mod == 0:
        return arr
    if k_mod == 2:
        return flip(flip(arr, axis=ax0), axis=ax1)
    if k_mod == 1:
        return swapaxes(flip(arr, axis=ax1), ax0, ax1)
    return flip(swapaxes(arr, ax0, ax1), axis=ax1)


def atleast_1d(*arys: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert inputs to arrays with at least one dimension.

    Args:
        *arys: One or more input arrays.

    Returns:
        An array, or list of arrays, each with a.ndim >= 1.
    """
    return _wrap(ops.atleast_1d(*[_to_tensor(a) for a in arys]))


def atleast_2d(*arys: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert inputs to arrays with at least two dimensions.

    Args:
        *arys: One or more array-like sequences.

    Returns:
        An array, or list of arrays, each with a.ndim >= 2.
    """
    return _wrap(ops.atleast_2d(*[_to_tensor(a) for a in arys]))


def atleast_3d(*arys: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert inputs to arrays with at least three dimensions.

    Args:
        *arys: One or more array-like sequences.

    Returns:
        An array, or list of arrays, each with a.ndim >= 3.
    """
    return _wrap(ops.atleast_3d(*[_to_tensor(a) for a in arys]))


def broadcast_to(x: ArrayLikeSpec, shape: tuple[int, ...]) -> ArrayLikeSpec:
    """Broadcast an array to a new shape.

    Args:
        x (ArrayLikeSpec): Argument x.
        shape (ArrayLikeSpec): Argument shape.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.broadcast_to(_to_tensor(x), shape=shape))


def broadcast_arrays(*args: ArrayLikeSpec) -> list[ndarray]:
    """Broadcast any number of arrays against each other.

    Args:
        *args: The arrays to broadcast.

    Returns:
        list[ndarray]: A list of broadcasted arrays.
    """
    return [_wrap(t) for t in ops.broadcast_arrays(*[_to_tensor(a) for a in args])]


def broadcast_shapes(*shapes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Broadcast the input shapes into a single shape.

    Args:
        *shapes: Input parameter ``shapes``.

    Returns:
        The operation result.
    """
    import functools

    from ml_switcheroo_compiler.core.shape import broadcast_shapes as _broadcast_shapes

    if not shapes:
        return ()
    return functools.reduce(_broadcast_shapes, shapes)


def concat(arrays: ArrayLikeSpec, axis: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for concat.

    Args:
        arrays: Argument arrays.
        axis: Argument axis.

    Returns:
        ArrayLikeSpec: The result.
    """
    return concatenate(arrays, axis=axis)


def matrix_transpose(x: ArrayLikeSpec, /) -> ArrayLikeSpec:
    """Transposes a matrix (or a stack of matrices) x.

    Args:
        x: Input array.

    Returns:
        The transposed array.
    """
    return _wrap(ops.matrix_transpose(_to_tensor(x)))


def permute_dims(a: ArrayLikeSpec, axes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``permute_dims``.

    Args:
        a: Input parameter ``a``.
        axes: Input parameter ``axes``.

    Returns:
        The operation result.
    """
    return transpose(a, axes)


def resize(a: ArrayLikeSpec, new_shape: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for resize.

    Args:
        a: Input parameter ``a``.
        new_shape: Input parameter ``new_shape``.

    Returns:
        The operation result.
    """
    return _wrap(ops.resize(_to_tensor(a), new_shape))


def shape(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """Get the shape of the array.

    Args:
        a: Input parameter ``a``.

    Returns:
        The operation result.
    """
    return asarray(a).shape


def ndim(a: ArrayLikeSpec) -> int:
    """Return the number of dimensions of an array.

    Args:
        a: Input array.

    Returns:
        The number of dimensions in a.
    """
    return len(_to_tensor(a).shape) if _to_tensor(a).shape is not None else 0


def size(a: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``size``.

    Args:
        a: Input parameter ``a``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return _wrap(ops.size(_to_tensor(a), axis=axis))


def append(arr: ArrayLikeSpec, values: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Append values to the end of an array.

    Args:
        arr: Values are appended to a copy of this array.
        values: These values are appended to a copy of arr.
        axis: The axis along which values are appended.

    Returns:
        A copy of arr with values appended to axis.
    """
    return _wrap(ops.append(_to_tensor(arr), _to_tensor(values), axis=axis))


def delete(arr: ArrayLikeSpec, obj: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array with sub-arrays along an axis deleted.

    Args:
        arr: Input array.
        obj: Indicate indices of sub-arrays to remove along the specified axis.
        axis: The axis along which to delete the subarray.

    Returns:
        A copy of arr with the elements specified by obj removed.
    """
    return _wrap(
        ops.delete(
            _to_tensor(arr),
            _to_tensor(obj) if isinstance(obj, (list, tuple, ndarray, Tensor, ArrayProtocol)) else obj,
            axis=axis,
        )
    )


def insert(arr: ArrayLikeSpec, obj: ArrayLikeSpec, values: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Insert values along the given axis before the given indices.

    Args:
        arr: Input array.
        obj: Object that defines the index or indices before which values is inserted.
        values: Values to insert into arr.
        axis: Axis along which to insert values.

    Returns:
        A copy of arr with values inserted.
    """
    return _wrap(ops.insert(_to_tensor(arr), obj, _to_tensor(values), axis=axis))


def trim_zeros(filt: ArrayLikeSpec, trim: str = "fb") -> ArrayLikeSpec:
    """Execute ``trim_zeros``.

    Args:
        filt: Input parameter ``filt``.
        trim: Input parameter ``trim``.

    Returns:
        The operation result.
    """
    return _wrap(ops.trim_zeros(_to_tensor(filt), trim))


def row_stack(tup: ArrayLikeSpec, *, dtype: DTypeSpec = None, casting: str = "same_kind") -> ArrayLikeSpec:
    """Stack arrays in sequence vertically (row wise).

    Args:
        tup: Sequence of arrays to stack.
        dtype: Optional output dtype.
        casting: Casting rule string.

    Returns:
        ArrayLikeSpec: Vertically stacked array.
    """
    _ = (dtype, casting)
    return vstack(tup)


__all__ = [
    "append",
    "array_split",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "block",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "column_stack",
    "concat",
    "concatenate",
    "delete",
    "dsplit",
    "dstack",
    "expand_dims",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "insert",
    "matrix_transpose",
    "moveaxis",
    "ndim",
    "pad",
    "permute_dims",
    "ravel",
    "repeat",
    "reshape",
    "resize",
    "roll",
    "rollaxis",
    "rot90",
    "row_stack",
    "shape",
    "size",
    "split",
    "squeeze",
    "stack",
    "swapaxes",
    "tile",
    "transpose",
    "trim_zeros",
    "unstack",
    "vsplit",
    "vstack",
]
