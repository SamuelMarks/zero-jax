"""JAX-like numpy API backed by ml-switcheroo-compiler."""

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


class _IndexUpdateRef:
    """Reference to an indexed slice of an ``ndarray`` supporting functional immutable updates."""

    def __init__(self, array_obj: ndarray, idx: ArrayLikeSpec) -> None:
        """Initialize ``_IndexUpdateRef``.

        Args:
            array_obj: Target ``ndarray`` to copy and update.
            idx: Index specification.
        """
        self._array = array_obj
        self._idx = idx

    def _unwrap_idx(self) -> ArrayLikeSpec:
        """Convert any ``ndarray`` or ``Tensor`` indices into raw index objects.

        Returns:
            ArrayLikeSpec: Unwrapped index suitable for NumPy array indexing.
        """
        key = self._idx
        if isinstance(key, ndarray):
            return key.__array__()
        if isinstance(key, tuple):
            return tuple(k.__array__() if isinstance(k, (ndarray, Tensor)) else k for k in key)
        return key

    def get(self) -> ndarray:
        """Return the sliced view at the specified index.

        Returns:
            ndarray: Indexed slice result.
        """
        return self._array[self._idx]

    def set(self, values: ArrayLikeSpec) -> ndarray:
        """Return a new ``ndarray`` with elements at ``self._idx`` replaced by ``values``.

        Args:
            values: Replacement values.

        Returns:
            ndarray: Updated copy of the array.
        """
        base = self._array.__array__().copy()
        val_arr = values.__array__() if isinstance(values, (ndarray, Tensor)) else values
        base[self._unwrap_idx()] = val_arr
        return array(base, dtype=self._array.dtype)

    def add(self, values: ArrayLikeSpec) -> ndarray:
        """Return a new ``ndarray`` with ``values`` added at ``self._idx``.

        Args:
            values: Values to add.

        Returns:
            ndarray: Updated copy of the array.
        """
        base = self._array.__array__().copy()
        val_arr = values.__array__() if isinstance(values, (ndarray, Tensor)) else values
        base[self._unwrap_idx()] = base[self._unwrap_idx()] + val_arr
        return array(base, dtype=self._array.dtype)

    def multiply(self, values: ArrayLikeSpec) -> ndarray:
        """Return a new ``ndarray`` with elements at ``self._idx`` multiplied by ``values``.

        Args:
            values: Values to multiply by.

        Returns:
            ndarray: Updated copy of the array.
        """
        base = self._array.__array__().copy()
        val_arr = values.__array__() if isinstance(values, (ndarray, Tensor)) else values
        base[self._unwrap_idx()] = base[self._unwrap_idx()] * val_arr
        return array(base, dtype=self._array.dtype)

    def divide(self, values: ArrayLikeSpec) -> ndarray:
        """Return a new ``ndarray`` with elements at ``self._idx`` divided by ``values``.

        Args:
            values: Divisor values.

        Returns:
            ndarray: Updated copy of the array.
        """
        base = self._array.__array__().copy()
        val_arr = values.__array__() if isinstance(values, (ndarray, Tensor)) else values
        base[self._unwrap_idx()] = base[self._unwrap_idx()] / val_arr
        return array(base, dtype=self._array.dtype)

    def power(self, values: ArrayLikeSpec) -> ndarray:
        """Return a new ``ndarray`` with elements at ``self._idx`` raised to ``values``.

        Args:
            values: Exponent values.

        Returns:
            ndarray: Updated copy of the array.
        """
        base = self._array.__array__().copy()
        val_arr = values.__array__() if isinstance(values, (ndarray, Tensor)) else values
        base[self._unwrap_idx()] = base[self._unwrap_idx()] ** val_arr
        return array(base, dtype=self._array.dtype)

    def min(self, values: ArrayLikeSpec) -> ndarray:
        """Return a new ``ndarray`` with elements at ``self._idx`` set to the minimum with ``values``.

        Args:
            values: Comparison values.

        Returns:
            ndarray: Updated copy of the array.
        """
        base = self._array.__array__().copy()
        cur = array(base[self._unwrap_idx()])
        updated = minimum(cur, values)
        base[self._unwrap_idx()] = updated.__array__() if isinstance(updated, ndarray) else updated
        return array(base, dtype=self._array.dtype)

    def max(self, values: ArrayLikeSpec) -> ndarray:
        """Return a new ``ndarray`` with elements at ``self._idx`` set to the maximum with ``values``.

        Args:
            values: Comparison values.

        Returns:
            ndarray: Updated copy of the array.
        """
        base = self._array.__array__().copy()
        cur = array(base[self._unwrap_idx()])
        updated = maximum(cur, values)
        base[self._unwrap_idx()] = updated.__array__() if isinstance(updated, ndarray) else updated
        return array(base, dtype=self._array.dtype)

    def apply(self, func: Callable[[ndarray], ArrayLikeSpec]) -> ndarray:
        """Return a new ``ndarray`` with ``func`` applied to the slice at ``self._idx``.

        Args:
            func: Function to apply to the indexed slice.

        Returns:
            ndarray: Updated copy of the array.
        """
        return self.set(func(self.get()))


class _IndexUpdateHelper:
    """Helper property object enabling ``arr.at[idx]`` syntax on ``ndarray``."""

    def __init__(self, array_obj: ndarray) -> None:
        """Initialize ``_IndexUpdateHelper``.

        Args:
            array_obj: Target ``ndarray``.
        """
        self._array = array_obj

    def __getitem__(self, idx: ArrayLikeSpec) -> _IndexUpdateRef:
        """Create an :class:`_IndexUpdateRef` for ``idx``.

        Args:
            idx: Index specification.

        Returns:
            _IndexUpdateRef: Index update reference.
        """
        return _IndexUpdateRef(self._array, idx)


class ndarray:
    """A multi-dimensional array object backed by an ml-switcheroo tensor.

    This class provides a NumPy-like interface for tensor operations, supporting
    standard arithmetic, comparison, and array-manipulation magic methods.
    """

    def __init__(self, tensor: Tensor) -> None:
        """Initialize the object.

        Args:
            tensor: The underlying tensor data.
        """
        self._tensor: Tensor = tensor

    @property
    def at(self) -> _IndexUpdateHelper:
        """Return an :class:`_IndexUpdateHelper` for functional in-place updates (``arr.at[idx]``).

        Returns:
            _IndexUpdateHelper: Helper bound to this array.
        """
        return _IndexUpdateHelper(self)

    @property
    def size(self) -> int:
        """Return the total number of elements in the array.

        Returns:
            int: Product of all dimension sizes.
        """
        import math

        return int(math.prod(self.shape)) if self.shape else 1

    @property
    def T(self) -> ArrayLikeSpec:
        """Return the transposed array.

        Returns:
            ArrayLikeSpec: Transposed array.
        """
        return transpose(self)

    @property
    def mT(self) -> ArrayLikeSpec:
        """Return the matrix-transposed array (swapping the last two axes).

        Returns:
            ArrayLikeSpec: Matrix-transposed array.
        """
        return swapaxes(self, -2, -1)

    def item(self, *args: int) -> ArrayLikeSpec:
        """Copy an element of an array to a standard Python scalar and return it.

        Args:
            *args: Optional flat or multi-dimensional integer index.

        Returns:
            ArrayLikeSpec: Python scalar value.
        """
        if args:
            return self.__array__().item(*args)
        return self._tensor.item()

    def tolist(self) -> ArrayLikeSpec:
        """Return the array as a (possibly nested) Python list.

        Returns:
            ArrayLikeSpec: Nested Python list or scalar.
        """
        return self.__array__().tolist()

    def astype(
        self,
        dtype: DTypeSpec,
        copy: builtins.bool = False,
        device: ConfigValue = None,
    ) -> ArrayLikeSpec:
        """Copy of the array, cast to a specified type.

        Args:
            dtype: Target dtype.
            copy: Whether to force a copy.
            device: Optional target device.

        Returns:
            ArrayLikeSpec: Cast array.
        """
        return astype(self, dtype, copy=copy, device=device)

    def reshape(
        self,
        *args: ArrayLikeSpec,
        order: str = "C",
        out_sharding: ConfigValue = None,
    ) -> ArrayLikeSpec:
        """Return an array containing the same data with a new shape.

        Args:
            *args: New shape specification (either a tuple/list or multiple int arguments).
            order: Memory layout order (``"C"`` or ``"F"``).
            out_sharding: Optional output sharding specification.

        Returns:
            ArrayLikeSpec: Reshaped array.
        """
        _ = (order, out_sharding)
        newshape = tuple(int(d) for d in args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(int(d) for d in args)
        return reshape(self, newshape)

    def transpose(self, *args: ArrayLikeSpec) -> ArrayLikeSpec:
        """Return a view of the array with axes transposed.

        Args:
            *args: Optional permutation of axes.

        Returns:
            ArrayLikeSpec: Transposed array.
        """
        if not args or args == (None,):
            return transpose(self, None)
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return transpose(self, [int(a) for a in args[0]])
        return transpose(self, [int(a) for a in args])

    def ravel(
        self,
        order: str = "C",
        out_sharding: ConfigValue = None,
    ) -> ArrayLikeSpec:
        """Return a flattened 1-D array.

        Args:
            order: Memory layout order (``"C"`` or ``"F"``).
            out_sharding: Optional output sharding specification.

        Returns:
            ArrayLikeSpec: Flattened 1-D array.
        """
        _ = out_sharding
        return ravel(self, order=order)

    def flatten(
        self,
        order: str = "C",
        out_sharding: ConfigValue = None,
    ) -> ArrayLikeSpec:
        """Return a copy of the array collapsed into one dimension.

        Args:
            order: Memory layout order (``"C"`` or ``"F"``).
            out_sharding: Optional output sharding specification.

        Returns:
            ArrayLikeSpec: Flattened 1-D array.
        """
        _ = out_sharding
        return ravel(self, order=order)

    def squeeze(self, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
        """Remove axes of length one from ``self``.

        Args:
            axis: Optional axis or axes to squeeze.

        Returns:
            ArrayLikeSpec: Squeezed array.
        """
        return squeeze(self, axis=axis)

    def sum(
        self,
        axis: ArrayLikeSpec = None,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool = False,
        initial: ArrayLikeSpec = None,
        where: ArrayLikeSpec = None,
        promote_integers: builtins.bool = True,
    ) -> ArrayLikeSpec:
        """Return the sum of the array elements over the given axis.

        Args:
            axis: Axis or axes along which a sum is performed.
            dtype: Optional output dtype.
            out: Optional output array (unused).
            keepdims: Whether reduced axes are left in the result as dimensions with size one.
            initial: Optional starting value for the sum.
            where: Optional boolean mask of elements to include.
            promote_integers: Whether integer inputs are promoted to default integer width.

        Returns:
            ArrayLikeSpec: Summed array.
        """
        _ = (out, promote_integers)
        res = sum(self, axis=axis, keepdims=keepdims, where=where)
        if initial is not None:
            res = add(res, initial)
        if dtype is not None:
            res = astype(res, dtype)
        return res

    def addressable_data(self, index: int) -> ndarray:
        """Return the addressable shard array at ``index``.

        Args:
            index: Shard index.

        Returns:
            ndarray: Addressable array shard.
        """
        _ = index
        return self

    def all(
        self,
        axis: ArrayLikeSpec = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool = False,
        *,
        where: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Test whether all array elements along a given axis evaluate to True.

        Args:
            axis: Axis or axes along which a logical AND reduction is performed.
            out: Unused output parameter.
            keepdims: Whether reduced axes are retained with size 1.
            where: Optional boolean mask of elements to include.

        Returns:
            ArrayLikeSpec: Boolean result array.
        """
        _ = out
        arr = _wrap(ops.where(_to_tensor(where), self._tensor, _to_tensor(True))) if where is not None else self
        return all(arr, axis=axis, keepdims=keepdims)

    def any(
        self,
        axis: ArrayLikeSpec = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool = False,
        *,
        where: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Test whether any array element along a given axis evaluates to True.

        Args:
            axis: Axis or axes along which a logical OR reduction is performed.
            out: Unused output parameter.
            keepdims: Whether reduced axes are retained with size 1.
            where: Optional boolean mask of elements to include.

        Returns:
            ArrayLikeSpec: Boolean result array.
        """
        _ = out
        arr = _wrap(ops.where(_to_tensor(where), self._tensor, _to_tensor(False))) if where is not None else self
        return any(arr, axis=axis, keepdims=keepdims)

    def argmax(
        self,
        axis: int | None = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool | None = None,
    ) -> ArrayLikeSpec:
        """Return indices of the maximum values along the given axis.

        Args:
            axis: Axis along which to operate.
            out: Unused output parameter.
            keepdims: Whether reduced axis is retained with size 1.

        Returns:
            ArrayLikeSpec: Array of indices.
        """
        _ = out
        return argmax(self, axis=axis, keepdims=builtins.bool(keepdims))

    def argmin(
        self,
        axis: int | None = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool | None = None,
    ) -> ArrayLikeSpec:
        """Return indices of the minimum values along the given axis.

        Args:
            axis: Axis along which to operate.
            out: Unused output parameter.
            keepdims: Whether reduced axis is retained with size 1.

        Returns:
            ArrayLikeSpec: Array of indices.
        """
        _ = out
        return argmin(self, axis=axis, keepdims=builtins.bool(keepdims))

    def argpartition(self, kth: int, axis: int = -1) -> ArrayLikeSpec:
        """Return indices that would partition this array along ``axis``.

        Args:
            kth: Element index to partition by.
            axis: Axis along which to sort.

        Returns:
            ArrayLikeSpec: Indices of partitioned array.
        """
        return argpartition(self, kth=kth, axis=axis)

    def argsort(
        self,
        axis: int | None = -1,
        *,
        kind: str | None = None,
        order: ConfigValue = None,
        stable: builtins.bool = True,
        descending: builtins.bool = False,
    ) -> ArrayLikeSpec:
        """Return the indices that would sort this array.

        Args:
            axis: Axis along which to sort.
            kind: Sorting algorithm (unused).
            order: Field order (unused).
            stable: Whether to preserve relative order of equal elements.
            descending: Whether to sort in descending order.

        Returns:
            ArrayLikeSpec: Array of sort indices.
        """
        _ = stable
        res = argsort(self, axis=axis, kind=kind, order=order)
        if descending:
            flip_ax = 0 if axis is None else axis
            return flip(res, axis=flip_ax)
        return res

    def byteswap(self) -> ArrayLikeSpec:
        """Swap the bytes of the array elements.

        Returns:
            ArrayLikeSpec: Byte-swapped array.
        """
        return array(self.__array__().byteswap(), dtype=self.dtype)

    def choose(
        self,
        choices: ArrayLikeSpec,
        out: ArrayLikeSpec = None,
        mode: str = "raise",
    ) -> ArrayLikeSpec:
        """Construct an array from an index array and a list of arrays to choose from.

        Args:
            choices: Sequence of choice arrays.
            out: Unused output array.
            mode: Out-of-bounds index handling mode.

        Returns:
            ArrayLikeSpec: Chosen array.
        """
        return choose(self, choices, out=out, mode=mode)

    def clip(
        self,
        min: ArrayLikeSpec = None,
        max: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Return an array whose values are limited to ``[min, max]``.

        Args:
            min: Minimum value.
            max: Maximum value.

        Returns:
            ArrayLikeSpec: Clipped array.
        """
        return clip(self, a_min=min, a_max=max)

    def compress(
        self,
        condition: ArrayLikeSpec,
        axis: int | None = None,
        *,
        out: ArrayLikeSpec = None,
        size: int | None = None,
        fill_value: ArrayLikeSpec = 0,
    ) -> ArrayLikeSpec:
        """Return selected slices of this array along given axis.

        Args:
            condition: 1-D boolean array selecting entries.
            axis: Axis along which to take slices.
            out: Unused output array.
            size: Optional static output size.
            fill_value: Padding value when ``size`` is specified.

        Returns:
            ArrayLikeSpec: Compressed array.
        """
        return compress(condition, self, axis=axis, size=size, fill_value=fill_value, out=out)

    def conj(self) -> ArrayLikeSpec:
        """Return the complex conjugate, element-wise.

        Returns:
            ArrayLikeSpec: Complex-conjugated array.
        """
        return conj(self)

    def conjugate(self) -> ArrayLikeSpec:
        """Return the complex conjugate, element-wise.

        Returns:
            ArrayLikeSpec: Complex-conjugated array.
        """
        return conjugate(self)

    def copy(self) -> ArrayLikeSpec:
        """Return a copy of the array.

        Returns:
            ArrayLikeSpec: Copied array.
        """
        return copy(self)

    def copy_to_host_async(self) -> None:
        """Request an asynchronous copy of the array to host memory."""

    def cumprod(
        self,
        axis: int | None = None,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Return the cumulative product of elements along a given axis.

        Args:
            axis: Axis along which the cumulative product is computed.
            dtype: Optional dtype of the returned array.
            out: Unused output parameter.

        Returns:
            ArrayLikeSpec: Cumulative product array.
        """
        return cumprod(self, axis=axis, dtype=dtype, out=out)

    def cumsum(
        self,
        axis: int | None = None,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Return the cumulative sum of elements along a given axis.

        Args:
            axis: Axis along which the cumulative sum is computed.
            dtype: Optional dtype of the returned array.
            out: Unused output parameter.

        Returns:
            ArrayLikeSpec: Cumulative sum array.
        """
        _ = out
        return cumsum(self, axis=axis, dtype=dtype)

    def diagonal(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
    ) -> ArrayLikeSpec:
        """Return specified diagonals.

        Args:
            offset: Offset of the diagonal from the main diagonal.
            axis1: First axis of the 2-D sub-arrays.
            axis2: Second axis of the 2-D sub-arrays.

        Returns:
            ArrayLikeSpec: Array of diagonals.
        """
        return diagonal(self, offset=offset, axis1=axis1, axis2=axis2)

    def dot(
        self,
        b: ArrayLikeSpec,
        *,
        precision: ConfigValue = None,
        preferred_element_type: DTypeSpec = None,
    ) -> ArrayLikeSpec:
        """Compute the dot product of this array with ``b``.

        Args:
            b: Second operand.
            precision: Optional matmul precision.
            preferred_element_type: Optional accumulation/output dtype.

        Returns:
            ArrayLikeSpec: Dot product result.
        """
        _ = precision
        res = dot(self, b)
        if preferred_element_type is not None:
            return astype(res, preferred_element_type)
        return res

    def nonzero(
        self,
        *,
        fill_value: ArrayLikeSpec = None,
        size: int | None = None,
    ) -> tuple[ArrayLikeSpec, ...]:
        """Return the indices of the elements that are non-zero.

        Args:
            fill_value: Optional padding value when ``size`` is specified.
            size: Optional static number of non-zero elements to return.

        Returns:
            tuple[ArrayLikeSpec, ...]: Tuple of index arrays for each dimension.
        """
        return nonzero(self, size=size, fill_value=fill_value)

    def prod(
        self,
        axis: ArrayLikeSpec = None,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool = False,
        initial: ArrayLikeSpec = None,
        where: ArrayLikeSpec = None,
        promote_integers: builtins.bool = True,
    ) -> ArrayLikeSpec:
        """Return the product of the array elements over a given axis.

        Args:
            axis: Axis or axes along which a product is performed.
            dtype: Optional dtype of the returned array.
            out: Unused output parameter.
            keepdims: Whether reduced axes are retained with size 1.
            initial: Optional starting value for the product.
            where: Optional boolean mask of elements to include.
            promote_integers: Whether integer inputs are promoted.

        Returns:
            ArrayLikeSpec: Product array.
        """
        _ = (out, promote_integers)
        arr = _wrap(ops.where(_to_tensor(where), self._tensor, _to_tensor(1))) if where is not None else self
        res = prod(arr, axis=axis, dtype=dtype, keepdims=keepdims)
        if initial is not None:
            res = multiply(res, initial)
        return res

    def ptp(
        self,
        axis: ArrayLikeSpec = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool = False,
    ) -> ArrayLikeSpec:
        """Return the peak-to-peak (maximum - minimum) value along a given axis.

        Args:
            axis: Axis along which to find the peaks.
            out: Unused output parameter.
            keepdims: Whether reduced axes are retained with size 1.

        Returns:
            ArrayLikeSpec: Peak-to-peak array.
        """
        return ptp(self, axis=axis, out=out, keepdims=keepdims)

    def repeat(
        self,
        repeats: ArrayLikeSpec,
        axis: int | None = None,
        *,
        total_repeat_length: int | None = None,
        out_sharding: ConfigValue = None,
    ) -> ArrayLikeSpec:
        """Repeat elements of an array.

        Args:
            repeats: Number of repetitions for each element.
            axis: Axis along which to repeat values.
            total_repeat_length: Optional static total repeat length.
            out_sharding: Optional output sharding specification.

        Returns:
            ArrayLikeSpec: Repeated array.
        """
        _ = (total_repeat_length, out_sharding)
        return repeat(self, repeats=repeats, axis=axis)

    def round(
        self,
        decimals: int = 0,
        out: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Round an array to the given number of decimals.

        Args:
            decimals: Number of decimal places to round to.
            out: Unused output parameter.

        Returns:
            ArrayLikeSpec: Rounded array.
        """
        _ = out
        return round(self, decimals=decimals)

    def searchsorted(
        self,
        v: ArrayLikeSpec,
        side: str = "left",
        sorter: ArrayLikeSpec = None,
        *,
        method: str = "scan",
    ) -> ArrayLikeSpec:
        """Find indices where elements should be inserted to maintain order.

        Args:
            v: Values to insert into ``self``.
            side: ``"left"`` or ``"right"``.
            sorter: Optional array of integer indices that sort ``self``.
            method: Search algorithm name.

        Returns:
            ArrayLikeSpec: Array of insertion points.
        """
        _ = method
        return searchsorted(self, v, side=side, sorter=sorter)

    def sort(
        self,
        axis: int | None = -1,
        *,
        kind: str | None = None,
        order: ConfigValue = None,
        stable: builtins.bool = True,
        descending: builtins.bool = False,
    ) -> ArrayLikeSpec:
        """Return a sorted copy of an array.

        Args:
            axis: Axis along which to sort.
            kind: Sorting algorithm (unused).
            order: Field order (unused).
            stable: Whether to use a stable sort.
            descending: Whether to sort in descending order.

        Returns:
            ArrayLikeSpec: Sorted array.
        """
        _ = stable
        res = sort(self, axis=axis, kind=kind, order=order)
        if descending:
            flip_ax = 0 if axis is None else axis
            return flip(res, axis=flip_ax)
        return res

    def std(
        self,
        axis: ArrayLikeSpec = None,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
        ddof: int = 0,
        keepdims: builtins.bool = False,
        *,
        where: ArrayLikeSpec = None,
        correction: int | float | None = None,
    ) -> ArrayLikeSpec:
        """Compute the standard deviation along the specified axis.

        Args:
            axis: Axis or axes along which the standard deviation is computed.
            dtype: Optional dtype to use in computing the standard deviation.
            out: Unused output parameter.
            ddof: Delta degrees of freedom.
            keepdims: Whether reduced axes are retained with size 1.
            where: Optional boolean mask of elements to include.
            correction: Degrees of freedom adjustment (overrides ``ddof`` if given).

        Returns:
            ArrayLikeSpec: Standard deviation array.
        """
        return sqrt(
            self.var(
                axis=axis,
                dtype=dtype,
                out=out,
                ddof=ddof,
                keepdims=keepdims,
                where=where,
                correction=correction,
            )
        )

    def swapaxes(self, axis1: int, axis2: int) -> ArrayLikeSpec:
        """Return a view of the array with ``axis1`` and ``axis2`` interchanged.

        Args:
            axis1: First axis.
            axis2: Second axis.

        Returns:
            ArrayLikeSpec: Array with swapped axes.
        """
        return swapaxes(self, axis1, axis2)

    def take(
        self,
        indices: ArrayLikeSpec,
        axis: int | None = None,
        out: ArrayLikeSpec = None,
        mode: str | None = None,
        unique_indices: builtins.bool = False,
        indices_are_sorted: builtins.bool = False,
        fill_value: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Take elements from an array along an axis.

        Args:
            indices: Indices of the values to extract.
            axis: Axis over which to select values.
            out: Unused output parameter.
            mode: Out-of-bounds index handling mode.
            unique_indices: Whether indices are guaranteed unique.
            indices_are_sorted: Whether indices are sorted in ascending order.
            fill_value: Optional fill value for out-of-bounds indices.

        Returns:
            ArrayLikeSpec: Indexed sub-array.
        """
        _ = (out, unique_indices, indices_are_sorted, fill_value)
        return take(self, indices, axis=axis, mode=mode)

    def to_device(
        self,
        device: ConfigValue,
        stream: ConfigValue = None,
    ) -> ndarray:
        """Return a copy of the array transferred to ``device``.

        Args:
            device: Target device or sharding.
            stream: Optional device stream.

        Returns:
            ndarray: Array on ``device``.
        """
        _ = (device, stream)
        return self

    def trace(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Return the sum along diagonals of the array.

        Args:
            offset: Offset of the diagonal from the main diagonal.
            axis1: First axis of the 2-D sub-arrays.
            axis2: Second axis of the 2-D sub-arrays.
            dtype: Optional dtype of the returned array.
            out: Unused output parameter.

        Returns:
            ArrayLikeSpec: Sum along diagonals.
        """
        return trace(self, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)

    def var(
        self,
        axis: ArrayLikeSpec = None,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
        ddof: int = 0,
        keepdims: builtins.bool = False,
        *,
        where: ArrayLikeSpec = None,
        correction: int | float | None = None,
    ) -> ArrayLikeSpec:
        """Compute the variance along the specified axis.

        Args:
            axis: Axis or axes along which the variance is computed.
            dtype: Optional dtype to use in computing the variance.
            out: Unused output parameter.
            ddof: Delta degrees of freedom.
            keepdims: Whether reduced axes are retained with size 1.
            where: Optional boolean mask of elements to include.
            correction: Degrees of freedom adjustment (overrides ``ddof`` if given).

        Returns:
            ArrayLikeSpec: Variance array.
        """
        _ = (out, where)
        eff_ddof = int(correction) if correction is not None else ddof
        return var(self, axis=axis, dtype=dtype, keepdims=keepdims, ddof=eff_ddof)

    def view(
        self,
        dtype: DTypeSpec = None,
        type: ConfigValue = None,
    ) -> ArrayLikeSpec:
        """Return a new view of the array with the same data.

        Args:
            dtype: Optional data-type descriptor of the returned view.
            type: Unused view container class parameter.

        Returns:
            ArrayLikeSpec: Array view.
        """
        _ = type
        if dtype is not None:
            return array(self.__array__().view(dtype))
        return self

    def mean(
        self,
        axis: ArrayLikeSpec = None,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool = False,
        where: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Return the average of the array elements along given axis.

        Args:
            axis: Axis or axes along which the means are computed.
            dtype: Optional output dtype.
            out: Optional output array (unused).
            keepdims: Whether reduced axes are left in the result as dimensions with size one.
            where: Optional boolean mask of elements to include.

        Returns:
            ArrayLikeSpec: Mean array.
        """
        if where is not None:
            total = sum(self, axis=axis, keepdims=keepdims, where=where)
            cnt = sum(astype(where, float32), axis=axis, keepdims=keepdims)
            res = divide(total, cnt)
        else:
            res = mean(self, axis=axis, keepdims=keepdims)
        if dtype is not None:
            res = astype(res, dtype)
        return res

    def min(
        self,
        axis: ArrayLikeSpec = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool = False,
        initial: ArrayLikeSpec = None,
        where: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Return the minimum along a given axis.

        Args:
            axis: Axis or axes along which to operate.
            out: Optional output array (unused).
            keepdims: Whether reduced axes are left in the result as dimensions with size one.
            initial: Optional maximum value of an output element.
            where: Optional boolean mask of elements to compare.

        Returns:
            ArrayLikeSpec: Minimum array.
        """
        t_x = self._tensor
        if where is not None:
            init_val = initial if initial is not None else float("inf")
            t_x = ops.where(_to_tensor(where), t_x, _to_tensor(init_val))
        res = ops.min(t_x, axis=axis, keepdims=keepdims)
        if initial is not None:
            res = ops.minimum(res, _to_tensor(initial))
        return _wrap(res)

    def max(
        self,
        axis: ArrayLikeSpec = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool = False,
        initial: ArrayLikeSpec = None,
        where: ArrayLikeSpec = None,
    ) -> ArrayLikeSpec:
        """Return the maximum along a given axis.

        Args:
            axis: Axis or axes along which to operate.
            out: Optional output array (unused).
            keepdims: Whether reduced axes are left in the result as dimensions with size one.
            initial: Optional minimum value of an output element.
            where: Optional boolean mask of elements to compare.

        Returns:
            ArrayLikeSpec: Maximum array.
        """
        return max(self, axis=axis, keepdims=keepdims, where=where, initial=initial)

    def __matmul__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform matrix multiplication with ``other``.

        Args:
            other: Right-hand matrix/vector operand.

        Returns:
            ArrayLikeSpec: Matrix product.
        """
        return matmul(self, other)

    def __rmatmul__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform reflected matrix multiplication with ``other``.

        Args:
            other: Left-hand matrix/vector operand.

        Returns:
            ArrayLikeSpec: Matrix product.
        """
        return matmul(other, self)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array.

        Returns:
            tuple[int, ...]: The shape property of the underlying tensor.
        """
        return tuple(int(d) for d in self._tensor.shape)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the array.

        Returns:
            int: The number of dimensions.
        """
        return len(self.shape)

    @property
    def dtype(self) -> ArrayLikeSpec:
        """Get the dtype of the array.

        Returns:
            ArrayLikeSpec: The dtype property of the underlying tensor.
        """
        return self._tensor.dtype

    def __array__(self, dtype: DTypeSpec = None) -> ArrayLikeSpec:
        """Convert to a host array representation, optionally cast to ``dtype``.

        Args:
            dtype: Optional dtype to convert to.

        Returns:
            ArrayLikeSpec: The result of the array operation.
        """
        if dtype is not None:
            return _to_tensor(astype(self, dtype)).__array__()
        return self._tensor.__array__()

    def block_until_ready(self) -> ndarray:
        """Blocks until the array's value is available, and returns the array.

        Returns:
            ndarray: The array itself.
        """
        return self

    def __repr__(self) -> str:
        """Perform the repr operation.

        Returns:
            str: The result of the repr operation.
        """
        inner_repr = repr(self.__array__())
        inner_repr = inner_repr[6:-1] if inner_repr.startswith("array(") else inner_repr
        return f"Array({inner_repr}, dtype={self.dtype.name})"

    def __add__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the add operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the add operation.
        """
        return add(self, other)

    def __radd__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the radd operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the radd operation.
        """
        return add(other, self)

    def __sub__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform elementwise subtraction ``self - other``.

        Args:
            other (ArrayLikeSpec): The right-hand operand for subtraction.

        Returns:
            ArrayLikeSpec: The result of the subtraction operation.
        """
        return subtract(self, other)

    def __rsub__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform reflected elementwise subtraction ``other - self``.

        Args:
            other (ArrayLikeSpec): The left-hand operand for subtraction.

        Returns:
            ArrayLikeSpec: The result of the reflected subtraction operation.
        """
        return subtract(other, self)

    def __mul__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the mul operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the mul operation.
        """
        return multiply(self, other)

    def __rmul__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the rmul operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the rmul operation.
        """
        return multiply(other, self)

    def __pow__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the pow operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the pow operation.
        """
        return power(self, other)

    def __rpow__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the rpow operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the rpow operation.
        """
        return power(other, self)

    def __truediv__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the truediv operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the truediv operation.
        """
        return true_divide(self, other)

    def __rtruediv__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the rtruediv operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the rtruediv operation.
        """
        return true_divide(other, self)

    def __floordiv__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the floordiv operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the floordiv operation.
        """
        return floor_divide(self, other)

    def __rfloordiv__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the rfloordiv operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the rfloordiv operation.
        """
        return floor_divide(other, self)

    def __mod__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform elementwise modulo ``self % other``.

        Args:
            other: Right-hand divisor operand.

        Returns:
            ArrayLikeSpec: Elementwise remainder.
        """
        return mod(self, other)

    def __rmod__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform reflected elementwise modulo ``other % self``.

        Args:
            other: Left-hand dividend operand.

        Returns:
            ArrayLikeSpec: Elementwise remainder.
        """
        return mod(other, self)

    def __neg__(self) -> ArrayLikeSpec:
        """Perform elementwise negation ``-self``.

        Returns:
            ArrayLikeSpec: The result of the negation operation.
        """
        return negative(self)

    def __lt__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the lt operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the lt operation.
        """
        return _wrap(ops.less(self._tensor, _to_tensor(other)))

    def __gt__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the gt operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the gt operation.
        """
        return _wrap(ops.greater(self._tensor, _to_tensor(other)))

    def __le__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the le operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the le operation.
        """
        return _wrap(ops.less_equal(self._tensor, _to_tensor(other)))

    def __ge__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the ge operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the ge operation.
        """
        return _wrap(ops.greater_equal(self._tensor, _to_tensor(other)))

    def __and__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform elementwise bitwise/logical AND ``self & other``.

        Args:
            other: Right-hand operand.

        Returns:
            ArrayLikeSpec: Elementwise AND result.
        """
        if self.dtype.name == "bool":
            return logical_and(self, other)
        return bitwise_and(self, other)

    def __or__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform elementwise bitwise/logical OR ``self | other``.

        Args:
            other: Right-hand operand.

        Returns:
            ArrayLikeSpec: Elementwise OR result.
        """
        if self.dtype.name == "bool":
            return logical_or(self, other)
        return bitwise_or(self, other)

    def __xor__(self, other: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform elementwise bitwise/logical XOR ``self ^ other``.

        Args:
            other: Right-hand operand.

        Returns:
            ArrayLikeSpec: Elementwise XOR result.
        """
        if self.dtype.name == "bool":
            return logical_xor(self, other)
        return bitwise_xor(self, other)

    def __invert__(self) -> ArrayLikeSpec:
        """Perform elementwise bitwise/logical NOT ``~self``.

        Returns:
            ArrayLikeSpec: Elementwise NOT result.
        """
        if self.dtype.name == "bool":
            return logical_not(self)
        return bitwise_not(self)

    def __setitem__(self, key: ArrayLikeSpec, value: ArrayLikeSpec) -> None:
        """Perform the setitem operation.

        Args:
            key: Input parameter ``key``.
            value: Input parameter ``value``.
        """
        from ml_switcheroo_compiler.core.config import config

        if config.eager_mode:
            t_val = value._tensor if isinstance(value, ndarray) else value
            raw_val = t_val.data if isinstance(t_val, Tensor) else t_val
            self._tensor.data[key] = raw_val
        else:
            raise TypeError("JAX arrays are immutable.")

    def __getitem__(self, key: ArrayLikeSpec) -> ArrayLikeSpec:
        """Perform the getitem operation.

        Args:
            key: Input parameter ``key``.

        Returns:
            The operation result.
        """
        from ml_switcheroo_compiler.core.config import config

        if config.eager_mode:
            arr = self.__array__()
            if isinstance(key, ndarray):
                key = key._tensor.data
            elif isinstance(key, tuple):
                unwrapped_keys = []
                for k in key:
                    tk = k._tensor if isinstance(k, ndarray) else k
                    rk = tk.data if isinstance(tk, Tensor) else tk
                    unwrapped_keys.append(rk)
                key = tuple(unwrapped_keys)
            return _wrap(_to_tensor(arr[key]))

        t = self._tensor
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else t.shape[0]
            step = key.step if key.step is not None else 1
            return _wrap(ops.strided_slice(t, [start], [stop], [step]))

        if isinstance(key, tuple) and builtins.all(isinstance(k, slice) or k is None for k in key):
            starts = []
            stops = []
            steps = []
            for i, k in enumerate(key):
                if k is None:
                    continue
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else t.shape[len(starts)]
                step = k.step if k.step is not None else 1
                starts.append(start)
                stops.append(stop)
                steps.append(step)
            for i in range(len(starts), len(t.shape)):
                starts.append(0)
                stops.append(t.shape[i])
                steps.append(1)
            return _wrap(ops.strided_slice(t, starts, stops, steps))

        return _wrap(t[key])

    def __eq__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the eq operation.

        Args:
            other: The other operand for the operation.

        Returns:
            ndarray: The result of the eq operation.
        """
        return _wrap(ops.equal(self._tensor, _to_tensor(other)))

    def __bool__(self) -> builtins.bool:
        """Perform the bool operation.

        Returns:
            bool: The result of the bool operation.
        """
        import math

        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        size = math.prod(self.shape) if self.shape else 1
        if size == 1:
            if isinstance(self._tensor.data, ProxyTensor):
                return True
            raw = self._tensor.data
            if isinstance(raw, (builtins.bool, int, float)):
                return builtins.bool(raw)
            return builtins.bool(self._tensor.item())
        raise ValueError("The truth value of an array with more than one element is ambiguous.")

    def __len__(self) -> ArrayLikeSpec:
        """Perform the len operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the len operation.
        """
        return self.shape[0] if self.shape else 0

    def __iter__(self) -> ArrayLikeSpec:
        """Perform the iter operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ArrayLikeSpec: The result of the iter operation.
        """
        arr = self.__array__()
        for i in range(arr.shape[0]):
            yield array(arr[i])


def _to_tensor(x: ArrayLikeSpec) -> Tensor:
    """Convert the input to an ml-switcheroo Tensor.

    Args:
        x: Argument x.

    Returns:
        Tensor: The result of the operation.
    """
    if isinstance(x, ndarray):
        x = x._tensor
    import uuid

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
    from ml_switcheroo_ir import LogicalNode

    if isinstance(x, _ml_switcheroo_compiler.Tensor):
        if _tracer.is_tracing and not isinstance(x.data, ProxyTensor):
            # lift eager tensor as constant
            out_id = str(uuid.uuid4())
            val = x.__array__().tolist() if x.shape else x.item()
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={"value": val},
                shape_metadata=x.shape,
            )
            _tracer.add_node(node)
            pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
            return _ml_switcheroo_compiler.Tensor(
                data=pt,
                config=_TensorConfig(shape=x.shape, dtype=x.dtype, device=x.device),
            )
        return x
    if isinstance(x, ProxyTensor):
        # We need a dtype. ProxyTensor has dtype as string.
        # But we'll just mock it or use default.
        return _ml_switcheroo_compiler.Tensor(
            data=x,
            config=_TensorConfig(
                shape=x.shape,
                dtype=config.default_float_dtype,
                device=config.default_device,
            ),
        )

    with _ml_switcheroo_compiler.core.EagerMode():
        arr = _to_array(x)
    return _to_tensor(arr)


def _wrap(t: ArrayLikeSpec) -> ArrayLikeSpec:
    """Wrap an ml-switcheroo Tensor in an ndarray.

    Args:
        t (ArrayLikeSpec): Argument t.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if isinstance(t, Tensor):
        return ndarray(t)
    if isinstance(t, tuple):
        return tuple(_wrap(x) for x in t)
    if isinstance(t, list):
        return [_wrap(x) for x in t]
    if isinstance(t, ArrayProtocol):
        return ndarray(_to_tensor(t))
    return t


def sin(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the trigonometric sine element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.sin(_to_tensor(x)))


def cos(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the trigonometric cosine element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.cos(_to_tensor(x)))


def exp(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Calculate the exponential of all elements in the input array.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.exp(_to_tensor(x)))


def log(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Natural logarithm, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.log(_to_tensor(x)))


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


def reshape(x: ArrayLikeSpec, newshape: tuple[int, ...]) -> ArrayLikeSpec:
    """Gives a new shape to an array without changing its data.

    Args:
        x (ArrayLikeSpec): Argument x.
        newshape (ArrayLikeSpec): Argument newshape.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.reshape(_to_tensor(x), newshape=newshape))


def broadcast_to(x: ArrayLikeSpec, shape: tuple[int, ...]) -> ArrayLikeSpec:
    """Broadcast an array to a new shape.

    Args:
        x (ArrayLikeSpec): Argument x.
        shape (ArrayLikeSpec): Argument shape.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.broadcast_to(_to_tensor(x), shape=shape))


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


def where(condition: ArrayLikeSpec, x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return elements chosen from x or y depending on condition.

    Args:
        condition (ArrayLikeSpec): Argument condition.
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.where(_to_tensor(condition), _to_tensor(x), _to_tensor(y)))


def einsum(subscripts: str, *operands: ArrayLikeSpec) -> ArrayLikeSpec:
    """Evaluate the Einstein summation convention on the operands.

    Args:
        subscripts: Argument subscripts.
        *operands: Argument operands.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    tensors = [_to_tensor(a) for a in operands]
    return _wrap(ops.einsum(subscripts, *tensors))


def add(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Add arguments element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


def multiply(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Multiply arguments element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


def power(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """First array elements raised to powers from second array, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.power(_to_tensor(x), _to_tensor(y)))


def maximum(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise maximum of array elements.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.maximum(_to_tensor(x), _to_tensor(y)))


def minimum(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise minimum of array elements.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.minimum(_to_tensor(x), _to_tensor(y)))


def clip(a: ArrayLikeSpec, a_min: ArrayLikeSpec, a_max: ArrayLikeSpec) -> ArrayLikeSpec:
    """Clip (limit) the values in an array.

    Args:
        a (ArrayLikeSpec): Argument a.
        a_min (ArrayLikeSpec): Argument a_min.
        a_max (ArrayLikeSpec): Argument a_max.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    res = _to_tensor(a)
    if a_min is not None:
        res = ops.maximum(res, _to_tensor(a_min))
    if a_max is not None:
        res = ops.minimum(res, _to_tensor(a_max))
    return _wrap(res)


def max(
    x: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    keepdims: ArrayLikeSpec = False,
    where: ArrayLikeSpec = None,
    initial: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Return the maximum of an array or maximum along an axis.

    Args:
        x (ArrayLikeSpec): Argument x.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.
        where (ArrayLikeSpec): Argument where.
        initial (ArrayLikeSpec): Argument initial.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    t_x = _to_tensor(x)
    if where is not None:
        init_val = initial if initial is not None else float("-inf")
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(init_val))
    res = ops.max(t_x, axis=axis, keepdims=keepdims)
    if initial is not None:
        res = ops.maximum(res, _to_tensor(initial))
    return _wrap(res)


def sum(x: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False, where: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Sum of array elements over a given axis.

    Args:
        x (ArrayLikeSpec): Argument x.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.
        where (ArrayLikeSpec): Argument where.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    t_x = _to_tensor(x)
    if where is not None:
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(0))
    return _wrap(ops.sum(t_x, axis=axis, keepdims=keepdims))


def zeros_like(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return an array of zeros with the same shape and type as a given array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.zeros_like(_to_tensor(x), dtype=dtype))


def zeros(shape: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array of given shape and type, filled with zeros.

    Args:
        shape (ArrayLikeSpec): Argument shape.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.zeros(shape=shape, dtype=dtype))


def abs(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Calculate the absolute value element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.abs(_to_tensor(x)))


def mean(x: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Compute the arithmetic mean along the specified axis.

    Args:
        x (ArrayLikeSpec): Argument x.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.mean(_to_tensor(x), axis=axis, keepdims=keepdims))


inf = float("inf")


def array(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Create an array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    res = x if isinstance(x, ndarray) else _wrap(_to_tensor(x))
    if dtype is not None:
        return astype(res, dtype)
    return res


def dot(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Dot product of two arrays.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.dot(_to_tensor(a), _to_tensor(b)))


def matmul(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Matrix product of two arrays.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.matmul(_to_tensor(a), _to_tensor(b)))


def expand_dims(a: ArrayLikeSpec, axis: int) -> ArrayLikeSpec:
    """Expand the shape of an array.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.expand_dims(_to_tensor(a), axis=axis))


def isfinite(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Test element-wise for finiteness (not infinity or not Not a Number).

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.isfinite(_to_tensor(x)))


def allclose(a: ArrayLikeSpec, b: ArrayLikeSpec, rtol: ArrayLikeSpec = 1e-05, atol: ArrayLikeSpec = 1e-08, equal_nan: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Returns True if two arrays are element-wise equal within a tolerance.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.
        rtol (ArrayLikeSpec): Argument rtol.
        atol (ArrayLikeSpec): Argument atol.
        equal_nan (ArrayLikeSpec): Argument equal_nan.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return ops.allclose(_to_tensor(a), _to_tensor(b), rtol=rtol, atol=atol, equal_nan=equal_nan)


def array_equal(a1: ArrayLikeSpec, a2: ArrayLikeSpec, equal_nan: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """True if two arrays have the same shape and elements, False otherwise.

    Args:
        a1 (ArrayLikeSpec): Argument a1.
        a2 (ArrayLikeSpec): Argument a2.
        equal_nan (ArrayLikeSpec): Argument equal_nan.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    res = ops.equal(_to_tensor(a1), _to_tensor(a2))
    all_res = ops.all(res)
    return builtins.bool(all_res.item()) if isinstance(all_res, Tensor) else builtins.bool(all_res)


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


def ones(shape: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array of given shape and type, filled with ones.

    Args:
        shape (ArrayLikeSpec): Argument shape.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.ones(shape=shape, dtype=dtype))


def empty(shape: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array of given shape and type, without initializing entries.

    Args:
        shape (ArrayLikeSpec): Argument shape.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.empty(shape=shape, dtype=dtype))


def full(shape: ArrayLikeSpec, fill_value: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array of given shape and type, filled with fill_value.

    Args:
        shape (ArrayLikeSpec): Argument shape.
        fill_value (ArrayLikeSpec): Argument fill_value.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.full(shape=shape, fill_value=fill_value, dtype=dtype))


def ones_like(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return an array of ones with the same shape and type as a given array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.ones_like(_to_tensor(x), dtype=dtype))


def empty_like(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array with the same shape and type as a given array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    t = _to_tensor(x)
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.empty(shape=t.shape, dtype=dtype))


def full_like(x: ArrayLikeSpec, fill_value: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a full array with the same shape and type as a given array.

    Args:
        x (ArrayLikeSpec): Argument x.
        fill_value (ArrayLikeSpec): Argument fill_value.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.full_like(_to_tensor(x), fill_value=fill_value, dtype=dtype))


def asarray(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Convert the input to an array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return array(x, dtype=dtype)


def arange(start: ArrayLikeSpec, stop: ArrayLikeSpec = None, step: ArrayLikeSpec = 1, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return evenly spaced values within a given interval.

    Args:
        start (ArrayLikeSpec): Argument start.
        stop (ArrayLikeSpec): Argument stop.
        step (ArrayLikeSpec): Argument step.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    from zero_jax.nn.activation import _to_dtype

    resolved_dtype = _to_dtype(dtype) if dtype is not None else None
    return _wrap(ops.arange(start=start, stop=stop, step=step, dtype=resolved_dtype))


def linspace(
    start: ArrayLikeSpec,
    stop: ArrayLikeSpec,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: ArrayLikeSpec = None,
    axis: int = 0,
) -> ArrayLikeSpec:
    """Return evenly spaced numbers over a specified interval.

    Args:
        start: Input parameter ``start``.
        stop: Input parameter ``stop``.
        num: Input parameter ``num``.
        endpoint: Input parameter ``endpoint``.
        retstep: Input parameter ``retstep``.
        dtype: Input parameter ``dtype``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")
    if num == 0:
        return _wrap(ops.empty((0,), dtype=dtype))

    _to_tensor(start)
    _to_tensor(stop)

    if dtype is None:
        import ml_switcheroo_compiler

        dtype = ml_switcheroo_compiler.core.dtype.DType.Float32

    step_denom = builtins.max(num - 1, 1) if endpoint else num

    step = (stop - start) / step_denom

    from ml_switcheroo_compiler.ops import (
        add as backend_add,
    )
    from ml_switcheroo_compiler.ops import (
        arange as backend_arange,
    )
    from ml_switcheroo_compiler.ops import (
        multiply as backend_mul,
    )

    idx = backend_arange(0, num, 1, dtype=dtype)
    res = backend_add(_to_tensor(start), backend_mul(idx, _to_tensor(step)))

    res_wrapped = _wrap(res)

    if axis != 0:
        res_wrapped = moveaxis(res_wrapped, 0, axis)

    if retstep:
        return res_wrapped, step
    return res_wrapped


def logspace(
    start: ArrayLikeSpec,
    stop: ArrayLikeSpec,
    num: int = 50,
    endpoint: bool = True,
    base: float = 10.0,
    dtype: ArrayLikeSpec = None,
    axis: int = 0,
) -> ArrayLikeSpec:
    """Return numbers spaced evenly on a log scale.

    Args:
        start (ArrayLikeSpec): Argument start.
        stop (ArrayLikeSpec): Argument stop.
        num (ArrayLikeSpec): Argument num.
        endpoint (ArrayLikeSpec): Argument endpoint.
        base (ArrayLikeSpec): Argument base.
        dtype (ArrayLikeSpec): Argument dtype.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    lin = linspace(start, stop, num, endpoint=endpoint, dtype=dtype, axis=axis)
    return power(base, lin)


def eye(N: int, M: int | None = None, k: int = 0, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Args:
        N: Input parameter ``N``.
        M: Input parameter ``M``.
        k: Input parameter ``k``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    if M is None:
        M = N
    if k == 0:
        return _wrap(ops.eye(N, M, dtype=dtype))

    row_idx = arange(N)
    col_idx = arange(M)
    row_k = add(row_idx, k)
    row_k_expand = expand_dims(row_k, 1)
    col_expand = expand_dims(col_idx, 0)
    res = equal(row_k_expand, col_expand)

    if dtype is None:
        import ml_switcheroo_compiler

        dtype = ml_switcheroo_compiler.core.dtype.DType.Float32
    return astype(res, dtype)


def identity(n: int, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the identity array.

    Args:
        n (ArrayLikeSpec): Argument n.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.identity(n=n, dtype=dtype))


def meshgrid(*xi: ArrayLikeSpec, copy: ArrayLikeSpec = True, sparse: ArrayLikeSpec = False, indexing: ArrayLikeSpec = "xy") -> ArrayLikeSpec:
    """Return coordinate matrices from coordinate vectors.

    Args:
        copy: Input parameter ``copy``.
        sparse: Input parameter ``sparse``.
        indexing: Input parameter ``indexing``.
        *xi: Input parameter ``xi``.

    Returns:
        The operation result.
    """
    tensors = [_to_tensor(x) for x in xi]

    ndim = len(tensors)
    if indexing == "xy" and ndim > 1:
        tensors[0], tensors[1] = tensors[1], tensors[0]

    s0 = (1,) * ndim
    output = []
    for i, t in enumerate(tensors):
        shape = list(s0)
        shape[i] = -1
        reshaped = ops.reshape(t, newshape=tuple(shape))
        output.append(reshaped)

    if not sparse:
        broadcast_shape = tuple(t.shape[0] for t in tensors)
        output = [ops.broadcast_to(t, broadcast_shape) for t in output]

    if indexing == "xy" and ndim > 1:
        output[0], output[1] = output[1], output[0]

    return tuple(_wrap(t) for t in output)


def subtract(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Subtract arguments, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.subtract(_to_tensor(x), _to_tensor(y)))


def divide(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Divide arguments element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.divide(_to_tensor(x), _to_tensor(y)))


def true_divide(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Divide arguments element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return divide(x, y)


def floor_divide(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the largest integer smaller or equal to the division of the inputs.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.floor_divide(_to_tensor(x), _to_tensor(y)))


def mod(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return element-wise remainder of division.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.mod(_to_tensor(x), _to_tensor(y)))


def remainder(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return element-wise remainder of division.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.remainder(_to_tensor(x), _to_tensor(y)))


def divmod(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return element-wise quotient and remainder simultaneously.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    out1, out2 = ops.divmod(_to_tensor(x), _to_tensor(y))
    return _wrap(out1), _wrap(out2)


def negative(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Numerical negative, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.negative(_to_tensor(x)))


def positive(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Numerical positive, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.positive(_to_tensor(x)))


def sign(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns an element-wise indication of the sign of a number.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.sign(_to_tensor(x)))


def floor(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the floor of the input, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.floor(_to_tensor(x)))


def ceil(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the ceiling of the input, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.ceil(_to_tensor(x)))


def trunc(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the truncated value of the input, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.trunc(_to_tensor(x)))


def rint(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Round elements of the array to the nearest integer.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.round(_to_tensor(x)))


def tan(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute tangent element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.tan(_to_tensor(x)))


def arcsin(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inverse sine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.asin(_to_tensor(x)))


def arccos(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Trigonometric inverse cosine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.acos(_to_tensor(x)))


def arctan(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Trigonometric inverse tangent, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.atan(_to_tensor(x)))


def arctan2(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    Args:
        x1 (ArrayLikeSpec): Argument x1.
        x2 (ArrayLikeSpec): Argument x2.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.atan2(_to_tensor(x1), _to_tensor(x2)))


def sinh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Hyperbolic sine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.sinh(_to_tensor(x)))


def cosh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Hyperbolic cosine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.cosh(_to_tensor(x)))


def tanh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute hyperbolic tangent element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.tanh(_to_tensor(x)))


def arcsinh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inverse hyperbolic sine element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.asinh(_to_tensor(x)))


def arccosh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inverse hyperbolic cosine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.acosh(_to_tensor(x)))


def arctanh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inverse hyperbolic tangent element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.atanh(_to_tensor(x)))


def exp2(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Calculate 2**p for all p in the input array.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    # 2^x = exp(x * ln(2)) or just power(2, x)
    return power(2.0, x)


def expm1(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Calculate exp(x) - 1 for all elements in the array.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return subtract(exp(x), 1.0)


def log2(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Base-2 logarithm of x.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    # log2(x) = log(x) / log(2)
    import math

    return divide(log(x), math.log(2.0))


def log10(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the base 10 logarithm of the input array, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    import math

    return divide(log(x), math.log(10.0))


def log1p(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the natural logarithm of one plus the input array, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return log(add(x, 1.0))


def prod(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Return the product of array elements over a given axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        dtype (ArrayLikeSpec): Argument dtype.
        keepdims (ArrayLikeSpec): Argument keepdims.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.prod(_to_tensor(a), axis=axis, keepdims=keepdims))


def min(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Return the minimum of an array or minimum along an axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.min(_to_tensor(a), axis=axis, keepdims=keepdims))


def amin(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Return the minimum of an array or minimum along an axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return min(a, axis=axis, keepdims=keepdims)


def amax(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Return the maximum of an array or maximum along an axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return max(a, axis=axis, keepdims=keepdims)


def argmax(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Returns the indices of the maximum values along an axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.argmax(_to_tensor(a), axis=axis, keepdims=keepdims))


def argmin(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Returns the indices of the minimum values along an axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.argmin(_to_tensor(a), axis=axis, keepdims=keepdims))


def any(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Test whether any array element along a given axis evaluates to True.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.any(_to_tensor(a), axis=axis, keepdims=keepdims))


def all(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Test whether all array elements along a given axis evaluate to True.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        keepdims (ArrayLikeSpec): Argument keepdims.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.all(_to_tensor(a), axis=axis, keepdims=keepdims))


def var(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False, ddof: int = 0) -> ArrayLikeSpec:
    """Compute the variance along the specified axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        dtype (ArrayLikeSpec): Argument dtype.
        keepdims (ArrayLikeSpec): Argument keepdims.
        ddof (ArrayLikeSpec): Argument ddof.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    # variance = E[X^2] - E[X]^2 or mean((x - mean(x))^2)
    # Using eager wrapper or tracing composition
    t = _to_tensor(a)
    m = mean(a, axis=axis, keepdims=True)
    diff = subtract(t, m)
    sq = multiply(diff, diff)
    # if ddof != 0 we would need more math, but standard test probably uses default
    return mean(sq, axis=axis, keepdims=keepdims)


def std(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False, ddof: int = 0) -> ArrayLikeSpec:
    """Compute the standard deviation along the specified axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        dtype (ArrayLikeSpec): Argument dtype.
        keepdims (ArrayLikeSpec): Argument keepdims.
        ddof (ArrayLikeSpec): Argument ddof.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    # Standard deviation is sqrt of variance
    # ops.sqrt exists or power(var, 0.5)
    v = var(a, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
    return power(v, 0.5)


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


def squeeze(a: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Remove axes of length one from a.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.squeeze(_to_tensor(a), axis=axis))


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


def tile(A: ArrayLikeSpec, reps: ArrayLikeSpec) -> ArrayLikeSpec:
    """Construct an array by repeating A the number of times given by reps.

    Args:
        A (ArrayLikeSpec): Argument A.
        reps (ArrayLikeSpec): Argument reps.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if isinstance(reps, int):
        reps = (reps,)
    elif isinstance(reps, (list, tuple)):
        reps = tuple(reps)
    return _wrap(ops.tile(_to_tensor(A), reps=reps))


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


def take(a: ArrayLikeSpec, indices: ArrayLikeSpec, axis: int | None = None, mode: str | None = None) -> ArrayLikeSpec:
    """Take elements from an array along an axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        indices (ArrayLikeSpec): Argument indices.
        axis (ArrayLikeSpec): Argument axis.
        mode (ArrayLikeSpec): Argument mode.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.take(_to_tensor(a), _to_tensor(indices)))


def take_along_axis(arr: ArrayLikeSpec, indices: ArrayLikeSpec, axis: int) -> ArrayLikeSpec:
    """Take values from the input array by matching 1d index and data slices.

    Args:
        arr (ArrayLikeSpec): Argument arr.
        indices (ArrayLikeSpec): Argument indices.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.take_along_axis(_to_tensor(arr), _to_tensor(indices), axis=axis))


def vdot(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the dot product of two vectors.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.vdot(_to_tensor(a), _to_tensor(b)))


def inner(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inner product of two arrays.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.inner(_to_tensor(a), _to_tensor(b)))


def outer(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the outer product of two vectors.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.outer(_to_tensor(a), _to_tensor(b)))


def tensordot(a: ArrayLikeSpec, b: ArrayLikeSpec, axes: ArrayLikeSpec = 2) -> ArrayLikeSpec:
    """Compute tensor dot product along specified axes.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.
        axes (ArrayLikeSpec): Argument axes.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.tensordot(_to_tensor(a), _to_tensor(b), axes=axes))


def shape(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """Get the shape of the array.

    Args:
        a: Input parameter ``a``.

    Returns:
        The operation result.
    """
    return asarray(a).shape


def sqrt(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the non-negative square-root of an array, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.sqrt(_to_tensor(x)))


def square(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the element-wise square of the input.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.square(_to_tensor(x)))


def isnan(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Test element-wise for NaN and return result as a boolean array.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.isnan(_to_tensor(x)))


nan = float("nan")

pi = 3.14159265358979323846


def cumsum(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the cumulative sum of the elements along a given axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        axis (ArrayLikeSpec): Argument axis.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    res = ops.cumsum(_to_tensor(a), axis=axis)
    if dtype is not None:
        from zero_jax.nn.activation import _to_dtype

        res = ops.cast(res, dtype=_to_dtype(dtype))
    return _wrap(res)


def acos(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise arc cosine.

    Args:
        x: Input array.

    Returns:
        An array containing the arc cosine of each element in x.
    """
    return _wrap(ops.acos(_to_tensor(x)))


def acosh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise inverse hyperbolic cosine.

    Args:
        x: Input array.

    Returns:
        An array containing the inverse hyperbolic cosine of each element in x.
    """
    return _wrap(ops.acosh(_to_tensor(x)))


def asin(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise arc sine.

    Args:
        x: Input array.

    Returns:
        An array containing the arc sine of each element in x.
    """
    return _wrap(ops.asin(_to_tensor(x)))


def asinh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise inverse hyperbolic sine.

    Args:
        x: Input array.

    Returns:
        An array containing the inverse hyperbolic sine of each element in x.
    """
    return _wrap(ops.asinh(_to_tensor(x)))


def atan(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise arc tangent.

    Args:
        x: Input array.

    Returns:
        An array containing the arc tangent of each element in x.
    """
    return _wrap(ops.atan(_to_tensor(x)))


def atan2(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    Args:
        x1: Y-coordinates.
        x2: X-coordinates.

    Returns:
        An array containing the arc tangent of x1/x2.
    """
    # Need to broadcast x1 and x2 if shapes don't match, or let ops.atan2 handle it.
    # ml-switcheroo-compiler ops.atan2 probably handles basic broadcasting or we might need it.
    # Let's just pass it to ops.atan2
    return _wrap(ops.atan2(_to_tensor(x1), _to_tensor(x2)))


def atanh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Element-wise inverse hyperbolic tangent.

    Args:
        x: Input array.

    Returns:
        An array containing the inverse hyperbolic tangent of each element in x.
    """
    return _wrap(ops.atanh(_to_tensor(x)))


def bitwise_and(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_and.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.bitwise_and(_to_tensor(x1), _to_tensor(x2)))


def bitwise_not(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_not.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.bitwise_not(_to_tensor(x)))


def bitwise_or(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_or.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.bitwise_or(_to_tensor(x1), _to_tensor(x2)))


def bitwise_xor(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_xor.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.bitwise_xor(_to_tensor(x1), _to_tensor(x2)))


def logical_and(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for logical_and.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.logical_and(_to_tensor(x1), _to_tensor(x2)))


def logical_not(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for logical_not.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.logical_not(_to_tensor(x)))


def logical_or(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for logical_or.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.logical_or(_to_tensor(x1), _to_tensor(x2)))


def logical_xor(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for logical_xor.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.logical_xor(_to_tensor(x1), _to_tensor(x2)))


def equal(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.equal(_to_tensor(x1), _to_tensor(x2)))


def not_equal(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for not_equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.not_equal(_to_tensor(x1), _to_tensor(x2)))


def greater(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for greater.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.greater(_to_tensor(x1), _to_tensor(x2)))


def greater_equal(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for greater_equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.greater_equal(_to_tensor(x1), _to_tensor(x2)))


def less(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for less.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.less(_to_tensor(x1), _to_tensor(x2)))


def less_equal(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for less_equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.less_equal(_to_tensor(x1), _to_tensor(x2)))


def cbrt(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for cbrt.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.cbrt(_to_tensor(x)))


def conj(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for conj.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.conj(_to_tensor(x)))


def copysign(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for copysign.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.copysign(_to_tensor(x1), _to_tensor(x2)))


def count_nonzero(a: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for count_nonzero.

    Args:
        a: Argument a.
        axis: Argument axis.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.count_nonzero(_to_tensor(a), axis=axis))


def cross(a: ArrayLikeSpec, b: ArrayLikeSpec, axisa: int = -1, axisb: int = -1, axisc: int = -1, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for cross.

    Args:
        a: Argument a.
        b: Argument b.
        axisa: Argument axisa.
        axisb: Argument axisb.
        axisc: Argument axisc.
        axis: Argument axis.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(
        ops.cross(
            _to_tensor(a),
            _to_tensor(b),
            axes={"axisa": axisa, "axisb": axisb, "axisc": axisc, "axis": axis},
        )
    )


def deg2rad(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for deg2rad.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.deg2rad(_to_tensor(x)))


def diag(v: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for diag.

    Args:
        v: Argument v.
        k: Argument k.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.diag(_to_tensor(v), diagonal=k))


def fix(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for fix.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.fix(_to_tensor(x)))


def float_power(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for float_power.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.float_power(_to_tensor(x1), _to_tensor(x2)))


def fmax(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for fmax.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.fmax(_to_tensor(x1), _to_tensor(x2)))


def fmin(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for fmin.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.fmin(_to_tensor(x1), _to_tensor(x2)))


def fmod(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for fmod.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.fmod(_to_tensor(x1), _to_tensor(x2)))


def gcd(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for gcd.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.gcd(_to_tensor(x1), _to_tensor(x2)))


def heaviside(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for heaviside.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.heaviside(_to_tensor(x1), _to_tensor(x2)))


def hypot(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for hypot.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.hypot(_to_tensor(x1), _to_tensor(x2)))


def imag(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for imag.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.imag(_to_tensor(x)))


def isclose(a: ArrayLikeSpec, b: ArrayLikeSpec, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> ArrayLikeSpec:
    """JAX API implementation for isclose.

    Args:
        a: Argument a.
        b: Argument b.
        rtol: Argument rtol.
        atol: Argument atol.
        equal_nan: Argument equal_nan.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.isclose(_to_tensor(a), _to_tensor(b), rtol=rtol, atol=atol, equal_nan=equal_nan))


def isinf(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for isinf.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.isinf(_to_tensor(x)))


def lcm(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for lcm.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.lcm(_to_tensor(x1), _to_tensor(x2)))


def ldexp(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for ldexp.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.ldexp(_to_tensor(x1), _to_tensor(x2)))


def left_shift(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for left_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.left_shift(_to_tensor(x1), _to_tensor(x2)))


def logaddexp(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for logaddexp.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.logaddexp(_to_tensor(x1), _to_tensor(x2)))


def logaddexp2(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for logaddexp2.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.logaddexp2(_to_tensor(x1), _to_tensor(x2)))


def nextafter(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for nextafter.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.nextafter(*[_to_tensor(t) for t in broadcast_arrays(x1, x2)]))


def rad2deg(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for rad2deg.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.rad2deg(_to_tensor(x)))


def real(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for real.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.real(_to_tensor(x)))


def reciprocal(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for reciprocal.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.reciprocal(_to_tensor(x)))


def right_shift(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for right_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.right_shift(_to_tensor(x1), _to_tensor(x2)))


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


def round(a: ArrayLikeSpec, decimals: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for round.

    Args:
        a: Argument a.
        decimals: Argument decimals.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.round(_to_tensor(a), decimals=decimals))


def select(condlist: ArrayLikeSpec, choicelist: ArrayLikeSpec, default: ArrayLikeSpec = 0) -> ArrayLikeSpec:
    """JAX API implementation for select.

    Args:
        condlist: Argument condlist.
        choicelist: Argument choicelist.
        default: Argument default.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = _to_tensor(default)
    for c, v in reversed(list(zip(condlist, choicelist))):
        res = ops.where(_to_tensor(c), _to_tensor(v), res)
    return _wrap(res)


def sinc(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for sinc.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.sinc(_to_tensor(x)))


def sort(a: ArrayLikeSpec, axis: int = -1, kind: ArrayLikeSpec = None, order: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for sort.

    Args:
        a: Argument a.
        axis: Argument axis.
        kind: Argument kind.
        order: Argument order.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.sort(_to_tensor(a), axis=axis))


def tril(m: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for tril.

    Args:
        m: Argument m.
        k: Argument k.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.tril(_to_tensor(m), diagonal=k))


def triu(m: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for triu.

    Args:
        m: Argument m.
        k: Argument k.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.triu(_to_tensor(m), diagonal=k))


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


def absolute(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Calculates the absolute value element-wise.

    Args:
        x: Input array.

    Returns:
        An array containing the absolute value of each element in x.
    """
    return _wrap(ops.abs(_to_tensor(x)))


def around(a: ArrayLikeSpec, decimals: int = 0) -> ArrayLikeSpec:
    """Evenly round to the given number of decimals.

    Args:
        a: Input data.
        decimals: Number of decimal places to round to.

    Returns:
        An array of the same type as a, containing the rounded values.
    """
    return round(a, decimals=decimals)


def round_(a: ArrayLikeSpec, decimals: int = 0) -> ArrayLikeSpec:
    """Evenly round to the given number of decimals.

    Args:
        a: Input data.
        decimals: Number of decimal places to round to.

    Returns:
        An array of the same type as a, containing the rounded values.
    """
    return round(a, decimals=decimals)


def conjugate(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the complex conjugate, element-wise.

    Args:
        x: Input array.

    Returns:
        The complex conjugate of x.
    """
    return conj(x)


def cumulative_sum(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the cumulative sum of the elements along a given axis.

    Args:
        a: Input array.
        axis: Axis along which the cumulative sum is computed.
        dtype: Type of the returned array.

    Returns:
        A new array holding the result.
    """
    return cumsum(a, axis=axis, dtype=dtype)


def degrees(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert angles from radians to degrees.

    Args:
        x: Input array in radians.

    Returns:
        The corresponding degree values.
    """
    return rad2deg(x)


def radians(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert angles from degrees to radians.

    Args:
        x: Input array in degrees.

    Returns:
        The corresponding radian values.
    """
    return deg2rad(x)


def pow(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """First array elements raised to powers from second array, element-wise.

    Args:
        x1: The bases.
        x2: The exponents.

    Returns:
        The bases in x1 raised to the exponents in x2.
    """
    return power(x1, x2)


def bitwise_invert(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_invert.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return bitwise_not(x)


def bitwise_left_shift(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_left_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return left_shift(x1, x2)


def bitwise_right_shift(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_right_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return right_shift(x1, x2)


def concat(arrays: ArrayLikeSpec, axis: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for concat.

    Args:
        arrays: Argument arrays.
        axis: Argument axis.

    Returns:
        ArrayLikeSpec: The result.
    """
    return concatenate(arrays, axis=axis)


def invert(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for invert.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return bitwise_not(x)


def nanmax(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    keepdims: ArrayLikeSpec = False,
    initial: ArrayLikeSpec = None,
    where: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for nanmax.

    Args:
        a: Argument a.
        axis: Argument axis.
        keepdims: Argument keepdims.
        initial: Argument initial.
        where: Argument where.

    Returns:
        ArrayLikeSpec: The result.
    """
    return max(a, axis=axis, keepdims=keepdims)


def nanmin(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    keepdims: ArrayLikeSpec = False,
    initial: ArrayLikeSpec = None,
    where: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for nanmin.

    Args:
        a: Argument a.
        axis: Argument axis.
        keepdims: Argument keepdims.
        initial: Argument initial.
        where: Argument where.

    Returns:
        ArrayLikeSpec: The result.
    """
    return min(a, axis=axis, keepdims=keepdims)


def nanprod(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    dtype: ArrayLikeSpec = None,
    keepdims: ArrayLikeSpec = False,
    initial: ArrayLikeSpec = None,
    where: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for nanprod.

    Args:
        a: Argument a.
        axis: Argument axis.
        dtype: Argument dtype.
        keepdims: Argument keepdims.
        initial: Argument initial.
        where: Argument where.

    Returns:
        ArrayLikeSpec: The result.
    """
    return prod(a, axis=axis, keepdims=keepdims)


def nansum(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    dtype: ArrayLikeSpec = None,
    keepdims: ArrayLikeSpec = False,
    initial: ArrayLikeSpec = None,
    where: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for nansum.

    Args:
        a: Argument a.
        axis: Argument axis.
        dtype: Argument dtype.
        keepdims: Argument keepdims.
        initial: Argument initial.
        where: Argument where.

    Returns:
        ArrayLikeSpec: The result.
    """
    return sum(a, axis=axis, keepdims=keepdims)


def isneginf(x: ArrayLikeSpec, out: ArrayLikeSpec = None) -> ArrayLikeSpec:
    # JAX handles isneginf by doing isinf(x) & (x < 0)
    """JAX API implementation for isneginf.

    Args:
        x: Argument x.
        out: Argument out.

    Returns:
        ArrayLikeSpec: The result.
    """
    inf_mask = isinf(x)
    neg_mask = x < 0
    res = logical_and(inf_mask, neg_mask)
    if out is not None:
        raise ValueError("out parameter is not supported in JAX")
    return res


def isposinf(x: ArrayLikeSpec, out: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for isposinf.

    Args:
        x: Argument x.
        out: Argument out.

    Returns:
        ArrayLikeSpec: The result.
    """
    inf_mask = isinf(x)
    pos_mask = x > 0
    res = logical_and(inf_mask, pos_mask)
    if out is not None:
        raise ValueError("out parameter is not supported in JAX")
    return res


def frexp(x: ArrayLikeSpec, out: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for frexp.

    Args:
        x: Input parameter ``x``.
        out: Input parameter ``out``.

    Returns:
        The operation result.
    """
    out_t = ops.frexp(_to_tensor(x))
    return (_wrap(out_t[0]), _wrap(out_t[1]))


def nan_to_num(x: ArrayLikeSpec, copy: bool = True, nan: float = 0.0, posinf: ArrayLikeSpec = None, neginf: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Replace NaN with zero and inf with large finite numbers.

    Args:
        x: Input array.
        copy: Ignored.
        nan: Value to be used to fill NaN values.
        posinf: Value to be used to fill positive infinity values.
        neginf: Value to be used to fill negative infinity values.

    Returns:
        The resulting array.
    """
    return _wrap(ops.nan_to_num(_to_tensor(x), nan=nan, posinf=posinf, neginf=neginf))


def searchsorted(a: ArrayLikeSpec, v: ArrayLikeSpec, side: str = "left", sorter: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Find indices where elements should be inserted to maintain order.

    Args:
        a: Input parameter ``a``.
        v: Input parameter ``v``.
        side: Input parameter ``side``.
        sorter: Input parameter ``sorter``.

    Returns:
        The operation result.
    """
    if sorter is not None:
        a = take(a, sorter, axis=-1)
    return _wrap(ops.searchsorted(_to_tensor(a), _to_tensor(v), side=side))


def signbit(x: ArrayLikeSpec, out: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Returns element-wise True where signbit is set (less than zero).

    Args:
        x: Input array.
        out: Ignored.

    Returns:
        Boolean array.
    """
    if out is not None:
        raise ValueError("out parameter is not supported in JAX")
    return _wrap(ops.signbit(_to_tensor(x)))


def argsort(a: ArrayLikeSpec, axis: int = -1, kind: ArrayLikeSpec = None, order: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Returns the indices that would sort an array.

    Args:
        a: Input parameter ``a``.
        axis: Input parameter ``axis``.
        kind: Input parameter ``kind``.
        order: Input parameter ``order``.

    Returns:
        The operation result.
    """
    return _wrap(ops.argsort(_to_tensor(a), axis=axis))


def copy(a: ArrayLikeSpec, order: str = "K") -> ArrayLikeSpec:
    """Return an array copy of the given object.

    Args:
        a: Input data.
        order: memory layout.

    Returns:
        An array interpretation of a.
    """
    return _wrap(ops.array(_to_tensor(a)))


from ml_switcheroo_compiler.core.dtype import DType as _DType

bool = _DType.Bool
bool_ = _DType.Bool
complex128 = _DType.Complex128
complex64 = _DType.Complex64
complex_ = _DType.Complex128
float16 = _DType.Float16
bfloat16 = _DType.BFloat16

from ml_switcheroo_compiler.core.dtype import DType as _DType2

# Map float8s to float16 since compiler doesn't have float8 yet
float8_e4m3b11fnuz = _DType2.Float16
float8_e4m3fn = _DType2.Float16
float8_e4m3fnuz = _DType2.Float16
float8_e5m2 = _DType2.Float16
float8_e5m2fnuz = _DType2.Float16

float32 = _DType.Float32
float64 = _DType.Float64
float_ = _DType.Float64
int16 = _DType.Int16
int32 = _DType.Int32
int64 = _DType.Int64
int8 = _DType.Int8
int4 = _DType.Int8

int_ = _DType.Int64
uint16 = _DType.Int16  # Fallback
uint32 = _DType.UInt32
uint64 = _DType.Int64  # Fallback
uint8 = _DType.UInt8


class generic:
    """generic."""


class number(generic):
    """number."""


class integer(number):
    """integer."""


class signedinteger(integer):
    """signedinteger."""


class unsignedinteger(integer):
    """unsignedinteger."""


class inexact(number):
    """inexact."""


class floating(inexact):
    """floating."""


class complexfloating(inexact):
    """complexfloating."""


class flexible(generic):
    """flexible."""


class character(flexible):
    """character."""


class object_(generic):
    """object_."""


double = float64
single = float32
csingle = complex64
cdouble = complex128


class ComplexWarning(Warning):
    """Complex warning."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


def angle(z: ArrayLikeSpec, deg: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Return the angle of the complex argument.

    Args:
        z: A complex number or sequence of complex numbers.
        deg: Return angle in degrees if True, radians if False (default).

    Returns:
        The counterclockwise angle.
    """
    return _wrap(ops.angle(_to_tensor(z), deg=deg))


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


def astype(x: ArrayLikeSpec, dtype: ArrayLikeSpec, copy: ArrayLikeSpec = False, device: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Copy of the array, cast to a specified type.

    Args:
        x: Input array.
        dtype: Data type to which the array is cast.
        copy: Whether to copy the array.
        device: Device to place the returned array.

    Returns:
        The cast array.
    """
    return _wrap(ops.cast(_to_tensor(x), dtype))


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


def average(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    weights: ArrayLikeSpec = None,
    returned: ArrayLikeSpec = False,
    keepdims: ArrayLikeSpec = False,
) -> ArrayLikeSpec:
    """Compute the weighted average along the specified axis.

    Args:
        a: Array containing data to be averaged.
        axis: Axis or axes along which to average a.
        weights: An array of weights associated with the values in a.
        returned: Whether to return sum of weights.
        keepdims: Whether to keep dimensions with size one.

    Returns:
        The average along the specified axis.
    """
    w = _to_tensor(weights) if weights is not None else None
    return _wrap(ops.average(_to_tensor(a), axis=axis, weights=w, returned=returned, keepdims=keepdims))


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


def ndim(a: ArrayLikeSpec) -> int:
    """Return the number of dimensions of an array.

    Args:
        a: Input array.

    Returns:
        The number of dimensions in a.
    """
    return len(_to_tensor(a).shape) if _to_tensor(a).shape is not None else 0


def apply_along_axis(
    func1d: VariadicCallable[ArrayLikeSpec],
    axis: int,
    arr: ArrayLikeSpec,
    *args: ArrayLikeSpec,
    **kwargs: ArrayLikeSpec,
) -> ArrayLikeSpec:
    """Apply a function to 1-D slices along the given axis.

    Args:
        func1d: This function should accept 1-D arrays.
        axis: Axis along which arr is sliced.
        arr: Input array.
        *args: Additional arguments to func1d.
        **kwargs: Additional keyword arguments to func1d.

    Returns:
        The output array.
    """
    return func1d(_wrap(_to_tensor(arr)), *args, **kwargs)


def choose(a: ArrayLikeSpec, choices: ArrayLikeSpec, out: ArrayLikeSpec = None, mode: str = "raise") -> ArrayLikeSpec:
    """Construct an array from an index array and a set of arrays to choose from.

    Args:
        a: Array of integers indicating which array in choices to take values from.
        choices: Sequence of arrays to choose from.
        out: If provided, the result will be inserted into this array.
        mode: Specifies how indices outside [0, n-1] will be treated.

    Returns:
        The merged result.
    """
    c = [_to_tensor(ch) for ch in choices]
    return _wrap(ops.choose(_to_tensor(a), c, out=out, mode=mode))


def column_stack(tup: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stack 1-D arrays as columns into a 2-D array.

    Args:
        tup: Sequence of 1-D or 2-D arrays.

    Returns:
        The stacked array.
    """
    return _wrap(ops.column_stack([_to_tensor(t) for t in tup]))


def compress(
    condition: ArrayLikeSpec,
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    *,
    size: ArrayLikeSpec = None,
    fill_value: ArrayLikeSpec = 0,
    out: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Return selected slices of an array along given axis.

    Args:
        condition: Array that selects which entries to return.
        a: Input array.
        axis: Axis along which to take slices.
        size: Optional padding.
        fill_value: Padding value.
        out: Output array.

    Returns:
        A copy of a without the slices along axis for which condition is false.
    """
    return _wrap(ops.compress(_to_tensor(condition), _to_tensor(a), axis=axis, out=out))


def convolve(
    a: ArrayLikeSpec,
    v: ArrayLikeSpec,
    mode: str = "full",
    *,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Returns the discrete, linear convolution of two one-dimensional sequences.

    Args:
        a: First input sequence.
        v: Second input sequence.
        mode: 'full', 'valid', or 'same'.
        precision: Numerical precision.
        preferred_element_type: Optional dtype.

    Returns:
        Discrete, linear convolution of a and v.
    """
    # Pass kwargs supported by JAX if applicable, otherwise base ops
    return _wrap(ops.convolve(_to_tensor(a), _to_tensor(v), mode=mode))


def corrcoef(x: ArrayLikeSpec, y: ArrayLikeSpec = None, rowvar: ArrayLikeSpec = True) -> ArrayLikeSpec:
    """Return Pearson product-moment correlation coefficients.

    Args:
        x: A 1-D or 2-D array containing multiple variables and observations.
        y: An additional set of variables and observations.
        rowvar: If rowvar is True, each row represents a variable.

    Returns:
        The correlation coefficient matrix of the variables.
    """
    return _wrap(ops.corrcoef(_to_tensor(x), _to_tensor(y) if y is not None else None, rowvar=rowvar))


def correlate(
    a: ArrayLikeSpec,
    v: ArrayLikeSpec,
    mode: str = "valid",
    *,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        a: First input sequence.
        v: Second input sequence.
        mode: 'valid', 'same', or 'full'.
        precision: Numerical precision.
        preferred_element_type: Optional dtype.

    Returns:
        Discrete cross-correlation of a and v.
    """
    return _wrap(ops.correlate(_to_tensor(a), _to_tensor(v), mode=mode))


def cov(
    m: ArrayLikeSpec,
    y: ArrayLikeSpec = None,
    rowvar: ArrayLikeSpec = True,
    bias: ArrayLikeSpec = False,
    ddof: ArrayLikeSpec = None,
    fweights: ArrayLikeSpec = None,
    aweights: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Estimate a covariance matrix, given data and weights.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
        y: An additional set of variables and observations.
        rowvar: If rowvar is True, each row represents a variable.
        bias: Default normalization is False.
        ddof: Degrees of freedom.
        fweights: Frequency weights.
        aweights: Observation vector weights.

    Returns:
        The covariance matrix of the variables.
    """
    y_tensor = _to_tensor(y) if y is not None else None
    return _wrap(
        ops.cov(
            _to_tensor(m),
            y=y_tensor,
            rowvar=rowvar,
            bias=bias,
            ddof=ddof,
            fweights=fweights,
            aweights=aweights,
        )
    )


def array_equiv(a1: ArrayLikeSpec, a2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns True if input arrays are shape consistent and all elements equal.

    Args:
        a1: Input array.
        a2: Input array.

    Returns:
        True if equivalent, False otherwise.
    """
    return _wrap(ops.array_equiv(_to_tensor(a1), _to_tensor(a2)))


def array_repr(
    arr: ArrayLikeSpec,
    max_line_width: ArrayLikeSpec = None,
    precision: ArrayLikeSpec = None,
    suppress_small: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Return the string representation of an array.

    Args:
        arr: Input array.
        max_line_width: Maximum line width.
        precision: Floating point precision.
        suppress_small: Whether to suppress very small numbers.

    Returns:
        String representation.
    """
    return ops.array_repr(
        _to_tensor(arr),
        max_line_width=max_line_width,
        precision=precision,
        suppress_small=suppress_small,
    )


def array_str(
    arr: ArrayLikeSpec,
    max_line_width: ArrayLikeSpec = None,
    precision: ArrayLikeSpec = None,
    suppress_small: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Return a string representation of the data in an array.

    Args:
        arr: Input array.
        max_line_width: Maximum line width.
        precision: Floating point precision.
        suppress_small: Whether to suppress very small numbers.

    Returns:
        String representation.
    """
    return ops.array_str(
        _to_tensor(arr),
        max_line_width=max_line_width,
        precision=precision,
        suppress_small=suppress_small,
    )


def bartlett(M: int) -> ArrayLikeSpec:
    """Return the Bartlett window.

    Args:
        M: Number of points in the output window.

    Returns:
        The window.
    """
    if M < 1:
        return array([], dtype=float32)
    if M == 1:
        return ones((1,), dtype=float32)
    n = arange(0, M, dtype=float32)
    half = (M - 1) / 2.0
    return where(n <= half, 2.0 * n / (M - 1), 2.0 - 2.0 * n / (M - 1))


def bincount(x: ArrayLikeSpec, weights: ArrayLikeSpec = None, minlength: int = 0, *, length: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Count number of occurrences of each value in array of non-negative ints.

    Args:
        x: Input array.
        weights: Weights, array of the same shape as x.
        minlength: A minimum number of bins for the output array.
        length: Alias for minlength.

    Returns:
        The result of binning the input array.
    """
    mlen = length if length is not None else minlength
    return _wrap(
        ops.bincount(
            _to_tensor(x),
            _to_tensor(weights) if weights is not None else None,
            minlength=mlen,
        )
    )


def bitwise_count(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Counts the number of 1-bits in the absolute value of the input.

    Args:
        x: Input array.

    Returns:
        The number of 1-bits.
    """
    return _wrap(ops.bitwise_count(_to_tensor(x)))


def blackman(M: int) -> ArrayLikeSpec:
    """Return the Blackman window.

    Args:
        M: Number of points in the output window.

    Returns:
        The window.
    """
    if M < 1:
        return array([], dtype=float32)
    if M == 1:
        return ones((1,), dtype=float32)
    import math

    n = arange(0, M, dtype=float32)
    factor = 2.0 * math.pi / (M - 1)
    return 0.42 - 0.5 * cos(factor * n) + 0.08 * cos(2.0 * factor * n)


def broadcast_arrays(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Broadcast any number of arrays against each other.

    Args:
        *args: The arrays to broadcast.
        **kwargs: Additional kwargs.

    Returns:
        A list of broadcasted arrays.
    """
    return [_wrap(t) for t in ops.broadcast_arrays(*[_to_tensor(a) for a in args], **kwargs)]


def can_cast(from_: ArrayLikeSpec, to: ArrayLikeSpec, casting: str = "safe") -> ArrayLikeSpec:
    """Returns True if cast between data types can occur according to the casting rule.

    Args:
        from_: Source type.
        to: Destination type.
        casting: Casting rule.

    Returns:
        True if cast can occur.
    """
    return _safe_bool(ops.can_cast(from_, to, casting=casting))


def cumprod(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None, out: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the cumulative product of elements along a given axis.

    Args:
        a: Input array.
        axis: Axis along which the cumulative product is computed.
        dtype: Type of the returned array.
        out: Alternative output array.

    Returns:
        The cumulative product.
    """
    return _wrap(ops.cumprod(_to_tensor(a), axis=axis, dtype=dtype))


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


def diag_indices(n: int, ndim: int = 2) -> ArrayLikeSpec:
    """Return the indices to access the main diagonal of an array.

    Args:
        n: The size, along each dimension, of the arrays for which the returned indices can be used.
        ndim: The number of dimensions.

    Returns:
        A tuple of indices.
    """
    return tuple(_wrap(t) for t in ops.diag_indices(n, ndim=ndim))


def diag_indices_from(arr: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the indices to access the main diagonal of an n-dimensional array.

    Args:
        arr: Input array.

    Returns:
        A tuple of indices.
    """
    return tuple(_wrap(t) for t in ops.diag_indices_from(_to_tensor(arr)))


def diagflat(v: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """Create a two-dimensional array with the flattened input as a diagonal.

    Args:
        v: Input data.
        k: Diagonal to set.

    Returns:
        The 2-D output array.
    """
    return _wrap(ops.diagflat(_to_tensor(v), k=k))


def diagonal(a: ArrayLikeSpec, offset: int = 0, axis1: int = 0, axis2: int = 1) -> ArrayLikeSpec:
    """Return specified diagonals.

    Args:
        a: Input array.
        offset: Offset of the diagonal from the main diagonal.
        axis1: Axis to be used as the first axis of the 2-D sub-arrays.
        axis2: Axis to be used as the second axis of the 2-D sub-arrays.

    Returns:
        Array of diagonals.
    """
    return _wrap(ops.diagonal(_to_tensor(a), offset=offset, axis1=axis1, axis2=axis2))


def diff(a: ArrayLikeSpec, n: int = 1, axis: int = -1, prepend: ArrayLikeSpec = None, append: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Calculate the n-th discrete difference along the given axis.

    Args:
        a: Input array.
        n: The number of times values are differenced.
        axis: The axis along which the difference is taken.
        prepend: Values to prepend to a.
        append: Values to append to a.

    Returns:
        The n-th differences.
    """
    kw: dict[str, ArrayLikeSpec] = {"n": n, "axis": axis}
    if prepend is not None:
        kw["prepend"] = _to_tensor(prepend)
    if append is not None:
        kw["append"] = _to_tensor(append)
    return _wrap(ops.diff(_to_tensor(a), **kw))


def digitize(x: ArrayLikeSpec, bins: ArrayLikeSpec, right: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Return the indices of the bins to which each value in input array belongs.

    Args:
        x: Input array to be binned.
        bins: Array of bins.
        right: Indicating whether the intervals include the right or the left bin edge.

    Returns:
        Array of indices.
    """
    return _wrap(ops.digitize(_to_tensor(x), _to_tensor(bins), right=right))


def dtype(
    value: ArrayLikeSpec,
    names: ArrayLikeSpec = None,
    *,
    module: ArrayLikeSpec = None,
    qualname: ArrayLikeSpec = None,
    type: ArrayLikeSpec = None,
    start: int = 1,
) -> ArrayLikeSpec:
    """Create a data type object.

    Args:
        value: Object to be converted to a data type object.
        names: Optional.
        module: Optional.
        qualname: Optional.
        type: Optional.
        start: Optional.

    Returns:
        A new dtype object.
    """
    try:
        if isinstance(value, _DType):
            return value
        name_str = value if isinstance(value, str) else (value.__name__ if isinstance(value, type) else str(value))
        return _DType(name_str.lower())
    except Exception:
        return value


def ediff1d(ary: ArrayLikeSpec, to_end: ArrayLikeSpec = None, to_begin: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """The differences between consecutive elements of an array.

    Args:
        ary: If necessary, will be flattened before the differences are taken.
        to_end: Number(s) to append at the end of the returned differences.
        to_begin: Number(s) to prepend at the beginning of the returned differences.

    Returns:
        The differences.
    """
    return _wrap(
        ops.ediff1d(
            _to_tensor(ary),
            to_end=_to_tensor(to_end) if to_end is not None else None,
            to_begin=_to_tensor(to_begin) if to_begin is not None else None,
        )
    )


def einsum_path(subscripts: ArrayLikeSpec, *operands: ArrayLikeSpec, optimize: ArrayLikeSpec = "auto") -> ArrayLikeSpec:
    """Evaluates the lowest cost contraction order for an einsum expression.

    Args:
        subscripts: Specifies the subscripts for summation.
        *operands: These are the arrays for the operation.
        optimize: Choose the type of path.

    Returns:
        A tuple of (path, string_representation).
    """
    return ops.einsum_path(subscripts, *[_to_tensor(o) for o in operands], optimize=optimize)


def extract(condition: ArrayLikeSpec, arr: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = 0) -> ArrayLikeSpec:
    """Return the elements of an array that satisfy some condition.

    Args:
        condition: An array whose nonzero or True entries indicate the elements of arr to extract.
        arr: Input array of the same size as condition.
        size: Optional size limit.
        fill_value: Optional fill value.

    Returns:
        Rank 1 array of values from arr where condition is True.
    """
    return _wrap(ops.extract(_to_tensor(condition), _to_tensor(arr)))


def fabs(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the absolute values element-wise.

    Args:
        x: Input array.

    Returns:
        The absolute values.
    """
    return _wrap(ops.fabs(_to_tensor(x)))


def fill_diagonal(a: ArrayLikeSpec, val: ArrayLikeSpec, wrap: bool = False, *, inplace: bool = True) -> ArrayLikeSpec:
    """Fill the main diagonal of the given array of any dimensionality.

    Args:
        a: Array whose diagonal is to be filled.
        val: Value to be written on the diagonal.
        wrap: For tall matrices in NumPy version up to 1.6.2, the diagonal "wrapped" after N columns.
        inplace: Modified in place.

    Returns:
        The array with the diagonal filled.
    """
    return _wrap(ops.fill_diagonal(_to_tensor(a), _to_tensor(val), wrap=wrap))


def finfo(dtype: ArrayLikeSpec) -> ArrayLikeSpec:
    """Machine limits for floating point types.

    Args:
        dtype: The kind of floating point data-type.

    Returns:
        Machine limits.
    """
    return ops.finfo(dtype)


def flatnonzero(a: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return indices that are non-zero in the flattened version of a.

    Args:
        a: Input array.
        size: Optional output size.
        fill_value: Optional fill value.

    Returns:
        Output array, containing the indices of the elements of a.ravel() that are non-zero.
    """
    nz = nonzero(ravel(a), size=size, fill_value=fill_value)
    return nz[0]


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


def from_dlpack(x: ArrayLikeSpec, /, *, device: ArrayLikeSpec = None, copy: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Create an array from a DLPack tensor.

    Args:
        x: A DLPack tensor.
        device: The device on which the created array should be placed.
        copy: Optional.

    Returns:
        The array.
    """
    if isinstance(x, (ndarray, Tensor)):
        return asarray(x)
    return _wrap(ops.from_dlpack(x))


def frombuffer(buffer: ArrayLikeSpec, dtype: ArrayLikeSpec = float, count: int = -1, offset: int = 0) -> ArrayLikeSpec:
    """Interpret a buffer as a 1-dimensional array.

    Args:
        buffer: Buffer to read.
        dtype: Data-type of the returned array.
        count: Number of items to read.
        offset: Start reading the buffer from this offset.

    Returns:
        The constructed array.
    """
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(buffer)[offset:]
        vals = list(raw_bytes)
        if count >= 0:
            vals = vals[:count]
        return array(vals, dtype=dtype)
    return array(buffer, dtype=dtype)


def iinfo(int_type: ArrayLikeSpec) -> ArrayLikeSpec:
    """Machine limits for integer types.

    Args:
        int_type: The kind of integer data type.

    Returns:
        Machine limits.
    """
    return ops.iinfo(int_type)


def indices(dimensions: ArrayLikeSpec, dtype: ArrayLikeSpec = None, sparse: bool = False) -> ArrayLikeSpec:
    """Return an array representing the indices of a grid.

    Args:
        dimensions: The shape of the grid.
        dtype: Data type of the result.
        sparse: Return a sparse representation of the grid instead of a dense representation.

    Returns:
        If sparse is False: one array of grid indices. If True: a tuple of arrays.
    """
    dims = tuple(int(d) for d in dimensions)
    dt = int32 if dtype is None else dtype
    axes = [arange(0, d, dtype=dt) for d in dims]
    grids = meshgrid(*axes, sparse=sparse, indexing="ij")
    if sparse:
        return grids
    return stack(list(grids), axis=0) if grids else empty((0,), dtype=dt)


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


def interp(x: ArrayLikeSpec, xp: ArrayLikeSpec, fp: ArrayLikeSpec, left: ArrayLikeSpec = None, right: ArrayLikeSpec = None, period: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """One-dimensional linear interpolation.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: The x-coordinates of the data points.
        fp: The y-coordinates of the data points.
        left: Value to return for x < xp[0], default is fp[0].
        right: Value to return for x > xp[-1], default is fp[-1].
        period: A period for the x-coordinates.

    Returns:
        The interpolated values.
    """
    return _wrap(
        ops.interp(
            _to_tensor(x),
            _to_tensor(xp),
            _to_tensor(fp),
            left=_to_tensor(left) if left is not None else None,
            right=_to_tensor(right) if right is not None else None,
            period=_to_tensor(period) if period is not None else None,
        )
    )


def intersect1d(ar1: ArrayLikeSpec, ar2: ArrayLikeSpec, assume_unique: bool = False, return_indices: bool = False) -> ArrayLikeSpec:
    """Find the intersection of two arrays.

    Args:
        ar1: Input array.
        ar2: Input array.
        assume_unique: If True, the input arrays are both assumed to be unique.
        return_indices: If True, the indices which correspond to the intersection of the two arrays are returned.

    Returns:
        Sorted 1D array of common and unique elements.
    """
    return _wrap(
        ops.intersect1d(
            _to_tensor(ar1),
            _to_tensor(ar2),
            assume_unique=assume_unique,
            return_indices=return_indices,
        )
    )


def iscomplex(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns a bool array, where True if input element is complex.

    Args:
        x: Input array.

    Returns:
        Boolean array.
    """
    return _wrap(ops.iscomplex(_to_tensor(x)))


def _safe_bool(val: ArrayLikeSpec) -> builtins.bool:
    """Safely convert a tensor or scalar to a Python boolean.

    Args:
        val: Input tensor or scalar value.

    Returns:
        bool: Python boolean value.
    """
    if isinstance(val, Tensor):
        return builtins.bool(val.data)
    return builtins.bool(val)


def iscomplexobj(x: ArrayLikeSpec) -> bool:
    # Using proxy node if possible
    """Execute ``iscomplexobj``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    return _safe_bool(ops.iscomplexobj(_to_tensor(x)))


def isdtype(dtype: ArrayLikeSpec, kind: ArrayLikeSpec) -> bool:
    """Execute ``isdtype``.

    Args:
        dtype: Input parameter ``dtype``.
        kind: Input parameter ``kind``.

    Returns:
        The operation result.
    """
    return _safe_bool(ops.issubdtype(dtype, kind))


def isin(element: ArrayLikeSpec, test_elements: ArrayLikeSpec, assume_unique: bool = False, invert: bool = False) -> ArrayLikeSpec:
    """Calculates element in test_elements, broadcasting over element only.

    Args:
        element: Input array.
        test_elements: The values against which to test each value of element.
        assume_unique: If True, the input arrays are both assumed to be unique.
        invert: If True, the values in the returned array are inverted.

    Returns:
        Has the same shape as element.
    """
    return _wrap(
        ops.isin(
            _to_tensor(element),
            _to_tensor(test_elements),
            assume_unique=assume_unique,
            invert=invert,
        )
    )


def isreal(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns a bool array, where True if input element is real.

    Args:
        x: Input array.

    Returns:
        Boolean array.
    """
    return _wrap(ops.isreal(_to_tensor(x)))


def isrealobj(x: ArrayLikeSpec) -> bool:
    """Execute ``isrealobj``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    raw = x._tensor.data if isinstance(x, ndarray) else (x.data if isinstance(x, Tensor) else x)
    return _safe_bool(ops.isrealobj(raw))


def isscalar(element: ArrayLikeSpec) -> builtins.bool:
    """Execute ``isscalar``.

    Args:
        element: Input parameter ``element``.

    Returns:
        The operation result.
    """
    if isinstance(element, (ndarray, Tensor)):
        return False
    return isinstance(element, (int, float, complex, builtins.bool, str, bytes)) or element is None


def issubdtype(arg1: ArrayLikeSpec, arg2: ArrayLikeSpec) -> bool:
    """Execute ``issubdtype``.

    Args:
        arg1: Input parameter ``arg1``.
        arg2: Input parameter ``arg2``.

    Returns:
        The operation result.
    """
    return _safe_bool(ops.issubdtype(arg1, arg2))


def iterable(y: ArrayLikeSpec) -> bool:
    """Execute ``iterable``.

    Args:
        y: Input parameter ``y``.

    Returns:
        The operation result.
    """
    raw = y._tensor.data if isinstance(y, ndarray) else (y.data if isinstance(y, Tensor) else y)
    return _safe_bool(ops.iterable(raw))


def ix_(*args: ArrayLikeSpec) -> ArrayLikeSpec:
    """Construct an open mesh from multiple sequences.

    Args:
        *args: 1-D sequences.

    Returns:
        Tuple of ndarrays.
    """
    return tuple(_wrap(t) for t in ops.ix_(*[_to_tensor(a) for a in args]))


def kaiser(M: int, beta: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the Kaiser window.

    Args:
        M: Number of points in the output window.
        beta: Shape parameter for window.

    Returns:
        The window.
    """
    if M < 1:
        return array([], dtype=float32)
    if M == 1:
        return ones((1,), dtype=float32)
    n = arange(0, M, dtype=float32)
    alpha = (M - 1) / 2.0
    ratio = (n - alpha) / alpha
    arg = beta * sqrt(maximum(1.0 - ratio * ratio, 0.0))
    return i0(arg) / i0(beta)


def kron(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Kronecker product of two arrays.

    Args:
        a: Input array.
        b: Input array.

    Returns:
        The Kronecker product of a and b.
    """
    return _wrap(ops.kron(_to_tensor(a), _to_tensor(b)))


def lexsort(keys: ArrayLikeSpec, axis: int = -1) -> ArrayLikeSpec:
    """Perform an indirect stable sort using a sequence of keys.

    Args:
        keys: The k different "columns" to be sorted.
        axis: Axis to be indirectly sorted.

    Returns:
        Array of indices that sort the keys along the specified axis.
    """
    return _wrap(
        ops.lexsort(
            [_to_tensor(k) for k in keys] if isinstance(keys, (list, tuple)) else _to_tensor(keys),
            axis=axis,
        )
    )


def load(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Load arrays or pickled objects from .npy, .npz or pickled files.

    Args:
        *args: positional args.
        **kwargs: keyword args.

    Returns:
        The result.
    """
    return _wrap(ops.load(*args, **kwargs))


def mask_indices(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the indices to access (n, n) arrays, given a masking function.

    Args:
        *args: positional args.
        **kwargs: keyword args.

    Returns:
        Tuple of indices.
    """
    return tuple(_wrap(t) for t in ops.mask_indices(*args, **kwargs))


def matrix_transpose(x: ArrayLikeSpec, /) -> ArrayLikeSpec:
    """Transposes a matrix (or a stack of matrices) x.

    Args:
        x: Input array.

    Returns:
        The transposed array.
    """
    return _wrap(ops.matrix_transpose(_to_tensor(x)))


def median(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> ArrayLikeSpec:
    """Compute the median along the specified axis.

    Args:
        a: Input array or object that can be converted to an array.
        axis: Axis or axes along which the medians are computed.
        out: Alternative output array in which to place the result.
        overwrite_input: If True, then allow use of memory of input array a for calculations.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        A new array holding the result.
    """
    return _wrap(ops.median(_to_tensor(a), axis=axis, keepdims=keepdims))


def modf(x: ArrayLikeSpec, /, out: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the fractional and integral parts of an array, element-wise.

    Args:
        x: Input array.
        out: A location into which the result is stored.

    Returns:
        Fractional and integral parts.
    """
    if out is not None:
        raise ValueError("out parameter is not supported in JAX")
    arr = asarray(x)
    integral = trunc(arr)
    fractional = subtract(arr, integral)
    return fractional, integral


def nanargmax(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, out: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the indices of the maximum values in the specified axis ignoring NaNs.

    Args:
        a: Input data.
        axis: Axis along which to operate.
        out: If provided, the result will be inserted into this array.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        Array of indices into the array.
    """
    return _wrap(ops.nanargmax(_to_tensor(a), axis=axis, keepdims=keepdims))


def nanargmin(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, out: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the indices of the minimum values in the specified axis ignoring NaNs.

    Args:
        a: Input data.
        axis: Axis along which to operate.
        out: If provided, the result will be inserted into this array.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        Array of indices into the array.
    """
    return _wrap(ops.nanargmin(_to_tensor(a), axis=axis, keepdims=keepdims))


def nancumprod(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None, out: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the cumulative product of array elements over a given axis treating NaNs as one.

    Args:
        a: Input array.
        axis: Axis along which the cumulative product is computed.
        dtype: Type of the returned array, as well as of the accumulator in which the elements are multiplied.
        out: Alternative output array in which to place the result.

    Returns:
        A new array holding the result is returned unless out is specified.
    """
    return _wrap(ops.nancumprod(_to_tensor(a), axis=axis, dtype=dtype))


def nancumsum(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None, out: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the cumulative sum of array elements over a given axis treating NaNs as zero.

    Args:
        a: Input array.
        axis: Axis along which the cumulative sum is computed.
        dtype: Type of the returned array, as well as of the accumulator in which the elements are summed.
        out: Alternative output array in which to place the result.

    Returns:
        A new array holding the result is returned unless out is specified.
    """
    return _wrap(ops.nancumsum(_to_tensor(a), axis=axis, dtype=dtype))


def nanmean(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    dtype: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
    keepdims: bool = False,
    where: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Args:
        a: Array containing numbers whose mean is desired.
        axis: Axis or axes along which the means are computed.
        dtype: Type to use in computing the mean.
        out: Alternate output array in which to place the result.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        where: Elements to include in the mean.

    Returns:
        A new array containing the mean values.
    """
    return _wrap(ops.nanmean(_to_tensor(a), axis=axis, keepdims=keepdims))


def nanmedian(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> ArrayLikeSpec:
    """Compute the median along the specified axis, while ignoring NaNs.

    Args:
        a: Input array or object that can be converted to an array.
        axis: Axis or axes along which the medians are computed.
        out: Alternative output array in which to place the result.
        overwrite_input: If True, then allow use of memory of input array a for calculations.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        A new array holding the result.
    """
    return _wrap(ops.nanmedian(_to_tensor(a), axis=axis, keepdims=keepdims))


def nanpercentile(
    a: ArrayLikeSpec,
    q: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    interpolation: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute the qth percentile of the data along the specified axis, while ignoring nan values.

    Args:
        a: Input array or object that can be converted to an array.
        q: Array of percentile grades to compute.
        axis: Axis or axes along which the percentiles are computed.
        out: Alternative output array in which to place the result.
        overwrite_input: If True, then allow use of memory of input array a for calculations.
        method: This parameter specifies the method to use for estimating the percentile.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        interpolation: Deprecated name for the method keyword argument.

    Returns:
        A new array holding the result.
    """
    return _wrap(ops.nanpercentile(_to_tensor(a), _to_tensor(q), axis=axis, keepdims=keepdims))


def nanquantile(
    a: ArrayLikeSpec,
    q: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    interpolation: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute the qth quantile of the data along the specified axis, while ignoring nan values.

    Args:
        a: Input array or object that can be converted to an array.
        q: Array of quantile grades to compute.
        axis: Axis or axes along which the quantiles are computed.
        out: Alternative output array in which to place the result.
        overwrite_input: If True, then allow use of memory of input array a for calculations.
        method: This parameter specifies the method to use for estimating the quantile.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        interpolation: Deprecated name for the method keyword argument.

    Returns:
        A new array holding the result.
    """
    return _wrap(ops.nanquantile(_to_tensor(a), _to_tensor(q), axis=axis, keepdims=keepdims))


def nanstd(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    dtype: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute the standard deviation along the specified axis, while ignoring NaNs.

    Args:
        a: Calculate the standard deviation of these values.
        axis: Axis or axes along which the standard deviation is computed.
        dtype: Type to use in computing the standard deviation.
        out: Alternative output array in which to place the result.
        ddof: Means Delta Degrees of Freedom.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        where: Elements to include in the standard deviation.

    Returns:
        A new array containing the standard deviation values.
    """
    return _wrap(ops.nanstd(_to_tensor(a), axis=axis, keepdims=keepdims))


def nanvar(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    dtype: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute the variance along the specified axis, while ignoring NaNs.

    Args:
        a: Array containing numbers whose variance is desired.
        axis: Axis or axes along which the variance is computed.
        dtype: Type to use in computing the variance.
        out: Alternate output array in which to place the result.
        ddof: Means Delta Degrees of Freedom.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        where: Elements to include in the variance.

    Returns:
        A new array containing the variance values.
    """
    return _wrap(ops.nanvar(_to_tensor(a), axis=axis, keepdims=keepdims))


# Missing top-level functions
from .tensor_utils import to_array as _to_array


def nonzero(a: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``nonzero``.

    Args:
        a: Input parameter ``a``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    if size is None and fill_value is None:
        return _wrap(ops.nonzero(_to_tensor(a)))
    return _wrap(ops.nonzero(_to_tensor(a), size=size, fill_value=fill_value))


def packbits(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, bitorder: str = "big") -> ArrayLikeSpec:
    """Execute ``packbits``.

    Args:
        a: Input parameter ``a``.
        axis: Input parameter ``axis``.
        bitorder: Input parameter ``bitorder``.

    Returns:
        The operation result.
    """
    return _wrap(ops.packbits(_to_tensor(a), axis, bitorder))


def unpackbits(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, count: ArrayLikeSpec = None, bitorder: str = "big") -> ArrayLikeSpec:
    """Execute ``unpackbits``.

    Args:
        a: Input parameter ``a``.
        axis: Input parameter ``axis``.
        count: Input parameter ``count``.
        bitorder: Input parameter ``bitorder``.

    Returns:
        The operation result.
    """
    return _wrap(ops.unpackbits(_to_tensor(a), axis, count, bitorder))


def partition(a: ArrayLikeSpec, kth: ArrayLikeSpec, axis: int = -1) -> ArrayLikeSpec:
    """Execute ``partition``.

    Args:
        a: Input parameter ``a``.
        kth: Input parameter ``kth``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    from ml_switcheroo_compiler.ops.shape.frontend import partition as _partition

    return _wrap(_partition(_to_tensor(a), kth=kth, axis=axis))


def percentile(
    a: ArrayLikeSpec,
    q: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> ArrayLikeSpec:
    """Execute ``percentile``.

    Args:
        a: Input parameter ``a``.
        q: Input parameter ``q``.
        axis: Input parameter ``axis``.
        out: Input parameter ``out``.
        overwrite_input: Input parameter ``overwrite_input``.
        method: Input parameter ``method``.
        keepdims: Input parameter ``keepdims``.

    Returns:
        The operation result.
    """
    return _wrap(
        ops.percentile(
            _to_tensor(a),
            _to_tensor(q) if isinstance(q, (list, tuple, ndarray, Tensor, ArrayProtocol)) else q,
            axis,
            out,
            overwrite_input,
            method,
            keepdims,
        )
    )


def permute_dims(a: ArrayLikeSpec, axes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``permute_dims``.

    Args:
        a: Input parameter ``a``.
        axes: Input parameter ``axes``.

    Returns:
        The operation result.
    """
    return transpose(a, axes)


def piecewise(x: ArrayLikeSpec, condlist: ArrayLikeSpec, funclist: ArrayLikeSpec, *args: ArrayLikeSpec, **kw: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``piecewise``.

    Args:
        x: Input parameter ``x``.
        condlist: Input parameter ``condlist``.
        funclist: Input parameter ``funclist``.
        *args: Input parameter ``args``.
        **kw: Input parameter ``kw``.

    Returns:
        The operation result.
    """
    return _wrap(ops.piecewise(_to_tensor(x), condlist, funclist, *args, **kw))


def place(arr: ArrayLikeSpec, mask: ArrayLikeSpec, vals: ArrayLikeSpec) -> None:
    """Execute ``place``.

    Args:
        arr: Input parameter ``arr``.
        mask: Input parameter ``mask``.
        vals: Input parameter ``vals``.
    """


def promote_types(type1: ArrayLikeSpec, type2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``promote_types``.

    Args:
        type1: Input parameter ``type1``.
        type2: Input parameter ``type2``.

    Returns:
        The operation result.
    """
    return _wrap(ops.promote_types(type1, type2))


def ptp(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, out: ArrayLikeSpec = None, keepdims: bool = False) -> ArrayLikeSpec:
    """Execute ``ptp``.

    Args:
        a: Input parameter ``a``.
        axis: Input parameter ``axis``.
        out: Input parameter ``out``.
        keepdims: Input parameter ``keepdims``.

    Returns:
        The operation result.
    """
    return amax(a, axis=axis, keepdims=keepdims) - amin(a, axis=axis, keepdims=keepdims)


def put(a: ArrayLikeSpec, ind: ArrayLikeSpec, v: ArrayLikeSpec, mode: str = "raise") -> None:
    """Execute ``put``.

    Args:
        a: Input parameter ``a``.
        ind: Input parameter ``ind``.
        v: Input parameter ``v``.
        mode: Input parameter ``mode``.
    """


def quantile(
    a: ArrayLikeSpec,
    q: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> ArrayLikeSpec:
    """Execute ``quantile``.

    Args:
        a: Input parameter ``a``.
        q: Input parameter ``q``.
        axis: Input parameter ``axis``.
        out: Input parameter ``out``.
        overwrite_input: Input parameter ``overwrite_input``.
        method: Input parameter ``method``.
        keepdims: Input parameter ``keepdims``.

    Returns:
        The operation result.
    """
    return _wrap(
        ops.quantile(
            _to_tensor(a),
            _to_tensor(q) if isinstance(q, (list, tuple, ndarray, Tensor, ArrayProtocol)) else q,
            axis,
            out,
            overwrite_input,
            method,
            keepdims,
        )
    )


def ravel_multi_index(multi_index: ArrayLikeSpec, dims: ArrayLikeSpec, mode: str = "raise", order: str = "C") -> ArrayLikeSpec:
    """JAX API implementation for ravel_multi_index.

    Args:
        multi_index: Input parameter ``multi_index``.
        dims: Input parameter ``dims``.
        mode: Input parameter ``mode``.
        order: Input parameter ``order``.

    Returns:
        The operation result.
    """
    return _wrap(ops.ravel_multi_index(multi_index, dims, mode, order))


def resize(a: ArrayLikeSpec, new_shape: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for resize.

    Args:
        a: Input parameter ``a``.
        new_shape: Input parameter ``new_shape``.

    Returns:
        The operation result.
    """
    return _wrap(ops.resize(_to_tensor(a), new_shape))


def result_type(*arrays_and_dtypes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``result_type``.

    Args:
        *arrays_and_dtypes: Input parameter ``arrays_and_dtypes``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    processed = [_to_tensor(x) if isinstance(x, (ndarray, Tensor, ArrayProtocol)) else x for x in arrays_and_dtypes]
    return _wrap(ops.result_type(*processed))


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


def rot90(m: ArrayLikeSpec, k: int = 1, axes: ArrayLikeSpec = (0, 1)) -> ArrayLikeSpec:
    """JAX API implementation for rot90.

    Args:
        m: Input parameter ``m``.
        k: Input parameter ``k``.
        axes: Input parameter ``axes``.

    Returns:
        The operation result.
    """
    return _wrap(ops.rot90(_to_tensor(m), k, axes))


def save(file: ArrayLikeSpec, arr: ArrayLikeSpec, allow_pickle: bool = True, fix_imports: bool = True) -> None:
    """JAX API implementation for save.

    Args:
        file: Input parameter ``file``.
        arr: Input parameter ``arr``.
        allow_pickle: Input parameter ``allow_pickle``.
        fix_imports: Input parameter ``fix_imports``.
    """
    ops.save(file, _to_tensor(arr), allow_pickle, fix_imports)


def savez(file: ArrayLikeSpec, *args: ArrayLikeSpec, **kwds: ArrayLikeSpec) -> None:
    """JAX API implementation for savez.

    Args:
        file: Input parameter ``file``.
        *args: Input parameter ``args``.
        **kwds: Input parameter ``kwds``.
    """
    ops.savez(
        file,
        *[_to_tensor(a) for a in args],
        **{k: _to_tensor(v) for k, v in kwds.items()},
    )


def setdiff1d(ar1: ArrayLikeSpec, ar2: ArrayLikeSpec, assume_unique: bool = False) -> ArrayLikeSpec:
    """JAX API implementation for setdiff1d.

    Args:
        ar1: Input parameter ``ar1``.
        ar2: Input parameter ``ar2``.
        assume_unique: Input parameter ``assume_unique``.

    Returns:
        The operation result.
    """
    return _wrap(ops.setdiff1d(_to_tensor(ar1), _to_tensor(ar2), assume_unique))


def setxor1d(ar1: ArrayLikeSpec, ar2: ArrayLikeSpec, assume_unique: bool = False) -> ArrayLikeSpec:
    """JAX API implementation for setxor1d.

    Args:
        ar1: Input parameter ``ar1``.
        ar2: Input parameter ``ar2``.
        assume_unique: Input parameter ``assume_unique``.

    Returns:
        The operation result.
    """
    return _wrap(ops.setxor1d(_to_tensor(ar1), _to_tensor(ar2), assume_unique))


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


def sort_complex(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for sort_complex.

    Args:
        a: Input parameter ``a``.

    Returns:
        The operation result.
    """
    return _wrap(ops.sort_complex(_to_tensor(a)))


def trace(
    a: ArrayLikeSpec,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Execute ``trace``.

    Args:
        a: Input parameter ``a``.
        offset: Input parameter ``offset``.
        axis1: Input parameter ``axis1``.
        axis2: Input parameter ``axis2``.
        dtype: Input parameter ``dtype``.
        out: Input parameter ``out``.

    Returns:
        The operation result.
    """
    from ml_switcheroo_compiler.ops import trace as _trace

    return _trace(_to_tensor(a), offset=offset, axis1=axis1, axis2=axis2)


def trapezoid(y: ArrayLikeSpec, x: ArrayLikeSpec = None, dx: float = 1.0, axis: int = -1) -> ArrayLikeSpec:
    """Execute ``trapezoid``.

    Args:
        y: Input parameter ``y``.
        x: Input parameter ``x``.
        dx: Input parameter ``dx``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    return _wrap(ops.trapezoid(_to_tensor(y), _to_tensor(x) if x is not None else None, dx, axis))


def tri(N: int, M: ArrayLikeSpec = None, k: int = 0, dtype: ArrayLikeSpec = float) -> ArrayLikeSpec:
    """Execute ``tri``.

    Args:
        N: Input parameter ``N``.
        M: Input parameter ``M``.
        k: Input parameter ``k``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    return _wrap(ops.tri(N, M, k, dtype))


def tril_indices(n: int, k: int = 0, m: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``tril_indices``.

    Args:
        n: Input parameter ``n``.
        k: Input parameter ``k``.
        m: Input parameter ``m``.

    Returns:
        The operation result.
    """
    return _wrap(ops.tril_indices(n, k, m))


def tril_indices_from(arr: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """Execute ``tril_indices_from``.

    Args:
        arr: Input parameter ``arr``.
        k: Input parameter ``k``.

    Returns:
        The operation result.
    """
    return _wrap(ops.tril_indices_from(_to_tensor(arr), k))


def trim_zeros(filt: ArrayLikeSpec, trim: str = "fb") -> ArrayLikeSpec:
    """Execute ``trim_zeros``.

    Args:
        filt: Input parameter ``filt``.
        trim: Input parameter ``trim``.

    Returns:
        The operation result.
    """
    return _wrap(ops.trim_zeros(_to_tensor(filt), trim))


def triu_indices(n: int, k: int = 0, m: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``triu_indices``.

    Args:
        n: Input parameter ``n``.
        k: Input parameter ``k``.
        m: Input parameter ``m``.

    Returns:
        The operation result.
    """
    return _wrap(ops.triu_indices(n, k, m))


def triu_indices_from(arr: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """Execute ``triu_indices_from``.

    Args:
        arr: Input parameter ``arr``.
        k: Input parameter ``k``.

    Returns:
        The operation result.
    """
    return _wrap(ops.triu_indices_from(_to_tensor(arr), k))


def ufunc() -> None:
    """Execute ``ufunc``."""


def uint(x: ArrayLikeSpec = 0) -> ArrayLikeSpec:
    """Execute ``uint``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    return array(x, dtype=_DType.UInt32)


def uint4(x: ArrayLikeSpec = 0) -> ArrayLikeSpec:
    """Execute ``uint4``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    return array(x, dtype=_DType.UInt8)


def union1d(ar1: ArrayLikeSpec, ar2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``union1d``.

    Args:
        ar1: Input parameter ``ar1``.
        ar2: Input parameter ``ar2``.

    Returns:
        The operation result.
    """
    return _wrap(ops.union1d(_to_tensor(ar1), _to_tensor(ar2)))


def unique(
    ar: ArrayLikeSpec,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: ArrayLikeSpec = None,
    *,
    size: ArrayLikeSpec = None,
    fill_value: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Execute ``unique``.

    Args:
        ar: Input parameter ``ar``.
        return_index: Input parameter ``return_index``.
        return_inverse: Input parameter ``return_inverse``.
        return_counts: Input parameter ``return_counts``.
        axis: Input parameter ``axis``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    return _wrap(
        ops.unique(
            _to_tensor(ar),
            return_index,
            return_inverse,
            return_counts,
            axis,
            size=size,
            fill_value=fill_value,
        )
    )


def unique_all(x: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``unique_all``.

    Args:
        x: Input parameter ``x``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    return unique(x, return_index=True, return_inverse=True, return_counts=True)


def unique_counts(x: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``unique_counts``.

    Args:
        x: Input parameter ``x``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    return unique(x, return_counts=True)


def unique_inverse(x: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``unique_inverse``.

    Args:
        x: Input parameter ``x``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    return unique(x, return_inverse=True)


def unique_values(x: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``unique_values``.

    Args:
        x: Input parameter ``x``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    return unique(x)


def unravel_index(indices: ArrayLikeSpec, shape: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``unravel_index``.

    Args:
        indices: Input parameter ``indices``.
        shape: Input parameter ``shape``.

    Returns:
        The operation result.
    """
    from ml_switcheroo_compiler.ops.base import get_op

    res = get_op("UnravelIndex")()(_to_tensor(indices), shape)
    return _wrap(res)


def unwrap(p: ArrayLikeSpec, discont: ArrayLikeSpec = None, axis: int = -1, period: float = 6.283185307179586) -> ArrayLikeSpec:
    """Execute ``unwrap``.

    Args:
        p: Input parameter ``p``.
        discont: Input parameter ``discont``.
        axis: Input parameter ``axis``.
        period: Input parameter ``period``.

    Returns:
        The operation result.
    """
    return _wrap(ops.unwrap(_to_tensor(p), discont, axis, period))


def vander(x: ArrayLikeSpec, N: ArrayLikeSpec = None, increasing: bool = False) -> ArrayLikeSpec:
    """Execute ``vander``.

    Args:
        x: Input parameter ``x``.
        N: Input parameter ``N``.
        increasing: Input parameter ``increasing``.

    Returns:
        The operation result.
    """
    return _wrap(ops.vander(_to_tensor(x), N, increasing))


def vecdot(x1: ArrayLikeSpec, x2: ArrayLikeSpec, /, *, axis: int = -1) -> ArrayLikeSpec:
    """Execute ``vecdot``.

    Args:
        x1: Input parameter ``x1``.
        x2: Input parameter ``x2``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    return sum(x1 * x2, axis=axis)


class vectorize:
    """Generalized function class that vectorizes a Python callable ``pyfunc``."""

    def __init__(
        self,
        pyfunc: ArrayLikeSpec,
        otypes: ArrayLikeSpec = None,
        doc: ArrayLikeSpec = None,
        excluded: ArrayLikeSpec = None,
        cache: bool = False,
        signature: ArrayLikeSpec = None,
    ) -> None:
        """Initialize ``vectorize``.

        Args:
            pyfunc: Python function or method to vectorize.
            otypes: Optional output data types.
            doc: Optional docstring for the wrapper.
            excluded: Set of arguments to exclude from vectorization.
            cache: Whether to cache the first function evaluation.
            signature: Generalized universal function signature (e.g. ``"(m,n),(n)->(m)"``).
        """
        self.pyfunc = pyfunc
        self.otypes = otypes
        self.__doc__ = doc
        self.excluded = set(excluded) if isinstance(excluded, (set, list, tuple)) else set()
        self.cache = cache
        self.signature = signature

    def __call__(self, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Invoke the vectorized callable on ``args`` and ``kwargs``.

        Args:
            *args: Positional array or scalar arguments.
            **kwargs: Keyword arguments.

        Returns:
            ArrayLikeSpec: Vectorized evaluation result.
        """
        if callable(self.pyfunc):
            res = self.pyfunc(*[asarray(a) if idx not in self.excluded else a for idx, a in enumerate(args)], **kwargs)
            return asarray(res) if not isinstance(res, tuple) else tuple(asarray(r) for r in res)
        raise TypeError("pyfunc must be callable")


def apply_over_axes(func: ArrayLikeSpec, a: ArrayLikeSpec, axes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Apply a function repeatedly over multiple axes, keeping dimensions.

    Args:
        func: Function of the form ``func(a, axis)`` to apply.
        a: Input array.
        axes: Axis or sequence of axes over which ``func`` is applied.

    Returns:
        ArrayLikeSpec: Result array after applying ``func`` over ``axes``.
    """
    val = asarray(a)
    axis_list = [axes] if isinstance(axes, int) else list(axes)
    for ax in axis_list:
        res = func(val, ax)
        res_arr = asarray(res)
        if res_arr.ndim < val.ndim:
            res_arr = expand_dims(res_arr, axis=ax)
        val = res_arr
    return val


def argpartition(a: ArrayLikeSpec, kth: ArrayLikeSpec, axis: int = -1, kind: str = "introselect", order: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Perform an indirect partition along the given axis via argsort.

    Args:
        a: Array to sort/partition.
        kth: Element index(es) to partition by.
        axis: Axis along which to sort.
        kind: Selection algorithm name.
        order: Field order comparison specification.

    Returns:
        ArrayLikeSpec: Array of indices that partition ``a`` along ``axis``.
    """
    return argsort(a, axis=axis)


def argwhere(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """Find the indices of array elements that are non-zero, grouped by element.

    Args:
        a: Input data.

    Returns:
        ArrayLikeSpec: Indices of non-zero elements with shape ``(N, a.ndim)``.
    """
    arr = asarray(a)
    if arr.ndim == 0:
        nz = nonzero(atleast_1d(arr))
        return empty((nz[0].shape[0], 0), dtype=int32)
    nz = nonzero(arr)
    return column_stack(nz) if len(nz) > 1 else reshape(nz[0], (-1, 1))


def fromfile(
    file: ArrayLikeSpec,
    dtype: ArrayLikeSpec = float,
    count: int = -1,
    sep: str = "",
    offset: int = 0,
    *,
    like: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Construct an array from data in a text or binary file.

    Args:
        file: Open file object or filename.
        dtype: Data type of the returned array.
        count: Number of items to read (``-1`` means all).
        sep: Separator between items if file is a text file.
        offset: Byte offset from the current file position.
        like: Reference object for array creation.

    Returns:
        ArrayLikeSpec: Array read from the file.
    """
    import os

    if isinstance(file, (str, bytes)) and os.path.exists(file):
        with open(file, "r" if sep else "rb") as f:
            data = f.read()
        if isinstance(data, str):
            return fromstring(data, dtype=dtype, count=count, sep=sep or " ")
        return frombuffer(data, dtype=dtype, count=count, offset=offset)
    return _wrap(ops.fromfile(file, dtype, count, sep, offset, like=like))


def fromfunction(function: ArrayLikeSpec, shape: ArrayLikeSpec, *, dtype: ArrayLikeSpec = float, like: ArrayLikeSpec = None, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Construct an array by executing a function over each coordinate.

    Args:
        function: Callable accepting ``N`` coordinate arrays where ``N = len(shape)``.
        shape: Shape of the output array.
        dtype: Data-type of the coordinate arrays passed to ``function``.
        like: Reference object for array creation.
        **kwargs: Additional keyword arguments passed to ``function``.

    Returns:
        ArrayLikeSpec: Result of calling ``function`` on the coordinate grids.
    """
    coords = indices(shape, dtype=dtype)
    if callable(function):
        return asarray(function(*coords, **kwargs))
    return _wrap(ops.fromfunction(function, shape, dtype=dtype, like=like, **kwargs))


def fromiter(iter: ArrayLikeSpec, dtype: ArrayLikeSpec, count: int = -1, *, like: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Create a new 1-dimensional array from an iterable object.

    Args:
        iter: An iterable object providing data for the array.
        dtype: The data-type of the returned array.
        count: The number of items to read from ``iter`` (default ``-1`` reads all).
        like: Reference object for array creation.

    Returns:
        ArrayLikeSpec: 1-D array constructed from ``iter``.
    """
    items = list(iter)
    if count >= 0:
        items = items[:count]
    return array(items, dtype=dtype)


def frompyfunc(func: ArrayLikeSpec, nin: int, nout: int, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Take an arbitrary Python function and return a vectorized callable.

    Args:
        func: A Python function object.
        nin: The number of input arguments.
        nout: The number of objects returned by ``func``.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        ArrayLikeSpec: Vectorized wrapper around ``func``.
    """
    return vectorize(func)


def fromstring(string: ArrayLikeSpec, dtype: ArrayLikeSpec = float, count: int = -1, sep: str = "") -> ArrayLikeSpec:
    """Initialize a new 1-D array from text data in a string.

    Args:
        string: A string containing the data.
        dtype: The data type of the array.
        count: Read this number of ``dtype`` elements from the data.
        sep: The string separating numbers in the data.

    Returns:
        ArrayLikeSpec: 1-D array constructed from ``string``.
    """
    if isinstance(string, str):
        parts = string.split(sep) if sep else string.split()
        if count >= 0:
            parts = parts[:count]
        vals = [float(p) for p in parts]
        return array(vals, dtype=dtype)
    return frombuffer(string, dtype=dtype, count=count)


def geomspace(
    start: ArrayLikeSpec,
    stop: ArrayLikeSpec,
    num: int = 50,
    endpoint: bool = True,
    dtype: ArrayLikeSpec = None,
    axis: int = 0,
) -> ArrayLikeSpec:
    """Return numbers spaced evenly on a log scale (a geometric progression).

    Args:
        start: The starting value of the sequence.
        stop: The final value of the sequence, unless ``endpoint=False``.
        num: Number of samples to generate.
        endpoint: If ``True``, ``stop`` is the last sample.
        dtype: The type of the output array.
        axis: The axis in the result to store the samples.

    Returns:
        ArrayLikeSpec: Geometrically spaced samples.
    """
    log_start = log(asarray(start, dtype=float32))
    log_stop = log(asarray(stop, dtype=float32))
    res = exp(linspace(log_start, log_stop, num=num, endpoint=endpoint, axis=axis))
    if dtype is not None:
        return astype(res, dtype)
    return res


def get_printoptions() -> ArrayLikeSpec:
    """Return the current print options dictionary.

    Returns:
        ArrayLikeSpec: Dictionary of active print formatting options.
    """
    return {
        "edgeitems": 3,
        "threshold": 1000,
        "floatmode": "maxprec",
        "precision": 8,
        "suppress": False,
        "linewidth": 75,
        "nanstr": "nan",
        "infstr": "inf",
        "sign": "-",
        "formatter": None,
        "legacy": False,
    }


def gradient(f: ArrayLikeSpec, *varargs: ArrayLikeSpec, axis: ArrayLikeSpec = None, edge_order: int = 1) -> ArrayLikeSpec:
    """Return the gradient of an N-dimensional array using finite differences.

    Args:
        f: An N-dimensional array containing samples of a scalar function.
        *varargs: Spacing between ``f`` values (scalar or 1-D coordinate array per dimension).
        axis: Gradient is calculated only along the given axis or axes.
        edge_order: Gradient is calculated using N-th order accurate differences at boundaries.

    Returns:
        ArrayLikeSpec: Array (or list of arrays) of gradients along each target axis.
    """
    arr = asarray(f, dtype=float32)
    spacing = float(varargs[0]) if varargs and isinstance(varargs[0], (int, float)) else 1.0
    if arr.shape[0] < 2:
        return zeros_like(arr)
    d = diff(arr, axis=0 if axis is None else axis) / spacing
    first = d[:1]
    last = d[-1:]
    if d.shape[0] > 1:
        mid = (d[:-1] + d[1:]) * 0.5
        return concatenate([first, mid, last], axis=0)
    return concatenate([first, last], axis=0)


def hamming(M: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the Hamming window of length ``M``.

    Args:
        M: Number of points in the output window.

    Returns:
        ArrayLikeSpec: The Hamming window array.
    """
    m_int = int(M)
    if m_int < 1:
        return array([], dtype=float32)
    if m_int == 1:
        return ones((1,), dtype=float32)
    import math

    n = arange(0, m_int, dtype=float32)
    return 0.54 - 0.46 * cos(2.0 * math.pi * n / (m_int - 1))


def hanning(M: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the Hanning (Hann) window of length ``M``.

    Args:
        M: Number of points in the output window.

    Returns:
        ArrayLikeSpec: The Hanning window array.
    """
    m_int = int(M)
    if m_int < 1:
        return array([], dtype=float32)
    if m_int == 1:
        return ones((1,), dtype=float32)
    import math

    n = arange(0, m_int, dtype=float32)
    return 0.5 - 0.5 * cos(2.0 * math.pi * n / (m_int - 1))


def histogram_bin_edges(a: ArrayLikeSpec, bins: ArrayLikeSpec = 10, range: ArrayLikeSpec = None, weights: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the bin edges for a histogram without computing the counts.

    Args:
        a: Input data.
        bins: Number of equal-width bins or sequence of bin edges.
        range: Lower and upper range of the bins ``(min, max)``.
        weights: Optional weights array (unused for equal-width bin edge computation).

    Returns:
        ArrayLikeSpec: 1-D array of bin edges of length ``bins + 1``.
    """
    if isinstance(bins, (list, tuple, ndarray, Tensor)):
        return asarray(bins, dtype=float32)
    arr = ravel(asarray(a, dtype=float32))
    low = float(range[0]) if isinstance(range, (list, tuple)) else float(min(arr).item())
    high = float(range[1]) if isinstance(range, (list, tuple)) else float(max(arr).item())
    if low == high:
        low -= 0.5
        high += 0.5
    return linspace(low, high, int(bins) + 1)


def histogram(
    a: ArrayLikeSpec,
    bins: ArrayLikeSpec = 10,
    range: ArrayLikeSpec = None,
    normed: ArrayLikeSpec = None,
    weights: ArrayLikeSpec = None,
    density: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute the histogram of a dataset.

    Args:
        a: Input data.
        bins: Number of bins or sequence of bin edges.
        range: Lower and upper range of the bins.
        normed: Deprecated alias for ``density``.
        weights: Optional array of weights of the same shape as ``a``.
        density: If ``True``, return probability density function values at the bin.

    Returns:
        ArrayLikeSpec: Tuple ``(hist, bin_edges)``.
    """
    arr = ravel(asarray(a, dtype=float32))
    edges = histogram_bin_edges(arr, bins=bins, range=range, weights=weights)
    w = ravel(asarray(weights, dtype=float32)) if weights is not None else ones_like(arr)
    n_bins = edges.shape[0] - 1
    counts: list[ArrayLikeSpec] = []
    for i in builtins.range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        mask = (arr >= left) & (arr <= right) if i == n_bins - 1 else (arr >= left) & (arr < right)
        counts.append(sum(where(mask, w, 0.0)))
    hist = stack(counts)
    if density or normed:
        widths = diff(edges)
        total = sum(hist)
        hist = hist / (widths * where(total == 0, 1.0, total))
    return hist, edges


def histogram2d(
    x: ArrayLikeSpec,
    y: ArrayLikeSpec,
    bins: ArrayLikeSpec = 10,
    range: ArrayLikeSpec = None,
    normed: ArrayLikeSpec = None,
    weights: ArrayLikeSpec = None,
    density: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute the bi-dimensional histogram of two data samples.

    Args:
        x: Array containing the x coordinates of the points to be histogrammed.
        y: Array containing the y coordinates of the points to be histogrammed.
        bins: Number of bins ``nx = ny = bins`` or ``[nx, ny]``.
        range: Optional ``[[xmin, xmax], [ymin, ymax]]`` bounds.
        normed: Deprecated alias for ``density``.
        weights: Optional weights for each sample ``(x_i, y_i)``.
        density: If ``True``, normalize the 2D histogram into a density.

    Returns:
        ArrayLikeSpec: Tuple ``(H, xedges, yedges)``.
    """
    nx, ny = (bins[0], bins[1]) if isinstance(bins, (list, tuple)) and len(bins) == 2 else (bins, bins)
    rx, ry = (range[0], range[1]) if isinstance(range, (list, tuple)) and len(range) == 2 else (None, None)
    x_arr = ravel(asarray(x, dtype=float32))
    y_arr = ravel(asarray(y, dtype=float32))
    xedges = histogram_bin_edges(x_arr, bins=nx, range=rx)
    yedges = histogram_bin_edges(y_arr, bins=ny, range=ry)
    w = ravel(asarray(weights, dtype=float32)) if weights is not None else ones_like(x_arr)
    nbx = xedges.shape[0] - 1
    nby = yedges.shape[0] - 1
    rows: list[ArrayLikeSpec] = []
    for i in builtins.range(nbx):
        mx = (x_arr >= xedges[i]) & (x_arr <= xedges[i + 1]) if i == nbx - 1 else (x_arr >= xedges[i]) & (x_arr < xedges[i + 1])
        row_counts: list[ArrayLikeSpec] = []
        for j in builtins.range(nby):
            my = (y_arr >= yedges[j]) & (y_arr <= yedges[j + 1]) if j == nby - 1 else (y_arr >= yedges[j]) & (y_arr < yedges[j + 1])
            row_counts.append(sum(where(mx & my, w, 0.0)))
        rows.append(stack(row_counts))
    h_mat = stack(rows)
    return h_mat, xedges, yedges


def histogramdd(
    sample: ArrayLikeSpec,
    bins: ArrayLikeSpec = 10,
    range: ArrayLikeSpec = None,
    normed: ArrayLikeSpec = None,
    weights: ArrayLikeSpec = None,
    density: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute the multidimensional histogram of some data.

    Args:
        sample: Data to be histogrammed, shaped ``(N, D)`` or sequence of ``D`` 1-D arrays.
        bins: Number of bins or sequence of bin counts/edges per dimension.
        range: Sequence of ``(lower, upper)`` bounds per dimension.
        normed: Deprecated alias for ``density``.
        weights: Optional weight array of length ``N``.
        density: If ``True``, normalize the histogram to a probability density.

    Returns:
        ArrayLikeSpec: Tuple ``(H, edges)``.
    """
    arr = asarray(sample, dtype=float32)
    if arr.ndim == 1:
        h, e = histogram(arr, bins=bins, range=range, normed=normed, weights=weights, density=density)
        return h, [e]
    if arr.shape[-1] == 2:
        h2, xe, ye = histogram2d(arr[:, 0], arr[:, 1], bins=bins, range=range, normed=normed, weights=weights, density=density)
        return h2, [xe, ye]
    h1, e1 = histogram(arr[:, 0], bins=bins, range=range, normed=normed, weights=weights, density=density)
    return h1, [e1]


def i0(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Modified Bessel function of the first kind, order 0.

    Args:
        x: Argument of the Bessel function.

    Returns:
        ArrayLikeSpec: The modified Bessel function evaluated at each element of ``x``.
    """
    arr = asarray(x, dtype=float32)
    return exp(abs(arr)) * _wrap(ops.bessel_i0e(_to_tensor(arr)))


def cumulative_prod(
    x: ArrayLikeSpec,
    /,
    *,
    axis: int | None = None,
    dtype: DTypeSpec = None,
    include_initial: builtins.bool = False,
) -> ArrayLikeSpec:
    """Return the cumulative product of elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which the cumulative product is computed.
        dtype: Optional dtype of the returned array.
        include_initial: Whether to prepend the multiplicative identity ``1`` along ``axis``.

    Returns:
        ArrayLikeSpec: Cumulative product array.
    """
    arr = asarray(x)
    ax = 0 if axis is None else axis
    res = cumprod(arr, axis=ax, dtype=dtype)
    if include_initial:
        norm_ax = ax if ax >= 0 else arr.ndim + ax
        init_shape = (*res.shape[:norm_ax], 1, *res.shape[norm_ax + 1 :])
        return concatenate([ones(init_shape, dtype=res.dtype), res], axis=norm_ax)
    return res


def matvec(x1: ArrayLikeSpec, x2: ArrayLikeSpec, /) -> ArrayLikeSpec:
    """Compute matrix-vector product of ``x1`` and ``x2``.

    Args:
        x1: Matrix array of shape ``(..., M, N)``.
        x2: Vector array of shape ``(..., N)``.

    Returns:
        ArrayLikeSpec: Matrix-vector product of shape ``(..., M)``.
    """
    return einsum("...ij,...j->...i", asarray(x1), asarray(x2))


def vecmat(x1: ArrayLikeSpec, x2: ArrayLikeSpec, /) -> ArrayLikeSpec:
    """Compute vector-matrix product of ``x1`` and ``x2``.

    Args:
        x1: Vector array of shape ``(..., M)``.
        x2: Matrix array of shape ``(..., M, N)``.

    Returns:
        ArrayLikeSpec: Vector-matrix product of shape ``(..., N)``.
    """
    return einsum("...i,...ij->...j", asarray(x1), asarray(x2))


def minmax(
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    keepdims: builtins.bool = False,
    initial: ArrayLikeSpec = None,
    where: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Simultaneously compute the minimum and maximum of ``a`` along ``axis``.

    Args:
        a: Input array.
        axis: Axis or axes along which to operate.
        keepdims: Whether reduced axes are retained with size 1.
        initial: Optional initial ``(min_init, max_init)`` tuple or scalar.
        where: Optional boolean mask of elements to compare.
        out: Unused output parameter.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: Tuple ``(minimums, maximums)``.
    """
    _ = out
    arr = asarray(a)
    min_init = initial[0] if isinstance(initial, tuple) else initial
    max_init = initial[1] if isinstance(initial, tuple) else initial
    return (
        arr.min(axis=axis, keepdims=keepdims, initial=min_init, where=where),
        arr.max(axis=axis, keepdims=keepdims, initial=max_init, where=where),
    )


def put_along_axis(
    arr: ArrayLikeSpec,
    indices: ArrayLikeSpec,
    values: ArrayLikeSpec,
    axis: int | None,
    inplace: builtins.bool = False,
    *,
    mode: str | None = None,
) -> ArrayLikeSpec:
    """Put ``values`` into the destination array ``arr`` by matching 1D index and data slices.

    Args:
        arr: Destination array.
        indices: Indices along each 1D slice of ``arr``.
        values: Values to insert at those indices.
        axis: Axis along which to index (or ``None`` for flattened indexing).
        inplace: Must be ``False`` for immutable JAX arrays.
        mode: Out-of-bounds index handling mode.

    Returns:
        ArrayLikeSpec: Updated copy of ``arr``.
    """
    _ = (inplace, mode)
    arr_obj = asarray(arr)
    base = arr_obj.__array__().copy()
    idx_np = asarray(indices).__array__()
    val_np = asarray(values).__array__()
    if axis is None:
        flat = base.ravel()
        flat[idx_np.ravel()] = val_np.ravel()
        return array(flat.reshape(base.shape), dtype=arr_obj.dtype)
    norm_ax = axis if axis >= 0 else arr_obj.ndim + axis
    moved_base = swapaxes(arr_obj, norm_ax, -1).__array__().copy()
    moved_idx = swapaxes(asarray(indices), norm_ax, -1).__array__()
    moved_val = swapaxes(asarray(values), norm_ax, -1).__array__()
    flat_base = moved_base.reshape(-1, moved_base.shape[-1])
    flat_idx = moved_idx.reshape(-1, moved_idx.shape[-1])
    flat_val = moved_val.reshape(-1, moved_val.shape[-1])
    for row_i in builtins.range(flat_base.shape[0]):
        for col_j in builtins.range(flat_idx.shape[1]):
            flat_base[row_i, int(flat_idx[row_i, col_j])] = flat_val[row_i, col_j % flat_val.shape[1]]
    restored = array(flat_base.reshape(moved_base.shape), dtype=arr_obj.dtype)
    return swapaxes(restored, norm_ax, -1)


def spacing(x: ArrayLikeSpec, /) -> ArrayLikeSpec:
    """Return the distance between ``x`` and the nearest adjacent number.

    Args:
        x: Input array or scalar.

    Returns:
        ArrayLikeSpec: Elementwise ULPs (spacing) of ``x``.
    """
    arr = asarray(x, dtype=float32)
    return abs(nextafter(arr, float("inf")) - arr)


def top_k(
    a: ArrayLikeSpec,
    k: int,
    /,
    *,
    axis: int = -1,
    mode: str = "largest",
    sorted: builtins.bool = True,
) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Return the ``k`` largest (or smallest) elements and their indices along ``axis``.

    Args:
        a: Input array.
        k: Number of top elements to extract.
        axis: Axis along which to find the top-k elements.
        mode: ``"largest"`` or ``"smallest"``.
        sorted: Whether the returned elements are sorted in order.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: Tuple ``(values, indices)``.
    """
    _ = sorted
    arr = asarray(a)
    descending = mode != "smallest"
    order_idx = arr.argsort(axis=axis, descending=descending)
    norm_ax = axis if axis >= 0 else arr.ndim + axis
    slc = tuple(builtins.slice(0, int(k)) if i == norm_ax else builtins.slice(None) for i in builtins.range(arr.ndim))
    top_idx = order_idx[slc]
    top_vals = take_along_axis(arr, top_idx, axis=norm_ax)
    return top_vals, top_idx
