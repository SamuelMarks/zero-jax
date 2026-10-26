"""Core ndarray implementation and tensor conversion helpers."""

from __future__ import annotations

import builtins
from collections.abc import Callable

import ml_switcheroo_compiler as _ml_switcheroo_compiler
from ml_switcheroo_compiler import Tensor
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import TensorConfig as _TensorConfig
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import (
    ArrayLikeSpec,
    ArrayProtocol,
    ConfigValue,
    DTypeSpec,
    VariadicCallable,
)
from zero_jax.numpy.tensor_utils import to_array as _to_array


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

    def _traced_mask(self) -> Tensor | None:
        """Build a boolean broadcast mask Tensor when tracing an integer index update.

        Returns:
            Tensor | None: Boolean mask Tensor when tracing, or ``None`` in eager mode.
        """
        from ml_switcheroo_compiler.tracing.tracer import global_tracing_state

        from zero_jax.numpy._creation import arange

        if global_tracing_state.is_tracing and isinstance(self._idx, int) and len(self._array.shape) >= 1:
            dim0 = self._array.shape[0]
            idx_val = self._idx if self._idx >= 0 else dim0 + self._idx
            mask = ops.equal(arange(0, dim0)._tensor, _to_tensor(int(idx_val)))
            if len(self._array.shape) > 1:
                mask = ops.reshape(mask, (dim0,) + (1,) * (len(self._array.shape) - 1))
            return mask
        return None

    def _traced_where(self, mask: Tensor, updated: Tensor) -> ndarray:
        """Select between ``updated`` and ``self._array._tensor`` under ``mask`` while preserving shape.

        Args:
            mask: Boolean broadcast mask tensor.
            updated: Candidate updated tensor.

        Returns:
            ndarray: Shape-preserving traced ``ndarray``.
        """
        raw = ops.where(mask, updated, self._array._tensor)
        proxy = ProxyTensor(id=raw.data.id, shape=self._array.shape, dtype=self._array._tensor.dtype.value)
        return ndarray(Tensor(data=proxy, config=self._array._tensor.config))

    def set(self, values: ArrayLikeSpec) -> ndarray:
        """Return a new ``ndarray`` with elements at ``self._idx`` replaced by ``values``.

        Args:
            values: Replacement values.

        Returns:
            ndarray: Updated copy of the array.
        """
        from zero_jax.numpy._creation import array

        mask = self._traced_mask()
        if mask is not None:
            return self._traced_where(mask, _to_tensor(values))
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
        from zero_jax.numpy._creation import array

        mask = self._traced_mask()
        if mask is not None:
            return self._traced_where(mask, ops.add(self._array._tensor, _to_tensor(values)))
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
        from zero_jax.numpy._creation import array

        mask = self._traced_mask()
        if mask is not None:
            return self._traced_where(mask, ops.multiply(self._array._tensor, _to_tensor(values)))
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
        from zero_jax.numpy._creation import array

        mask = self._traced_mask()
        if mask is not None:
            return self._traced_where(mask, ops.divide(self._array._tensor, _to_tensor(values)))
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
        from zero_jax.numpy._creation import array

        mask = self._traced_mask()
        if mask is not None:
            return self._traced_where(mask, ops.power(self._array._tensor, _to_tensor(values)))
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
        from zero_jax.numpy._creation import array
        from zero_jax.numpy._math import minimum

        mask = self._traced_mask()
        if mask is not None:
            return self._traced_where(mask, ops.minimum(self._array._tensor, _to_tensor(values)))
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
        from zero_jax.numpy._creation import array
        from zero_jax.numpy._math import maximum

        mask = self._traced_mask()
        if mask is not None:
            return self._traced_where(mask, ops.maximum(self._array._tensor, _to_tensor(values)))
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
        from zero_jax.numpy._manipulation import transpose

        return transpose(self)

    @property
    def mT(self) -> ArrayLikeSpec:
        """Return the matrix-transposed array (swapping the last two axes).

        Returns:
            ArrayLikeSpec: Matrix-transposed array.
        """
        from zero_jax.numpy._manipulation import swapaxes

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
    ) -> ndarray:
        """Copy of the array, cast to a specified type.

        Args:
            dtype: Target dtype.
            copy: Whether to force a copy.
            device: Optional target device.

        Returns:
            ndarray: Cast array.
        """
        from zero_jax.numpy._dtypes import astype

        return astype(self, dtype, copy=copy, device=device)

    def reshape(
        self,
        *args: ArrayLikeSpec,
        order: str = "C",
        out_sharding: ConfigValue = None,
    ) -> ndarray:
        """Return an array containing the same data with a new shape.

        Args:
            *args: New shape specification (either a tuple/list or multiple int arguments).
            order: Memory layout order (``"C"`` or ``"F"``).
            out_sharding: Optional output sharding specification.

        Returns:
            ndarray: Reshaped array.
        """
        from zero_jax.numpy._manipulation import reshape

        _ = (order, out_sharding)
        newshape = tuple(int(d) for d in args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(int(d) for d in args)
        return reshape(self, newshape)

    def transpose(self, *args: ArrayLikeSpec) -> ndarray:
        """Return a view of the array with axes transposed.

        Args:
            *args: Optional permutation of axes.

        Returns:
            ndarray: Transposed array.
        """
        from zero_jax.numpy._manipulation import transpose

        if not args or args == (None,):
            return transpose(self, None)
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return transpose(self, [int(a) for a in args[0]])
        return transpose(self, [int(a) for a in args])

    def ravel(
        self,
        order: str = "C",
        out_sharding: ConfigValue = None,
    ) -> ndarray:
        """Return a flattened 1-D array.

        Args:
            order: Memory layout order (``"C"`` or ``"F"``).
            out_sharding: Optional output sharding specification.

        Returns:
            ndarray: Flattened 1-D array.
        """
        from zero_jax.numpy._manipulation import ravel

        _ = out_sharding
        return ravel(self, order=order)

    def flatten(
        self,
        order: str = "C",
        out_sharding: ConfigValue = None,
    ) -> ndarray:
        """Return a copy of the array collapsed into one dimension.

        Args:
            order: Memory layout order (``"C"`` or ``"F"``).
            out_sharding: Optional output sharding specification.

        Returns:
            ndarray: Flattened 1-D array.
        """
        from zero_jax.numpy._manipulation import ravel

        _ = out_sharding
        return ravel(self, order=order)

    def squeeze(self, axis: ArrayLikeSpec = None) -> ndarray:
        """Remove axes of length one from ``self``.

        Args:
            axis: Optional axis or axes to squeeze.

        Returns:
            ndarray: Squeezed array.
        """
        from zero_jax.numpy._manipulation import squeeze

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
    ) -> ndarray:
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
            ndarray: Summed array.
        """
        from zero_jax.numpy._dtypes import astype
        from zero_jax.numpy._math import add
        from zero_jax.numpy._reductions import sum

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
    ) -> ndarray:
        """Test whether all array elements along a given axis evaluate to True.

        Args:
            axis: Axis or axes along which a logical AND reduction is performed.
            out: Unused output parameter.
            keepdims: Whether reduced axes are retained with size 1.
            where: Optional boolean mask of elements to include.

        Returns:
            ndarray: Boolean result array.
        """
        from zero_jax.numpy._reductions import all

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
    ) -> ndarray:
        """Test whether any array element along a given axis evaluates to True.

        Args:
            axis: Axis or axes along which a logical OR reduction is performed.
            out: Unused output parameter.
            keepdims: Whether reduced axes are retained with size 1.
            where: Optional boolean mask of elements to include.

        Returns:
            ndarray: Boolean result array.
        """
        from zero_jax.numpy._reductions import any

        _ = out
        arr = _wrap(ops.where(_to_tensor(where), self._tensor, _to_tensor(False))) if where is not None else self
        return any(arr, axis=axis, keepdims=keepdims)

    def argmax(
        self,
        axis: int | None = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool | None = None,
    ) -> ndarray:
        """Return indices of the maximum values along the given axis.

        Args:
            axis: Axis along which to operate.
            out: Unused output parameter.
            keepdims: Whether reduced axis is retained with size 1.

        Returns:
            ndarray: Array of indices.
        """
        from zero_jax.numpy._reductions import argmax

        _ = out
        return argmax(self, axis=axis, keepdims=builtins.bool(keepdims))

    def argmin(
        self,
        axis: int | None = None,
        out: ArrayLikeSpec = None,
        keepdims: builtins.bool | None = None,
    ) -> ndarray:
        """Return indices of the minimum values along the given axis.

        Args:
            axis: Axis along which to operate.
            out: Unused output parameter.
            keepdims: Whether reduced axis is retained with size 1.

        Returns:
            ndarray: Array of indices.
        """
        from zero_jax.numpy._reductions import argmin

        _ = out
        return argmin(self, axis=axis, keepdims=builtins.bool(keepdims))

    def argpartition(self, kth: int, axis: int = -1) -> ndarray:
        """Return indices that would partition this array along ``axis``.

        Args:
            kth: Element index to partition by.
            axis: Axis along which to sort.

        Returns:
            ndarray: Indices of partitioned array.
        """
        from zero_jax.numpy._indexing_set import argpartition

        return argpartition(self, kth=kth, axis=axis)

    def argsort(
        self,
        axis: int | None = -1,
        *,
        kind: str | None = None,
        order: ConfigValue = None,
        stable: builtins.bool = True,
        descending: builtins.bool = False,
    ) -> ndarray:
        """Return the indices that would sort this array.

        Args:
            axis: Axis along which to sort.
            kind: Sorting algorithm (unused).
            order: Field order (unused).
            stable: Whether to preserve relative order of equal elements.
            descending: Whether to sort in descending order.

        Returns:
            ndarray: Array of sort indices.
        """
        from zero_jax.numpy._indexing_set import argsort
        from zero_jax.numpy._manipulation import flip

        _ = stable
        res = argsort(self, axis=axis, kind=kind, order=order)
        if descending:
            flip_ax = 0 if axis is None else axis
            return flip(res, axis=flip_ax)
        return res

    def byteswap(self) -> ndarray:
        """Swap the bytes of the array elements.

        Returns:
            ndarray: Byte-swapped array.
        """
        from zero_jax.numpy._creation import array

        return array(self.__array__().byteswap(), dtype=self.dtype)

    def choose(
        self,
        choices: ArrayLikeSpec,
        out: ArrayLikeSpec = None,
        mode: str = "raise",
    ) -> ndarray:
        """Construct an array from an index array and a list of arrays to choose from.

        Args:
            choices: Sequence of choice arrays.
            out: Unused output array.
            mode: Out-of-bounds index handling mode.

        Returns:
            ndarray: Chosen array.
        """
        from zero_jax.numpy._indexing_set import choose

        return choose(self, choices, out=out, mode=mode)

    def clip(
        self,
        min: ArrayLikeSpec = None,
        max: ArrayLikeSpec = None,
    ) -> ndarray:
        """Return an array whose values are limited to ``[min, max]``.

        Args:
            min: Minimum value.
            max: Maximum value.

        Returns:
            ndarray: Clipped array.
        """
        from zero_jax.numpy._math import clip

        return clip(self, a_min=min, a_max=max)

    def compress(
        self,
        condition: ArrayLikeSpec,
        axis: int | None = None,
        *,
        out: ArrayLikeSpec = None,
        size: int | None = None,
        fill_value: ArrayLikeSpec = 0,
    ) -> ndarray:
        """Return selected slices of this array along given axis.

        Args:
            condition: 1-D boolean array selecting entries.
            axis: Axis along which to take slices.
            out: Unused output array.
            size: Optional static output size.
            fill_value: Padding value when ``size`` is specified.

        Returns:
            ndarray: Compressed array.
        """
        from zero_jax.numpy._indexing_set import compress

        return compress(condition, self, axis=axis, size=size, fill_value=fill_value, out=out)

    def conj(self) -> ndarray:
        """Return the complex conjugate, element-wise.

        Returns:
            ndarray: Complex-conjugated array.
        """
        from zero_jax.numpy._math import conj

        return conj(self)

    def conjugate(self) -> ndarray:
        """Return the complex conjugate, element-wise.

        Returns:
            ndarray: Complex-conjugated array.
        """
        from zero_jax.numpy._math import conjugate

        return conjugate(self)

    def copy(self) -> ndarray:
        """Return a copy of the array.

        Returns:
            ndarray: Copied array.
        """
        from zero_jax.numpy._creation import copy

        return copy(self)

    def copy_to_host_async(self) -> None:
        """Request an asynchronous copy of the array to host memory."""

    def cumprod(
        self,
        axis: int | None = None,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
    ) -> ndarray:
        """Return the cumulative product of elements along a given axis.

        Args:
            axis: Axis along which the cumulative product is computed.
            dtype: Optional dtype of the returned array.
            out: Unused output parameter.

        Returns:
            ndarray: Cumulative product array.
        """
        from zero_jax.numpy._reductions import cumprod

        return cumprod(self, axis=axis, dtype=dtype, out=out)

    def cumsum(
        self,
        axis: int | None = None,
        dtype: DTypeSpec = None,
        out: ArrayLikeSpec = None,
    ) -> ndarray:
        """Return the cumulative sum of elements along a given axis.

        Args:
            axis: Axis along which the cumulative sum is computed.
            dtype: Optional dtype of the returned array.
            out: Unused output parameter.

        Returns:
            ndarray: Cumulative sum array.
        """
        from zero_jax.numpy._reductions import cumsum

        _ = out
        return cumsum(self, axis=axis, dtype=dtype)

    def diagonal(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
    ) -> ndarray:
        """Return specified diagonals.

        Args:
            offset: Offset of the diagonal from the main diagonal.
            axis1: First axis of the 2-D sub-arrays.
            axis2: Second axis of the 2-D sub-arrays.

        Returns:
            ndarray: Array of diagonals.
        """
        from zero_jax.numpy._creation import diagonal

        return diagonal(self, offset=offset, axis1=axis1, axis2=axis2)

    def dot(
        self,
        b: ArrayLikeSpec,
        *,
        precision: ConfigValue = None,
        preferred_element_type: DTypeSpec = None,
    ) -> ndarray:
        """Compute the dot product of this array with ``b``.

        Args:
            b: Second operand.
            precision: Optional matmul precision.
            preferred_element_type: Optional accumulation/output dtype.

        Returns:
            ndarray: Dot product result.
        """
        from zero_jax.numpy._dtypes import astype
        from zero_jax.numpy._windows_poly import dot

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
    ) -> tuple[ndarray, ...]:
        """Return the indices of the elements that are non-zero.

        Args:
            fill_value: Optional padding value when ``size`` is specified.
            size: Optional static number of non-zero elements to return.

        Returns:
            tuple[ndarray, ...]: Tuple of index arrays for each dimension.
        """
        from zero_jax.numpy._indexing_set import nonzero

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
    ) -> ndarray:
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
            ndarray: Product array.
        """
        from zero_jax.numpy._math import multiply
        from zero_jax.numpy._reductions import prod

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
    ) -> ndarray:
        """Return the peak-to-peak (maximum - minimum) value along a given axis.

        Args:
            axis: Axis along which to find the peaks.
            out: Unused output parameter.
            keepdims: Whether reduced axes are retained with size 1.

        Returns:
            ndarray: Peak-to-peak array.
        """
        from zero_jax.numpy._reductions import ptp

        return ptp(self, axis=axis, out=out, keepdims=keepdims)

    def repeat(
        self,
        repeats: ArrayLikeSpec,
        axis: int | None = None,
        *,
        total_repeat_length: int | None = None,
        out_sharding: ConfigValue = None,
    ) -> ndarray:
        """Repeat elements of an array.

        Args:
            repeats: Number of repetitions for each element.
            axis: Axis along which to repeat values.
            total_repeat_length: Optional static total repeat length.
            out_sharding: Optional output sharding specification.

        Returns:
            ndarray: Repeated array.
        """
        from zero_jax.numpy._manipulation import repeat

        _ = (total_repeat_length, out_sharding)
        return repeat(self, repeats=repeats, axis=axis)

    def round(
        self,
        decimals: int = 0,
        out: ArrayLikeSpec = None,
    ) -> ndarray:
        """Round an array to the given number of decimals.

        Args:
            decimals: Number of decimal places to round to.
            out: Unused output parameter.

        Returns:
            ndarray: Rounded array.
        """
        from zero_jax.numpy._math import round

        _ = out
        return round(self, decimals=decimals)

    def searchsorted(
        self,
        v: ArrayLikeSpec,
        side: str = "left",
        sorter: ArrayLikeSpec = None,
        *,
        method: str = "scan",
    ) -> ndarray:
        """Find indices where elements should be inserted to maintain order.

        Args:
            v: Values to insert into ``self``.
            side: ``"left"`` or ``"right"``.
            sorter: Optional array of integer indices that sort ``self``.
            method: Search algorithm name.

        Returns:
            ndarray: Array of insertion points.
        """
        from zero_jax.numpy._indexing_set import searchsorted

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
    ) -> ndarray:
        """Return a sorted copy of an array.

        Args:
            axis: Axis along which to sort.
            kind: Sorting algorithm (unused).
            order: Field order (unused).
            stable: Whether to use a stable sort.
            descending: Whether to sort in descending order.

        Returns:
            ndarray: Sorted array.
        """
        from zero_jax.numpy._indexing_set import sort
        from zero_jax.numpy._manipulation import flip

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
    ) -> ndarray:
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
            ndarray: Standard deviation array.
        """
        from zero_jax.numpy._math import sqrt

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

    def swapaxes(self, axis1: int, axis2: int) -> ndarray:
        """Return a view of the array with ``axis1`` and ``axis2`` interchanged.

        Args:
            axis1: First axis.
            axis2: Second axis.

        Returns:
            ndarray: Array with swapped axes.
        """
        from zero_jax.numpy._manipulation import swapaxes

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
    ) -> ndarray:
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
            ndarray: Indexed sub-array.
        """
        from zero_jax.numpy._indexing_set import take

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
    ) -> ndarray:
        """Return the sum along diagonals of the array.

        Args:
            offset: Offset of the diagonal from the main diagonal.
            axis1: First axis of the 2-D sub-arrays.
            axis2: Second axis of the 2-D sub-arrays.
            dtype: Optional dtype of the returned array.
            out: Unused output parameter.

        Returns:
            ndarray: Sum along diagonals.
        """
        from zero_jax.numpy._windows_poly import trace

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
    ) -> ndarray:
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
            ndarray: Variance array.
        """
        from zero_jax.numpy._reductions import var

        _ = (out, where)
        eff_ddof = int(correction) if correction is not None else ddof
        return var(self, axis=axis, dtype=dtype, keepdims=keepdims, ddof=eff_ddof)

    def view(
        self,
        dtype: DTypeSpec = None,
        type: ConfigValue = None,
    ) -> ndarray:
        """Return a new view of the array with the same data.

        Args:
            dtype: Optional data-type descriptor of the returned view.
            type: Unused view container class parameter.

        Returns:
            ndarray: Array view.
        """
        from zero_jax.numpy._creation import array

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
    ) -> ndarray:
        """Return the average of the array elements along given axis.

        Args:
            axis: Axis or axes along which the means are computed.
            dtype: Optional output dtype.
            out: Optional output array (unused).
            keepdims: Whether reduced axes are left in the result as dimensions with size one.
            where: Optional boolean mask of elements to include.

        Returns:
            ndarray: Mean array.
        """
        from zero_jax.numpy._dtypes import astype, float32
        from zero_jax.numpy._math import divide
        from zero_jax.numpy._reductions import mean, sum

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
    ) -> ndarray:
        """Return the minimum along a given axis.

        Args:
            axis: Axis or axes along which to operate.
            out: Optional output array (unused).
            keepdims: Whether reduced axes are left in the result as dimensions with size one.
            initial: Optional maximum value of an output element.
            where: Optional boolean mask of elements to compare.

        Returns:
            ndarray: Minimum array.
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
    ) -> ndarray:
        """Return the maximum along a given axis.

        Args:
            axis: Axis or axes along which to operate.
            out: Optional output array (unused).
            keepdims: Whether reduced axes are left in the result as dimensions with size one.
            initial: Optional minimum value of an output element.
            where: Optional boolean mask of elements to compare.

        Returns:
            ndarray: Maximum array.
        """
        from zero_jax.numpy._reductions import max

        return max(self, axis=axis, keepdims=keepdims, where=where, initial=initial)

    def __matmul__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform matrix multiplication with ``other``.

        Args:
            other: Right-hand matrix/vector operand.

        Returns:
            ndarray: Matrix product.
        """
        from zero_jax.numpy._windows_poly import matmul

        return matmul(self, other)

    def __rmatmul__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform reflected matrix multiplication with ``other``.

        Args:
            other: Left-hand matrix/vector operand.

        Returns:
            ndarray: Matrix product.
        """
        from zero_jax.numpy._windows_poly import matmul

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
        from zero_jax.numpy._dtypes import astype

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

    def __add__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the add operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the add operation.
        """
        from zero_jax.numpy._math import add

        return add(self, other)

    def __radd__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the radd operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the radd operation.
        """
        from zero_jax.numpy._math import add

        return add(other, self)

    def __sub__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform elementwise subtraction ``self - other``.

        Args:
            other (ArrayLikeSpec): The right-hand operand for subtraction.

        Returns:
            ndarray: The result of the subtraction operation.
        """
        from zero_jax.numpy._math import subtract

        return subtract(self, other)

    def __rsub__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform reflected elementwise subtraction ``other - self``.

        Args:
            other (ArrayLikeSpec): The left-hand operand for subtraction.

        Returns:
            ndarray: The result of the reflected subtraction operation.
        """
        from zero_jax.numpy._math import subtract

        return subtract(other, self)

    def __mul__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the mul operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the mul operation.
        """
        from zero_jax.numpy._math import multiply

        return multiply(self, other)

    def __rmul__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the rmul operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the rmul operation.
        """
        from zero_jax.numpy._math import multiply

        return multiply(other, self)

    def __pow__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the pow operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the pow operation.
        """
        from zero_jax.numpy._math import power

        return power(self, other)

    def __rpow__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the rpow operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the rpow operation.
        """
        from zero_jax.numpy._math import power

        return power(other, self)

    def __truediv__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the truediv operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the truediv operation.
        """
        from zero_jax.numpy._math import true_divide

        return true_divide(self, other)

    def __rtruediv__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the rtruediv operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the rtruediv operation.
        """
        from zero_jax.numpy._math import true_divide

        return true_divide(other, self)

    def __floordiv__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the floordiv operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the floordiv operation.
        """
        from zero_jax.numpy._math import floor_divide

        return floor_divide(self, other)

    def __rfloordiv__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the rfloordiv operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the rfloordiv operation.
        """
        from zero_jax.numpy._math import floor_divide

        return floor_divide(other, self)

    def __mod__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform elementwise modulo ``self % other``.

        Args:
            other: Right-hand divisor operand.

        Returns:
            ndarray: Elementwise remainder.
        """
        from zero_jax.numpy._math import mod

        return mod(self, other)

    def __rmod__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform reflected elementwise modulo ``other % self``.

        Args:
            other: Left-hand dividend operand.

        Returns:
            ndarray: Elementwise remainder.
        """
        from zero_jax.numpy._math import mod

        return mod(other, self)

    def __neg__(self) -> ndarray:
        """Perform elementwise negation ``-self``.

        Returns:
            ndarray: The result of the negation operation.
        """
        from zero_jax.numpy._math import negative

        return negative(self)

    def __lt__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the lt operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the lt operation.
        """
        return _wrap(ops.less(self._tensor, _to_tensor(other)))

    def __gt__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the gt operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the gt operation.
        """
        return _wrap(ops.greater(self._tensor, _to_tensor(other)))

    def __le__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the le operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the le operation.
        """
        return _wrap(ops.less_equal(self._tensor, _to_tensor(other)))

    def __ge__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform the ge operation.

        Args:
            other (ArrayLikeSpec): The other operand for the operation.

        Returns:
            ndarray: The result of the ge operation.
        """
        return _wrap(ops.greater_equal(self._tensor, _to_tensor(other)))

    def __and__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform elementwise bitwise/logical AND ``self & other``.

        Args:
            other: Right-hand operand.

        Returns:
            ndarray: Elementwise AND result.
        """
        from zero_jax.numpy._math import bitwise_and, logical_and

        if self.dtype.name == "bool":
            return logical_and(self, other)
        return bitwise_and(self, other)

    def __or__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform elementwise bitwise/logical OR ``self | other``.

        Args:
            other: Right-hand operand.

        Returns:
            ndarray: Elementwise OR result.
        """
        from zero_jax.numpy._math import bitwise_or, logical_or

        if self.dtype.name == "bool":
            return logical_or(self, other)
        return bitwise_or(self, other)

    def __xor__(self, other: ArrayLikeSpec) -> ndarray:
        """Perform elementwise bitwise/logical XOR ``self ^ other``.

        Args:
            other: Right-hand operand.

        Returns:
            ndarray: Elementwise XOR result.
        """
        from zero_jax.numpy._math import bitwise_xor, logical_xor

        if self.dtype.name == "bool":
            return logical_xor(self, other)
        return bitwise_xor(self, other)

    def __invert__(self) -> ndarray:
        """Perform elementwise bitwise/logical NOT ``~self``.

        Returns:
            ndarray: Elementwise NOT result.
        """
        from zero_jax.numpy._math import bitwise_not, logical_not

        if self.dtype.name == "bool":
            return logical_not(self)
        return bitwise_not(self)

    def __setitem__(self, key: ArrayLikeSpec, value: ArrayLikeSpec) -> None:
        """Disallow in-place mutation of ``ndarray`` instances.

        Args:
            key: Index specification.
            value: Replacement value.

        Raises:
            TypeError: Always raised because JAX arrays are immutable.
        """
        _ = (key, value)
        raise TypeError("JAX arrays are immutable; use arr.at[idx].set(val) instead.")

    def __getitem__(self, key: ArrayLikeSpec) -> ndarray:
        """Perform the getitem operation.

        Args:
            key: Input parameter ``key``.

        Returns:
            ndarray: The operation result.
        """
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
            dim0 = t.shape[0]
            start = (key.start if key.start >= 0 else dim0 + key.start) if key.start is not None else 0
            stop = (key.stop if key.stop >= 0 else dim0 + key.stop) if key.stop is not None else dim0
            step = key.step if key.step is not None else 1
            return _wrap(ops.strided_slice(t, [start], [stop], [step]))

        if isinstance(key, tuple) and builtins.all(isinstance(k, slice) or k is None for k in key):
            starts = []
            stops = []
            steps = []
            for k in key:
                if k is None:
                    continue
                dim_sz = t.shape[len(starts)]
                start = (k.start if k.start >= 0 else dim_sz + k.start) if k.start is not None else 0
                stop = (k.stop if k.stop >= 0 else dim_sz + k.stop) if k.stop is not None else dim_sz
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

    def __len__(self) -> int:
        """Perform the len operation.

        Returns:
            int: The result of the len operation.
        """
        return self.shape[0] if self.shape else 0

    def __iter__(self) -> ArrayLikeSpec:
        """Perform the iter operation.

        Returns:
            ArrayLikeSpec: The result of the iter operation.
        """
        from zero_jax.numpy._creation import array

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
        from ml_switcheroo_ir import DType

        parsed_dtype = x.dtype if isinstance(x.dtype, DType) else (DType(x.dtype) if isinstance(x.dtype, str) else config.default_float_dtype)
        return _ml_switcheroo_compiler.Tensor(
            data=x,
            config=_TensorConfig(
                shape=x.shape,
                dtype=parsed_dtype,
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


def _ensure_tensor(x: ArrayLikeSpec) -> Tensor:
    """Convert an array-like input into a compiler ``Tensor``.

    Args:
        x: Input array-like value.

    Returns:
        Tensor: Compiler tensor representation.
    """
    return _to_tensor(x)


__all__ = [
    "_IndexUpdateHelper",
    "_IndexUpdateRef",
    "_ensure_tensor",
    "_to_tensor",
    "_wrap",
    "ndarray",
]
