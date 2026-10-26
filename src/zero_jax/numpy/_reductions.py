"""Reduction, statistical, difference, and histogram operations for zero_jax.numpy."""

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
from zero_jax.numpy._creation import asarray, linspace, ones, ones_like, zeros_like
from zero_jax.numpy._dtypes import float32
from zero_jax.numpy._manipulation import concatenate, expand_dims, ravel, stack
from zero_jax.numpy._math import multiply, power, subtract
from zero_jax.numpy._ndarray import _to_tensor, _wrap, ndarray
from zero_jax.numpy.tensor_utils import to_array as _to_array


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
    _ = dtype
    t = _to_tensor(a)
    m = mean(a, axis=axis, keepdims=True)
    diff = subtract(t, m)
    sq = multiply(diff, diff)
    res = mean(sq, axis=axis, keepdims=keepdims)
    if ddof != 0:
        import math

        total_n = math.prod(t.shape) if t.shape else 1
        denom = builtins.max(int(total_n) - int(ddof), 1)
        res = multiply(res, float(total_n) / float(denom))
    return res


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
    _ = (initial, where)
    t_a = _to_tensor(a)
    cleaned = ops.where(ops.isnan(t_a), _to_tensor(float("inf")), t_a)
    return min(_wrap(cleaned), axis=axis, keepdims=keepdims)


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
    t_a = _to_tensor(a)
    cleaned = ops.where(ops.isnan(t_a), _to_tensor(float("-inf")), t_a)
    return max(_wrap(cleaned), axis=axis, keepdims=keepdims, where=where, initial=initial)


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
    _ = (dtype, initial)
    t_a = _to_tensor(a)
    cleaned = ops.where(ops.isnan(t_a), _to_tensor(0), t_a)
    return sum(_wrap(cleaned), axis=axis, keepdims=keepdims, where=where)


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
    _ = (initial, where)
    t_a = _to_tensor(a)
    cleaned = ops.where(ops.isnan(t_a), _to_tensor(1), t_a)
    return prod(_wrap(cleaned), axis=axis, dtype=dtype, keepdims=keepdims)


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


def trapezoid(y: ArrayLikeSpec, x: ArrayLikeSpec = None, dx: float = 1.0, axis: int = -1) -> ArrayLikeSpec:
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        y: Input array to integrate.
        x: Optional sample points corresponding to ``y`` values.
        dx: Spacing between sample points when ``x`` is ``None``.
        axis: Axis along which to integrate.

    Returns:
        ArrayLikeSpec: Definite integral approximated by trapezoidal rule.
    """
    from zero_jax.scipy.integrate import trapezoid as _sp_trapezoid

    return _sp_trapezoid(y, x=x, dx=dx, axis=axis)


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
    return nanmedian(a, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims)


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
    from zero_jax.numpy._indexing_set import take
    from zero_jax.numpy._math import _float_asarray

    arr = ravel(_float_asarray(a))
    edges = histogram_bin_edges(arr, bins=bins, range=range, weights=weights)
    w = ravel(asarray(weights, dtype=arr.dtype)) if weights is not None else ones_like(arr)
    n_bins = edges.shape[0] - 1
    counts: list[ArrayLikeSpec] = []
    for i in builtins.range(n_bins):
        left = take(edges, i)
        right = take(edges, i + 1)
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
    from zero_jax.numpy._indexing_set import take
    from zero_jax.numpy._math import _float_asarray

    nx, ny = (bins[0], bins[1]) if isinstance(bins, (list, tuple)) and len(bins) == 2 else (bins, bins)
    rx, ry = (range[0], range[1]) if isinstance(range, (list, tuple)) and len(range) == 2 else (None, None)
    x_arr = ravel(_float_asarray(x))
    y_arr = ravel(asarray(y, dtype=x_arr.dtype))
    xedges = histogram_bin_edges(x_arr, bins=nx, range=rx)
    yedges = histogram_bin_edges(y_arr, bins=ny, range=ry)
    w = ravel(asarray(weights, dtype=x_arr.dtype)) if weights is not None else ones_like(x_arr)
    nbx = xedges.shape[0] - 1
    nby = yedges.shape[0] - 1
    rows: list[ArrayLikeSpec] = []
    for i in builtins.range(nbx):
        xl = take(xedges, i)
        xr = take(xedges, i + 1)
        mx = (x_arr >= xl) & (x_arr <= xr) if i == nbx - 1 else (x_arr >= xl) & (x_arr < xr)
        row_counts: list[ArrayLikeSpec] = []
        for j in builtins.range(nby):
            yl = take(yedges, j)
            yr = take(yedges, j + 1)
            my = (y_arr >= yl) & (y_arr <= yr) if j == nby - 1 else (y_arr >= yl) & (y_arr < yr)
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
    from zero_jax.numpy._math import _float_asarray

    arr = _float_asarray(sample)
    if arr.ndim == 1:
        h, e = histogram(arr, bins=bins, range=range, normed=normed, weights=weights, density=density)
        return h, [e]
    if arr.shape[-1] == 2:
        h2, xe, ye = histogram2d(arr[:, 0], arr[:, 1], bins=bins, range=range, normed=normed, weights=weights, density=density)
        return h2, [xe, ye]
    h1, e1 = histogram(arr[:, 0], bins=bins, range=range, normed=normed, weights=weights, density=density)
    return h1, [e1]


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
    from zero_jax.numpy._math import _float_asarray

    arr = ravel(_float_asarray(a))
    if isinstance(bins, (list, tuple, ndarray, Tensor)):
        return asarray(bins, dtype=arr.dtype)
    low = float(range[0]) if isinstance(range, (list, tuple)) else float(min(arr).item())
    high = float(range[1]) if isinstance(range, (list, tuple)) else float(max(arr).item())
    if low == high:
        low -= 0.5
        high += 0.5
    return linspace(low, high, int(bins) + 1, dtype=arr.dtype)


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


def count_nonzero(a: ArrayLikeSpec, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for count_nonzero.

    Args:
        a: Argument a.
        axis: Argument axis.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.count_nonzero(_to_tensor(a), axis=axis))


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


from zero_jax.numpy._indexing_set import where

__all__ = [
    "all",
    "amax",
    "amin",
    "any",
    "apply_along_axis",
    "apply_over_axes",
    "argmax",
    "argmin",
    "average",
    "bincount",
    "count_nonzero",
    "cumprod",
    "cumsum",
    "cumulative_prod",
    "cumulative_sum",
    "diff",
    "digitize",
    "ediff1d",
    "gradient",
    "histogram",
    "histogram2d",
    "histogram_bin_edges",
    "histogramdd",
    "max",
    "mean",
    "median",
    "min",
    "minmax",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanprod",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
    "percentile",
    "prod",
    "ptp",
    "quantile",
    "std",
    "sum",
    "trapezoid",
    "var",
]
