"""Elementwise mathematical, logical, bitwise, and comparison operations for zero_jax.numpy."""

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
from zero_jax.numpy._creation import asarray, ones_like, zeros_like
from zero_jax.numpy._dtypes import float32, int32
from zero_jax.numpy._manipulation import broadcast_arrays
from zero_jax.numpy._ndarray import _to_tensor, _wrap
from zero_jax.numpy.tensor_utils import to_array as _to_array

inf = float("inf")
nan = float("nan")
pi = 3.141592653589793


def abs(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Calculate the absolute value element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.abs(_to_tensor(x)))


def absolute(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Calculates the absolute value element-wise.

    Args:
        x: Input array.

    Returns:
        An array containing the absolute value of each element in x.
    """
    return _wrap(ops.abs(_to_tensor(x)))


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


def add(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Add arguments element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


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


def angle(z: ArrayLikeSpec, deg: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """Return the angle of the complex argument.

    Args:
        z: A complex number or sequence of complex numbers.
        deg: Return angle in degrees if True, radians if False (default).

    Returns:
        The counterclockwise angle.
    """
    return _wrap(ops.angle(_to_tensor(z), deg=deg))


def arccos(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Trigonometric inverse cosine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.acos(_to_tensor(x)))


def arccosh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inverse hyperbolic cosine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.acosh(_to_tensor(x)))


def arcsin(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inverse sine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.asin(_to_tensor(x)))


def arcsinh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inverse hyperbolic sine element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.asinh(_to_tensor(x)))


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


def arctanh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inverse hyperbolic tangent element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.atanh(_to_tensor(x)))


def around(a: ArrayLikeSpec, decimals: int = 0) -> ArrayLikeSpec:
    """Evenly round to the given number of decimals.

    Args:
        a: Input data.
        decimals: Number of decimal places to round to.

    Returns:
        An array of the same type as a, containing the rounded values.
    """
    return round(a, decimals=decimals)


def array_equal(a1: ArrayLikeSpec, a2: ArrayLikeSpec, equal_nan: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """True if two arrays have the same shape and elements, False otherwise.

    Args:
        a1 (ArrayLikeSpec): Argument a1.
        a2 (ArrayLikeSpec): Argument a2.
        equal_nan (ArrayLikeSpec): Argument equal_nan.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    t1 = _to_tensor(a1)
    t2 = _to_tensor(a2)
    res = ops.logical_or(ops.equal(t1, t2), ops.logical_and(ops.isnan(t1), ops.isnan(t2))) if equal_nan else ops.equal(t1, t2)
    all_res = ops.all(res)
    return builtins.bool(t1.shape == t2.shape and (all_res.item() if isinstance(all_res, Tensor) else all_res))


def array_equiv(a1: ArrayLikeSpec, a2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns True if input arrays are shape consistent and all elements equal.

    Args:
        a1: Input array.
        a2: Input array.

    Returns:
        True if equivalent, False otherwise.
    """
    return _wrap(ops.array_equiv(_to_tensor(a1), _to_tensor(a2)))


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


def bitwise_count(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Counts the number of 1-bits in the absolute value of the input.

    Args:
        x: Input array.

    Returns:
        The number of 1-bits.
    """
    return _wrap(ops.bitwise_count(_to_tensor(x)))


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


def bitwise_right_shift(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_right_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return right_shift(x1, x2)


def bitwise_xor(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_xor.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.bitwise_xor(_to_tensor(x1), _to_tensor(x2)))


def cbrt(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for cbrt.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.cbrt(_to_tensor(x)))


def ceil(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the ceiling of the input, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.ceil(_to_tensor(x)))


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


def conj(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for conj.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.conj(_to_tensor(x)))


def conjugate(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the complex conjugate, element-wise.

    Args:
        x: Input array.

    Returns:
        The complex conjugate of x.
    """
    return conj(x)


def copysign(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for copysign.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.copysign(_to_tensor(x1), _to_tensor(x2)))


def cos(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the trigonometric cosine element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.cos(_to_tensor(x)))


def cosh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Hyperbolic cosine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.cosh(_to_tensor(x)))


def deg2rad(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for deg2rad.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.deg2rad(_to_tensor(x)))


def degrees(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert angles from radians to degrees.

    Args:
        x: Input array in radians.

    Returns:
        The corresponding degree values.
    """
    return rad2deg(x)


def divide(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Divide arguments element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.divide(_to_tensor(x), _to_tensor(y)))


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


def equal(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.equal(_to_tensor(x1), _to_tensor(x2)))


def exp(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Calculate the exponential of all elements in the input array.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.exp(_to_tensor(x)))


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


def fabs(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the absolute values element-wise.

    Args:
        x: Input array.

    Returns:
        The absolute values.
    """
    return _wrap(ops.fabs(_to_tensor(x)))


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


def floor(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the floor of the input, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.floor(_to_tensor(x)))


def floor_divide(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the largest integer smaller or equal to the division of the inputs.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.floor_divide(_to_tensor(x), _to_tensor(y)))


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


def gcd(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for gcd.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.gcd(_to_tensor(x1), _to_tensor(x2)))


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


def _float_asarray(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert ``x`` to an ``ndarray`` preserving floating-point precision.

    Args:
        x: Input array-like value.

    Returns:
        ArrayLikeSpec: Floating-point array.
    """
    raw = asarray(x)
    dt_str = str(raw.dtype).lower()
    if "float" in dt_str or "bfloat" in dt_str:
        return raw
    return asarray(raw, dtype=config.default_float_dtype)


def i0(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Modified Bessel function of the first kind, order 0.

    Args:
        x: Argument of the Bessel function.

    Returns:
        ArrayLikeSpec: The modified Bessel function evaluated at each element of ``x``.
    """
    arr = _float_asarray(x)
    return exp(abs(arr)) * _wrap(ops.bessel_i0e(_to_tensor(arr)))


def imag(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for imag.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.imag(_to_tensor(x)))


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
    _ = period
    from zero_jax.numpy._indexing_set import searchsorted, take, where

    x_arr = _float_asarray(x)
    xp_arr = asarray(xp, dtype=x_arr.dtype)
    fp_arr = asarray(fp, dtype=x_arr.dtype)
    n = int(xp_arr.shape[0])
    idx = clip(searchsorted(xp_arr, x_arr, side="right") - 1, 0, max(n - 2, 0))
    idx_int = asarray(idx, dtype=int32)
    x0 = take(xp_arr, idx_int)
    x1 = take(xp_arr, idx_int + 1) if n > 1 else x0
    y0 = take(fp_arr, idx_int)
    y1 = take(fp_arr, idx_int + 1) if n > 1 else y0
    denom = x1 - x0
    slope = (y1 - y0) / where(denom == 0.0, 1.0, denom)
    res = y0 + slope * (x_arr - x0)
    xp_first = take(xp_arr, 0)
    xp_last = take(xp_arr, max(n - 1, 0))
    left_val = take(fp_arr, 0) if left is None else left
    right_val = take(fp_arr, max(n - 1, 0)) if right is None else right
    res = where(x_arr < xp_first, left_val, res)
    return where(x_arr > xp_last, right_val, res)


def invert(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for invert.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return bitwise_not(x)


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


def isfinite(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Test element-wise for finiteness (not infinity or not Not a Number).

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.isfinite(_to_tensor(x)))


def isinf(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for isinf.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.isinf(_to_tensor(x)))


def isnan(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Test element-wise for NaN and return result as a boolean array.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.isnan(_to_tensor(x)))


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


def log(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Natural logarithm, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.log(_to_tensor(x)))


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


def mod(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return element-wise remainder of division.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.mod(_to_tensor(x), _to_tensor(y)))


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


def multiply(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Multiply arguments element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


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


def negative(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Numerical negative, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.negative(_to_tensor(x)))


def nextafter(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for nextafter.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.nextafter(*[_to_tensor(t) for t in broadcast_arrays(x1, x2)]))


def not_equal(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for not_equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.not_equal(_to_tensor(x1), _to_tensor(x2)))


def positive(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Numerical positive, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.positive(_to_tensor(x)))


def pow(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """First array elements raised to powers from second array, element-wise.

    Args:
        x1: The bases.
        x2: The exponents.

    Returns:
        The bases in x1 raised to the exponents in x2.
    """
    return power(x1, x2)


def power(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """First array elements raised to powers from second array, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.power(_to_tensor(x), _to_tensor(y)))


def rad2deg(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for rad2deg.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.rad2deg(_to_tensor(x)))


def radians(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert angles from degrees to radians.

    Args:
        x: Input array in degrees.

    Returns:
        The corresponding radian values.
    """
    return deg2rad(x)


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


def remainder(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return element-wise remainder of division.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.remainder(_to_tensor(x), _to_tensor(y)))


def right_shift(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for right_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.right_shift(_to_tensor(x1), _to_tensor(x2)))


def rint(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Round elements of the array to the nearest integer.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.round(_to_tensor(x)))


def round(a: ArrayLikeSpec, decimals: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for round.

    Args:
        a: Argument a.
        decimals: Argument decimals.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.round(_to_tensor(a), decimals=decimals))


def round_(a: ArrayLikeSpec, decimals: int = 0) -> ArrayLikeSpec:
    """Evenly round to the given number of decimals.

    Args:
        a: Input data.
        decimals: Number of decimal places to round to.

    Returns:
        An array of the same type as a, containing the rounded values.
    """
    return round(a, decimals=decimals)


def sign(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns an element-wise indication of the sign of a number.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.sign(_to_tensor(x)))


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


def sin(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the trigonometric sine element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.sin(_to_tensor(x)))


def sinc(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for sinc.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.sinc(_to_tensor(x)))


def sinh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Hyperbolic sine, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.sinh(_to_tensor(x)))


def spacing(x: ArrayLikeSpec, /) -> ArrayLikeSpec:
    """Return the distance between ``x`` and the nearest adjacent number.

    Args:
        x: Input array or scalar.

    Returns:
        ArrayLikeSpec: Elementwise ULPs (spacing) of ``x``.
    """
    arr = _float_asarray(x)
    return abs(nextafter(arr, float("inf")) - arr)


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


def subtract(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Subtract arguments, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.subtract(_to_tensor(x), _to_tensor(y)))


def tan(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute tangent element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.tan(_to_tensor(x)))


def tanh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute hyperbolic tangent element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.tanh(_to_tensor(x)))


def true_divide(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Divide arguments element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return divide(x, y)


def trunc(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the truncated value of the input, element-wise.

    Args:
        x (ArrayLikeSpec): Argument x.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.trunc(_to_tensor(x)))


def ufunc() -> None:
    """Execute ``ufunc``."""


def unwrap(p: ArrayLikeSpec, discont: ArrayLikeSpec = None, axis: int = -1, period: float = 6.283185307179586) -> ArrayLikeSpec:
    """Unwrap radian phase ``p`` by changing absolute jumps greater than ``discont`` to their ``2*pi`` complement.

    Args:
        p: Input phase array.
        discont: Maximum discontinuity between values, default ``period / 2``.
        axis: Axis along which unwrap will operate.
        period: Size of the range over which the input wraps.

    Returns:
        ArrayLikeSpec: Unwrapped phase array.
    """
    from zero_jax.numpy._indexing_set import where
    from zero_jax.numpy._manipulation import concatenate
    from zero_jax.numpy._reductions import cumsum, diff

    arr = _float_asarray(p)
    disc = period / 2.0 if discont is None else float(discont)
    dd = diff(arr, axis=axis)
    ddmod = mod(dd + period / 2.0, period) - period / 2.0
    ddmod = where((ddmod == -period / 2.0) & (dd > 0.0), period / 2.0, ddmod)
    ph_correct = where(abs(dd) < disc, 0.0, ddmod - dd)
    norm_ax = axis if axis >= 0 else arr.ndim + axis
    pad_shape = (*arr.shape[:norm_ax], 1, *arr.shape[norm_ax + 1 :])
    zeros_pad = zeros_like(arr)[:1] if arr.ndim == 1 else asarray(0.0, dtype=arr.dtype) * ones_like(arr[tuple(slice(0, 1) if i == norm_ax else slice(None) for i in range(arr.ndim))])
    _ = pad_shape
    return arr + concatenate([zeros_pad, cumsum(ph_correct, axis=axis)], axis=norm_ax)


__all__ = [
    "_float_asarray",
    "abs",
    "absolute",
    "acos",
    "acosh",
    "add",
    "allclose",
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "around",
    "array_equal",
    "array_equiv",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_count",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "cbrt",
    "ceil",
    "clip",
    "conj",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "divide",
    "divmod",
    "equal",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "fix",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "i0",
    "imag",
    "inf",
    "interp",
    "invert",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "nan",
    "nan_to_num",
    "negative",
    "nextafter",
    "not_equal",
    "pi",
    "positive",
    "pow",
    "power",
    "rad2deg",
    "radians",
    "real",
    "reciprocal",
    "remainder",
    "right_shift",
    "rint",
    "round",
    "round_",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
    "ufunc",
    "unwrap",
]
