"""Initializers for neural network weights."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Union

from ml_switcheroo_compiler import random
from ml_switcheroo_compiler.core.dtype import DType

import zero_jax._compiler_proxy_creation as creation
from zero_jax._types import ArrayLikeSpec, RealScalar
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
from zero_jax.numpy.lax_numpy import ndarray as Array

KeyArray = ArrayLikeSpec
Shape = Sequence[int]
Initializer = Callable[[KeyArray, Shape, ArrayLikeSpec], Array]
RealNumeric = "float | int"


def _to_dtype(dtype: ArrayLikeSpec) -> DType:
    """Convert a loosely specified dtype to a formal ``DType`` object.

    Args:
        dtype: Type to convert (string, Python type, ``NamedProtocol``, or ``DType``).

    Returns:
        DType: The corresponding parsed ``DType``.
    """
    from zero_jax._types import NamedProtocol

    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, str):
        return DType(dtype)
    if dtype is float:
        return DType("float64")
    if dtype is int:
        return DType("int64")
    if dtype is bool:
        return DType("bool")
    if dtype is complex:
        return DType("complex64")
    if isinstance(dtype, type):
        return DType(dtype.__name__)
    if isinstance(dtype, NamedProtocol):
        return DType(dtype.name)
    return DType(str(dtype))


def zeros(
    key: KeyArray,
    shape: Shape,
    dtype: ArrayLikeSpec = float,
    out_sharding: ArrayLikeSpec = None,
) -> Array:
    """Initialize an array with all zeros.

    Args:
        key: PRNG key (unused, kept for API parity).
        shape: Desired shape of the output array.
        dtype: Desired data type.
        out_sharding: Optional output sharding specification.

    Returns:
        Array: An array of zeros with the specified shape and dtype.
    """
    _ = (key, out_sharding)
    return _wrap(creation.zeros(shape=shape, dtype=_to_dtype(dtype)))


def ones(
    key: KeyArray,
    shape: Shape,
    dtype: ArrayLikeSpec = float,
    out_sharding: ArrayLikeSpec = None,
) -> Array:
    """Initialize an array with all ones.

    Args:
        key: PRNG key (unused, kept for API parity).
        shape: Desired shape of the output array.
        dtype: Desired data type.
        out_sharding: Optional output sharding specification.

    Returns:
        Array: An array of ones with the specified shape and dtype.
    """
    _ = (key, out_sharding)
    return _wrap(creation.ones(shape=shape, dtype=_to_dtype(dtype)))


def constant(value: RealScalar, dtype: ArrayLikeSpec = float) -> Initializer:
    """Return an initializer that generates arrays filled with a constant value.

    Args:
        value: The constant scalar value to fill the array with.
        dtype: Default data type.

    Returns:
        Initializer: A function that generates constant-filled arrays.
    """

    def init(key: ArrayLikeSpec, shape: ArrayLikeSpec, dtype: ArrayLikeSpec = dtype) -> ArrayLikeSpec:
        """Initialize an array filled with ``value``.

        Args:
            key: PRNG key (unused).
            shape: Shape of the output array.
            dtype: Data type override.

        Returns:
            ArrayLikeSpec: A constant-filled array.
        """
        return _wrap(creation.full(shape=shape, fill_value=value, dtype=_to_dtype(dtype)))

    return init


def uniform(scale: RealScalar = 0.01, dtype: ArrayLikeSpec = float) -> Initializer:
    """Return an initializer that samples from a uniform distribution ``[0, scale)``.

    Args:
        scale: Upper bound for the uniform distribution.
        dtype: Default data type of the generated array.

    Returns:
        Initializer: A function that initializes uniform arrays.
    """

    def init(key: ArrayLikeSpec, shape: ArrayLikeSpec, dtype: ArrayLikeSpec = dtype) -> ArrayLikeSpec:
        """Initialize an array with values from a uniform distribution.

        Args:
            key: PRNG key used for generating uniform values.
            shape: Shape of the array.
            dtype: Data type override.

        Returns:
            ArrayLikeSpec: An array of uniformly distributed values.
        """
        dt = _to_dtype(dtype)
        u = random.uniform(_to_tensor(key), shape=shape, dtype=dt, minval=-scale, maxval=scale)
        return _wrap(u)

    return init


def normal(stddev: RealScalar = 0.01, dtype: ArrayLikeSpec = float) -> Initializer:
    """Return an initializer that samples from a zero-mean normal distribution.

    Args:
        stddev: Standard deviation of the normal distribution.
        dtype: Default data type of the generated array.

    Returns:
        Initializer: A function that initializes normal-distributed arrays.
    """

    def init(key: ArrayLikeSpec, shape: ArrayLikeSpec, dtype: ArrayLikeSpec = dtype) -> ArrayLikeSpec:
        """Initialize an array with values from a normal distribution.

        Args:
            key: PRNG key used for generating normal values.
            shape: Shape of the array.
            dtype: Data type override.

        Returns:
            ArrayLikeSpec: An array of normally distributed values.
        """
        dt = _to_dtype(dtype)
        n = random.normal(_to_tensor(key), shape=shape, dtype=dt)
        import zero_jax._compiler_proxy_ops as ops

        return _wrap(ops.multiply(n, creation.full_like(n, stddev)))

    return init


def truncated_normal(
    stddev: RealScalar = 0.01,
    dtype: ArrayLikeSpec = float,
    lower: RealScalar = -2.0,
    upper: RealScalar = 2.0,
) -> Initializer:
    """Return an initializer that samples from a truncated normal distribution.

    Args:
        stddev: Standard deviation of the normal distribution before truncation.
        dtype: Default data type of the generated array.
        lower: Lower bound for truncation in standard deviations.
        upper: Upper bound for truncation in standard deviations.

    Returns:
        Initializer: A function that initializes truncated-normal arrays.
    """

    def init(key: ArrayLikeSpec, shape: ArrayLikeSpec, dtype: ArrayLikeSpec = dtype) -> ArrayLikeSpec:
        """Initialize an array with values from a truncated normal distribution.

        Args:
            key: PRNG key used for generating truncated normal values.
            shape: Shape of the array.
            dtype: Data type override.

        Returns:
            ArrayLikeSpec: An array of truncated normally distributed values.
        """
        dt = _to_dtype(dtype)
        n = random.truncated_normal(_to_tensor(key), lower, upper, shape=shape, dtype=dt)
        import zero_jax._compiler_proxy_ops as ops

        return _wrap(ops.multiply(n, creation.full_like(n, stddev)))

    return init


def _compute_fans(
    shape: Shape,
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
) -> tuple[int, int]:
    """Compute the fan-in and fan-out sizes for a weight tensor shape.

    Args:
        shape: Shape of the weight tensor.
        in_axis: Axis or axes for input feature dimensions.
        out_axis: Axis or axes for output feature dimensions.
        batch_axis: Axis or axes for batch dimensions.

    Returns:
        tuple[int, int]: ``(fan_in, fan_out)`` sizes.
    """
    rank = len(shape)
    if rank < 1:
        return 1, 1
    if rank == 1:
        return int(shape[0]), int(shape[0])

    in_axes = (in_axis,) if isinstance(in_axis, int) else tuple(in_axis)
    out_axes = (out_axis,) if isinstance(out_axis, int) else tuple(out_axis)
    batch_axes = (batch_axis,) if isinstance(batch_axis, int) else tuple(batch_axis)

    norm_in = tuple(ax % rank for ax in in_axes)
    norm_out = tuple(ax % rank for ax in out_axes)
    norm_batch = tuple(ax % rank for ax in batch_axes)

    in_size = math.prod(int(shape[i]) for i in norm_in)
    out_size = math.prod(int(shape[i]) for i in norm_out)
    batch_size = math.prod(int(shape[i]) for i in norm_batch)
    total_size = math.prod(int(d) for d in shape)
    receptive_field_size = max(1, total_size // max(1, in_size * out_size * batch_size))

    fan_in = int(in_size * receptive_field_size)
    fan_out = int(out_size * receptive_field_size)
    return fan_in, fan_out


def variance_scaling(
    scale: RealScalar,
    mode: str,
    distribution: str,
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: Sequence[int] = (),
    dtype: ArrayLikeSpec = float,
) -> Initializer:
    """Return an initializer that adapts its scale to the shape of the weight tensor.

    Args:
        scale: Scaling factor (positive float).
        mode: One of ``"fan_in"``, ``"fan_out"``, or ``"fan_avg"``.
        distribution: One of ``"truncated_normal"``, ``"normal"``, or ``"uniform"``.
        in_axis: Axis or axes corresponding to input dimensions.
        out_axis: Axis or axes corresponding to output dimensions.
        batch_axis: Axis or axes corresponding to batch dimensions.
        dtype: Default data type of the generated array.

    Returns:
        Initializer: A function that initializes variance-scaled arrays.
    """

    def init(key: ArrayLikeSpec, shape: ArrayLikeSpec, dtype: ArrayLikeSpec = dtype) -> ArrayLikeSpec:
        """Initialize a variance-scaled weight array.

        Args:
            key: PRNG key.
            shape: Shape of the weight array.
            dtype: Data type override.

        Returns:
            ArrayLikeSpec: Initialized weight array.

        Raises:
            ValueError: If ``mode`` or ``distribution`` is invalid.
        """
        shape_tuple = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)
        fan_in, fan_out = _compute_fans(shape_tuple, in_axis=in_axis, out_axis=out_axis, batch_axis=batch_axis)
        if mode == "fan_in":
            denominator = float(max(1, fan_in))
        elif mode == "fan_out":
            denominator = float(max(1, fan_out))
        elif mode == "fan_avg":
            denominator = float(max(1, fan_in + fan_out)) / 2.0
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")

        variance = float(scale) / denominator
        if distribution == "truncated_normal":
            stddev = math.sqrt(variance) / 0.87962566103423978
            return truncated_normal(stddev=stddev, dtype=dtype)(key, shape_tuple, dtype)
        if distribution == "normal":
            return normal(stddev=math.sqrt(variance), dtype=dtype)(key, shape_tuple, dtype)
        if distribution == "uniform":
            return uniform(scale=math.sqrt(3.0 * variance), dtype=dtype)(key, shape_tuple, dtype)
        raise ValueError(f"invalid distribution for variance scaling initializer: {distribution}")

    return init


def glorot_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: Sequence[int] = (),
    dtype: ArrayLikeSpec = float,
) -> Initializer:
    """Return a Glorot (Xavier) uniform initializer.

    Args:
        in_axis: Input dimension axis or axes.
        out_axis: Output dimension axis or axes.
        batch_axis: Batch dimension axis or axes.
        dtype: Default data type.

    Returns:
        Initializer: A Glorot uniform initializer function.
    """
    return variance_scaling(1.0, "fan_avg", "uniform", in_axis, out_axis, batch_axis, dtype)


xavier_uniform = glorot_uniform


def glorot_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: Sequence[int] = (),
    dtype: ArrayLikeSpec = float,
) -> Initializer:
    """Return a Glorot (Xavier) normal initializer.

    Args:
        in_axis: Input dimension axis or axes.
        out_axis: Output dimension axis or axes.
        batch_axis: Batch dimension axis or axes.
        dtype: Default data type.

    Returns:
        Initializer: A Glorot normal initializer function.
    """
    return variance_scaling(1.0, "fan_avg", "truncated_normal", in_axis, out_axis, batch_axis, dtype)


xavier_normal = glorot_normal


def lecun_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: Sequence[int] = (),
    dtype: ArrayLikeSpec = float,
) -> Initializer:
    """Return a LeCun uniform initializer.

    Args:
        in_axis: Input dimension axis or axes.
        out_axis: Output dimension axis or axes.
        batch_axis: Batch dimension axis or axes.
        dtype: Default data type.

    Returns:
        Initializer: A LeCun uniform initializer function.
    """
    return variance_scaling(1.0, "fan_in", "uniform", in_axis, out_axis, batch_axis, dtype)


def lecun_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: Sequence[int] = (),
    dtype: ArrayLikeSpec = float,
) -> Initializer:
    """Return a LeCun normal initializer.

    Args:
        in_axis: Input dimension axis or axes.
        out_axis: Output dimension axis or axes.
        batch_axis: Batch dimension axis or axes.
        dtype: Default data type.

    Returns:
        Initializer: A LeCun normal initializer function.
    """
    return variance_scaling(1.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype)


def he_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: Sequence[int] = (),
    dtype: ArrayLikeSpec = float,
) -> Initializer:
    """Return a He (Kaiming) uniform initializer.

    Args:
        in_axis: Input dimension axis or axes.
        out_axis: Output dimension axis or axes.
        batch_axis: Batch dimension axis or axes.
        dtype: Default data type.

    Returns:
        Initializer: A He uniform initializer function.
    """
    return variance_scaling(2.0, "fan_in", "uniform", in_axis, out_axis, batch_axis, dtype)


kaiming_uniform = he_uniform


def he_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: Sequence[int] = (),
    dtype: ArrayLikeSpec = float,
) -> Initializer:
    """Return a He (Kaiming) normal initializer.

    Args:
        in_axis: Input dimension axis or axes.
        out_axis: Output dimension axis or axes.
        batch_axis: Batch dimension axis or axes.
        dtype: Default data type.

    Returns:
        Initializer: A He normal initializer function.
    """
    return variance_scaling(2.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype)


kaiming_normal = he_normal


def orthogonal(scale: RealScalar = 1.0, column_axis: int = -1, dtype: ArrayLikeSpec = float) -> Initializer:
    """Return an initializer that generates orthogonal or semi-orthogonal matrices via QR decomposition.

    Args:
        scale: Multiplicative scale factor applied to the orthogonal matrix.
        column_axis: Axis containing the columns that should be orthogonal.
        dtype: Default data type of the array.

    Returns:
        Initializer: An orthogonal initializer function.
    """

    def init(key: ArrayLikeSpec, shape: ArrayLikeSpec, dtype: ArrayLikeSpec = dtype) -> ArrayLikeSpec:
        """Initialize an orthogonally constrained array using QR factorization.

        Args:
            key: PRNG key.
            shape: Shape of the output array (must have rank >= 2).
            dtype: Data type override.

        Returns:
            ArrayLikeSpec: An orthogonally initialized array.

        Raises:
            ValueError: If ``len(shape) < 2``.
        """
        import zero_jax._compiler_proxy_ops as ops
        from zero_jax.numpy import diagonal, moveaxis, reshape, sign, transpose
        from zero_jax.numpy.linalg import qr

        shape_tuple = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)
        if len(shape_tuple) < 2:
            raise ValueError(f"orthogonal initializer requires at least a 2D shape, got {shape_tuple}")

        dt = _to_dtype(dtype)
        col_ax = column_axis % len(shape_tuple)
        n_cols = shape_tuple[col_ax]
        rem_dims = [shape_tuple[i] for i in range(len(shape_tuple)) if i != col_ax]
        n_rows = math.prod(rem_dims)

        flat_shape = (max(n_rows, n_cols), min(n_rows, n_cols))
        a = _wrap(random.normal(_to_tensor(key), shape=flat_shape, dtype=dt))
        q_raw, r_raw = qr(a)
        q = _wrap(q_raw)
        r = _wrap(r_raw)
        d = sign(diagonal(r))
        q_signed = q * d
        if n_rows < n_cols:
            q_signed = transpose(q_signed)

        reshaped = reshape(q_signed, (*tuple(rem_dims), n_cols))
        moved = moveaxis(reshaped, -1, col_ax)
        t_moved = _to_tensor(moved)
        return _wrap(ops.multiply(t_moved, creation.full_like(t_moved, float(scale))))

    return init


def delta_orthogonal(scale: RealScalar = 1.0, column_axis: int = -1, dtype: ArrayLikeSpec = float) -> Initializer:
    """Return an initializer that generates delta-orthogonal convolutional kernels.

    Args:
        scale: Multiplicative scale factor applied to the central orthogonal matrix.
        column_axis: Output channel axis (defaults to ``-1``).
        dtype: Default data type of the kernel.

    Returns:
        Initializer: A delta-orthogonal initializer function.
    """

    def init(key: ArrayLikeSpec, shape: ArrayLikeSpec, dtype: ArrayLikeSpec = dtype) -> ArrayLikeSpec:
        """Initialize a delta-orthogonal convolutional kernel (rank 3, 4, or 5).

        Args:
            key: PRNG key.
            shape: Shape of the convolutional kernel (rank 3, 4, or 5).
            dtype: Data type override.

        Returns:
            ArrayLikeSpec: A delta-orthogonally initialized kernel.

        Raises:
            ValueError: If ``len(shape)`` is not in ``{3, 4, 5}``.
        """
        shape_tuple = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)
        if len(shape_tuple) not in (3, 4, 5):
            raise ValueError(f"delta_orthogonal requires a 3D, 4D, or 5D shape, got {shape_tuple}")

        orth_matrix = orthogonal(scale=scale, column_axis=-1, dtype=dtype)(key, shape_tuple[-2:], dtype)
        kernel = zeros(key, shape_tuple, dtype=dtype)
        mid = tuple(d // 2 for d in shape_tuple[:-2])
        idx = (*mid, slice(None), slice(None))
        return kernel.at[idx].set(orth_matrix)

    return init


__all__ = [
    "Array",
    "Callable",
    "DType",
    "Initializer",
    "KeyArray",
    "RealNumeric",
    "Sequence",
    "Shape",
    "Union",
    "_compute_fans",
    "_to_dtype",
    "constant",
    "delta_orthogonal",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "lecun_normal",
    "lecun_uniform",
    "normal",
    "ones",
    "orthogonal",
    "truncated_normal",
    "uniform",
    "variance_scaling",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
]
