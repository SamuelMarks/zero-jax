"""Initializers for neural network weights.

This module implements various initializers, such as constant, normal, uniform, and
variance scaling initializers (like Glorot/Xavier, He/Kaiming).
"""

import math
from typing import Any, Callable, Sequence, Union, Tuple
import numpy as np

# A basic type alias to represent JAX arrays for the purpose of typing here
Array = np.ndarray
KeyArray = Any  # Usually a PRNG key
Shape = Sequence[int]
Initializer = Callable[[KeyArray, Shape, Any], Array]
RealNumeric = Union[float, int]


def zeros(key: KeyArray, shape: Shape, dtype: Any = np.float64) -> Array:
    """An initializer that returns a constant array full of zeros.

    Args:
        key: PRNG key.
        shape: Shape of the array.
        dtype: Data type of the array.

    Returns:
        Array of zeros.
    """
    return np.zeros(shape, dtype=dtype)


def ones(key: KeyArray, shape: Shape, dtype: Any = np.float64) -> Array:
    """An initializer that returns a constant array full of ones.

    Args:
        key: PRNG key.
        shape: Shape of the array.
        dtype: Data type of the array.

    Returns:
        Array of ones.
    """
    return np.ones(shape, dtype=dtype)


def constant(value: RealNumeric, dtype: Any = np.float64) -> Initializer:
    """Builds an initializer that returns arrays full of a constant ``value``.

    Args:
        value: The constant value to fill the array with.
        dtype: Default data type.

    Returns:
        An initializer function.
    """

    def init(key: KeyArray, shape: Shape, dtype: Any = dtype) -> Array:
        return np.full(shape, value, dtype=dtype)

    return init


def uniform(scale: RealNumeric = 0.01, dtype: Any = np.float64) -> Initializer:
    """Builds an initializer that returns real uniformly-distributed random arrays.

    Args:
        scale: Scale of the uniform distribution ([-scale, scale)).
        dtype: Default data type.

    Returns:
        An initializer function.
    """

    def init(key: KeyArray, shape: Shape, dtype: Any = dtype) -> Array:
        # Note: In pure numpy, we just use random.uniform
        # Ideally we'd use zero_jax.random but here we use np.random for simplicity
        rng = np.random.default_rng(seed=hash(str(key)) % (2**32))
        return rng.uniform(-scale, scale, size=shape).astype(dtype)

    return init


def normal(stddev: RealNumeric = 0.01, dtype: Any = np.float64) -> Initializer:
    """Builds an initializer that returns real normally-distributed random arrays.

    Args:
        stddev: Standard deviation of the normal distribution.
        dtype: Default data type.

    Returns:
        An initializer function.
    """

    def init(key: KeyArray, shape: Shape, dtype: Any = dtype) -> Array:
        rng = np.random.default_rng(seed=hash(str(key)) % (2**32))
        return rng.normal(0, stddev, size=shape).astype(dtype)

    return init


def truncated_normal(
    stddev: RealNumeric = 0.01,
    dtype: Any = np.float64,
    lower: RealNumeric = -2.0,
    upper: RealNumeric = 2.0,
) -> Initializer:
    """Builds an initializer that returns truncated-normal random arrays.

    Args:
        stddev: Standard deviation of the normal distribution.
        dtype: Default data type.
        lower: Lower bound for truncation (in units of stddev).
        upper: Upper bound for truncation (in units of stddev).

    Returns:
        An initializer function.
    """

    def init(key: KeyArray, shape: Shape, dtype: Any = dtype) -> Array:
        rng = np.random.default_rng(seed=hash(str(key)) % (2**32))
        # Simple rejection sampling for truncated normal
        out = np.empty(shape, dtype=dtype)
        mask = np.ones(shape, dtype=bool)
        while np.any(mask):
            num_needed = np.sum(mask)
            samples = rng.normal(0, stddev, size=num_needed)
            valid = (samples >= lower * stddev) & (samples <= upper * stddev)
            out[mask] = np.where(valid, samples, out[mask])
            mask[mask] = ~valid
        return out

    return init


def _compute_fans(
    shape: Shape,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
) -> Tuple[int, int]:
    """Computes the number of input and output units for a weight shape."""
    if isinstance(in_axis, int):
        in_axis = [in_axis]
    if isinstance(out_axis, int):
        out_axis = [out_axis]

    receptive_field_size = 1
    for i, d in enumerate(shape):
        if i not in in_axis and i not in out_axis and i not in batch_axis:
            # Handle negative indices
            if (
                i - len(shape) not in in_axis
                and i - len(shape) not in out_axis
                and i - len(shape) not in batch_axis
            ):
                receptive_field_size *= d

    fan_in = receptive_field_size
    for ax in in_axis:
        fan_in *= shape[ax]

    fan_out = receptive_field_size
    for ax in out_axis:
        fan_out *= shape[ax]

    return fan_in, fan_out


def variance_scaling(
    scale: RealNumeric,
    mode: str,
    distribution: str,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """Initializer that adapts its scale to the shape of the weights tensor.

    Args:
        scale: Scaling factor.
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "truncated_normal", "normal", "uniform".
        in_axis: Axis or axes for input dimension.
        out_axis: Axis or axes for output dimension.
        batch_axis: Axis or axes for batch dimension.
        dtype: Default data type.

    Returns:
        An initializer function.
    """

    def init(key: KeyArray, shape: Shape, dtype: Any = dtype) -> Array:
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis, batch_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2.0
        else:
            raise ValueError(f"invalid mode {mode}")

        variance = scale / denominator

        if distribution == "truncated_normal":
            stddev = math.sqrt(variance) / 0.87962566103423978
            return truncated_normal(stddev, dtype)(key, shape, dtype)
        elif distribution == "normal":
            stddev = math.sqrt(variance)
            return normal(stddev, dtype)(key, shape, dtype)
        elif distribution == "uniform":
            limit = math.sqrt(3.0 * variance)
            return uniform(limit, dtype)(key, shape, dtype)
        else:
            raise ValueError(f"invalid distribution {distribution}")

    return init


def glorot_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """Builds a Glorot uniform initializer (aka Xavier uniform initializer).

    Args:
        in_axis: Axis or axes for input dimension.
        out_axis: Axis or axes for output dimension.
        batch_axis: Axis or axes for batch dimension.
        dtype: Default data type.

    Returns:
        An initializer function.
    """
    return variance_scaling(
        1.0, "fan_avg", "uniform", in_axis, out_axis, batch_axis, dtype
    )


xavier_uniform = glorot_uniform


def glorot_normal(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """Builds a Glorot normal initializer (aka Xavier normal initializer).

    Args:
        in_axis: Axis or axes for input dimension.
        out_axis: Axis or axes for output dimension.
        batch_axis: Axis or axes for batch dimension.
        dtype: Default data type.

    Returns:
        An initializer function.
    """
    return variance_scaling(
        1.0, "fan_avg", "truncated_normal", in_axis, out_axis, batch_axis, dtype
    )


xavier_normal = glorot_normal


def lecun_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """Builds a Lecun uniform initializer.

    Args:
        in_axis: Axis or axes for input dimension.
        out_axis: Axis or axes for output dimension.
        batch_axis: Axis or axes for batch dimension.
        dtype: Default data type.

    Returns:
        An initializer function.
    """
    return variance_scaling(
        1.0, "fan_in", "uniform", in_axis, out_axis, batch_axis, dtype
    )


def lecun_normal(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """Builds a Lecun normal initializer.

    Args:
        in_axis: Axis or axes for input dimension.
        out_axis: Axis or axes for output dimension.
        batch_axis: Axis or axes for batch dimension.
        dtype: Default data type.

    Returns:
        An initializer function.
    """
    return variance_scaling(
        1.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype
    )


def he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """Builds a He uniform initializer (aka Kaiming uniform initializer).

    Args:
        in_axis: Axis or axes for input dimension.
        out_axis: Axis or axes for output dimension.
        batch_axis: Axis or axes for batch dimension.
        dtype: Default data type.

    Returns:
        An initializer function.
    """
    return variance_scaling(
        2.0, "fan_in", "uniform", in_axis, out_axis, batch_axis, dtype
    )


kaiming_uniform = he_uniform


def he_normal(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """Builds a He normal initializer (aka Kaiming normal initializer).

    Args:
        in_axis: Axis or axes for input dimension.
        out_axis: Axis or axes for output dimension.
        batch_axis: Axis or axes for batch dimension.
        dtype: Default data type.

    Returns:
        An initializer function.
    """
    return variance_scaling(
        2.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype
    )


kaiming_normal = he_normal


def orthogonal(
    scale: RealNumeric = 1.0, column_axis: int = -1, dtype: Any = np.float64
) -> Initializer:
    """Builds an initializer that returns uniformly distributed orthogonal matrices.

    Args:
        scale: Scaling factor.
        column_axis: Axis that specifies the columns of the matrix.
        dtype: Default data type.

    Returns:
        An initializer function.
    """

    def init(key: KeyArray, shape: Shape, dtype: Any = dtype) -> Array:
        rng = np.random.default_rng(seed=hash(str(key)) % (2**32))

        if len(shape) < 2:
            raise ValueError("Orthogonal initializer requires at least a 2D shape.")

        n_rows = shape[column_axis]
        n_cols = np.prod(shape) // n_rows

        matrix = rng.normal(0, 1, size=(max(n_rows, n_cols), min(n_rows, n_cols)))
        q, r = np.linalg.qr(matrix)
        q *= np.sign(np.diag(r))

        if n_rows < n_cols:
            q = q.T

        q = q * scale

        # Reshape to original shape, putting column_axis back in place
        # This is a simplified version; real JAX implementation handles permutations carefully
        if column_axis == -1 or column_axis == len(shape) - 1:
            q = q.reshape(shape)
        else:
            axes = list(range(len(shape)))
            axes[column_axis], axes[-1] = axes[-1], axes[column_axis]
            transposed_shape = [shape[i] for i in axes]
            q = q.reshape(transposed_shape)
            q = np.transpose(q, axes)

        return q.astype(dtype)

    return init


def delta_orthogonal(
    scale: RealNumeric = 1.0, column_axis: int = -1, dtype: Any = np.float64
) -> Initializer:
    """Builds an initializer for delta orthogonal kernels.

    Args:
        scale: Scaling factor.
        column_axis: Axis that specifies the columns of the matrix.
        dtype: Default data type.

    Returns:
        An initializer function.
    """

    def init(key: KeyArray, shape: Shape, dtype: Any = dtype) -> Array:
        if len(shape) < 2:
            raise ValueError(
                "Delta orthogonal initializer requires at least a 2D shape."
            )

        # Delta orthogonal means identity matrix if square, or orthogonal matrix
        # inserted at the center of the spatial dimensions (assuming Conv kernels)
        out = np.zeros(shape, dtype=dtype)

        in_axis = -2 if column_axis == -1 else -1
        col_idx = column_axis if column_axis >= 0 else len(shape) + column_axis
        in_idx = in_axis if in_axis >= 0 else len(shape) + in_axis

        # Determine spatial dimensions
        spatial_shape = list(shape)
        for idx in sorted([col_idx, in_idx], reverse=True):
            spatial_shape.pop(idx)

        center = tuple(s // 2 for s in spatial_shape)

        # We need a 2D orthogonal matrix for the I/O dimensions
        [
            shape[i] for i in range(len(shape)) if i != column_axis
        ]  # Wait, this is tricky

        # Simplification: just generate an orthogonal matrix for the non-spatial dims
        # For a shape like [K, K, I, O] where column_axis is -1 (O)
        in_axis = -2 if column_axis == -1 else -1
        i_dim = shape[in_axis]
        o_dim = shape[column_axis]

        ortho_init = orthogonal(scale, -1, dtype)
        ortho_mat = ortho_init(key, (i_dim, o_dim), dtype)

        # Insert at center
        slices = []
        spatial_idx = 0
        for i, s in enumerate(shape):
            if i == col_idx or i == in_idx:
                slices.append(slice(None))
            else:
                slices.append(slice(center[spatial_idx], center[spatial_idx] + 1))
                spatial_idx += 1

        out[tuple(slices)] = ortho_mat.reshape(out[tuple(slices)].shape)
        return out

    return init
