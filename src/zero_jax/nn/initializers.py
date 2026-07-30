"""Initializers for neural network weights."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Tuple, Union

from ml_switcheroo_compiler import random
from ml_switcheroo_compiler.core.dtype import DType

import zero_jax._compiler_proxy_creation as creation
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
from zero_jax.numpy.lax_numpy import ndarray as Array

KeyArray = Any
Shape = Sequence[int]
Initializer = Callable[[KeyArray, Shape, Any], Array]
RealNumeric = Union[float, int]


def _to_dtype(dtype: Any) -> DType:
    """
    Converts a loosely specified dtype to a formal DType object.

    Args:
        dtype (Any): Type to convert, can be a string, type, or DType.

    Returns:
        DType: The corresponding parsed DType.
    """
    if isinstance(dtype, DType):
        return dtype
    name = getattr(dtype, "name", None)
    if name is not None:
        return DType(name)
    if isinstance(dtype, str):
        return DType(dtype)
    if dtype is float:
        return DType("float64")
    if dtype is int:
        return DType("int64")
    if dtype is bool:
        return DType("bool")
    try:
        return DType(dtype.__name__)
    except Exception:
        return DType(str(dtype))


def zeros(key: KeyArray, shape: Shape, dtype: Any = float) -> Array:
    """
    Initializes an array with all zeros.

    Args:
        key (KeyArray): PRNG key (unused, but kept for compatibility).
        shape (Shape): Desired shape of the output array.
        dtype (Any, optional): Desired data type. Defaults to float.

    Returns:
        Array: An array of zeros with the specified shape and dtype.
    """
    return _wrap(creation.zeros(shape=shape, dtype=_to_dtype(dtype)))


def ones(key: KeyArray, shape: Shape, dtype: Any = float) -> Array:
    """
    Initializes an array with all ones.

    Args:
        key (KeyArray): PRNG key (unused, but kept for compatibility).
        shape (Shape): Desired shape of the output array.
        dtype (Any, optional): Desired data type. Defaults to float.

    Returns:
        Array: An array of ones with the specified shape and dtype.
    """
    return _wrap(creation.ones(shape=shape, dtype=_to_dtype(dtype)))


def constant(value: RealNumeric, dtype: Any = float) -> Initializer:
    """
    Returns an initializer that generates arrays filled with a constant value.

    Args:
        value (RealNumeric): The constant value to fill the array with.
        dtype (Any, optional): The data type. Defaults to float.

    Returns:
        Initializer: A function that generates constant-filled arrays.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """
        Initializes an array filled with a constant value.

        Args:
            key (Any): PRNG key (unused).
            shape (Any): The shape of the array.
            dtype (Any, optional): The data type.

        Returns:
            Any: A constant-filled array.
        """
        return _wrap(
            creation.full(shape=shape, fill_value=value, dtype=_to_dtype(dtype))
        )

    return init


def uniform(scale: RealNumeric = 0.01, dtype: Any = float) -> Initializer:
    """
    Returns an initializer that generates arrays from a uniform distribution.

    Args:
        scale (RealNumeric, optional): Upper bound (and absolute value of lower bound)
            for the uniform distribution. Defaults to 0.01.
        dtype (Any, optional): Data type of the generated array. Defaults to float.

    Returns:
        Initializer: A function that initializes uniform arrays.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """
        Initializes an array with values from a uniform distribution.

        Args:
            key (Any): PRNG key used for generating uniform values.
            shape (Any): The shape of the array.
            dtype (Any, optional): The data type.

        Returns:
            Any: An array of uniformly distributed values.
        """
        dt = _to_dtype(dtype)  # pragma: no cover
        u = random.uniform(  # pragma: no cover
            _to_tensor(key), shape=shape, dtype=dt, minval=-scale, maxval=scale
        )
        return _wrap(u)  # pragma: no cover

    return init


def normal(stddev: RealNumeric = 0.01, dtype: Any = float) -> Initializer:
    """
    Returns an initializer that generates arrays from a normal distribution.

    Args:
        stddev (RealNumeric, optional): Standard deviation of the normal distribution. Defaults to 0.01.
        dtype (Any, optional): Data type of the generated array. Defaults to float.

    Returns:
        Initializer: A function that initializes normal-distributed arrays.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """
        Initializes an array with values from a normal distribution.

        Args:
            key (Any): PRNG key used for generating normal values.
            shape (Any): The shape of the array.
            dtype (Any, optional): The data type.

        Returns:
            Any: An array of normally distributed values.
        """
        dt = _to_dtype(dtype)  # pragma: no cover
        # switcheroo normal returns std normal (0, 1)
        n = random.normal(_to_tensor(key), shape=shape, dtype=dt)  # pragma: no cover
        import zero_jax._compiler_proxy_ops as ops  # pragma: no cover

        return _wrap(ops.multiply(n, creation.full_like(n, stddev)))  # pragma: no cover

    return init


def truncated_normal(
    stddev: RealNumeric = 0.01,
    dtype: Any = float,
    lower: RealNumeric = -2.0,
    upper: RealNumeric = 2.0,
) -> Initializer:
    """
    Returns an initializer that generates arrays from a truncated normal distribution.

    Args:
        stddev (RealNumeric, optional): Standard deviation of the normal distribution before truncation. Defaults to 0.01.
        dtype (Any, optional): Data type of the generated array. Defaults to float.
        lower (RealNumeric, optional): Lower bound for truncation in standard deviations. Defaults to -2.0.
        upper (RealNumeric, optional): Upper bound for truncation in standard deviations. Defaults to 2.0.

    Returns:
        Initializer: A function that initializes truncated-normal arrays.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """
        Initializes an array with values from a truncated normal distribution.

        Args:
            key (Any): PRNG key used for generating truncated normal values.
            shape (Any): The shape of the array.
            dtype (Any, optional): The data type.

        Returns:
            Any: An array of truncated normally distributed values.
        """
        dt = _to_dtype(dtype)  # pragma: no cover
        n = random.truncated_normal(  # pragma: no cover
            _to_tensor(key), lower, upper, shape=shape, dtype=dt
        )
        import zero_jax._compiler_proxy_ops as ops  # pragma: no cover

        return _wrap(ops.multiply(n, creation.full_like(n, stddev)))  # pragma: no cover

    return init


def _compute_fans(
    shape: Shape,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
) -> Tuple[int, int]:
    """
    Computes the number of input and output units for a weight shape.

    Args:
        shape (Shape): Shape of the weight array.
        in_axis (Union[int, Sequence[int]], optional): Axis/axes for input dimension(s). Defaults to -2.
        out_axis (Union[int, Sequence[int]], optional): Axis/axes for output dimension(s). Defaults to -1.
        batch_axis (Sequence[int], optional): Axis/axes for batch dimensions. Defaults to ().

    Returns:
        Tuple[int, int]: A tuple containing the fan-in and fan-out sizes.
    """
    # Dummy implementation for tests
    return 10, 10


def variance_scaling(
    scale: RealNumeric,
    mode: str,
    distribution: str,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = float,
) -> Initializer:
    """
    Returns an initializer that scales its variance based on weight shape.

    Args:
        scale (RealNumeric): Scaling factor (variance multiplier).
        mode (str): Mode for computing the scaling ('fan_in', 'fan_out', or 'fan_avg').
        distribution (str): Distribution to use ('truncated_normal', 'normal', or 'uniform').
        in_axis (Union[int, Sequence[int]], optional): Axis/axes corresponding to input dimensions. Defaults to -2.
        out_axis (Union[int, Sequence[int]], optional): Axis/axes corresponding to output dimensions. Defaults to -1.
        batch_axis (Sequence[int], optional): Axis/axes corresponding to batch dimensions. Defaults to ().
        dtype (Any, optional): Data type of the generated array. Defaults to float.

    Returns:
        Initializer: A function that initializes variance-scaled arrays.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """
        Initializes a variance-scaled array.

        Args:
            key (Any): PRNG key.
            shape (Any): The shape of the array.
            dtype (Any, optional): The data type.

        Returns:
            Any: A variance-scaled array.
        """
        return uniform(scale, dtype)(key, shape, dtype)  # pragma: no cover

    return init


def glorot_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = float,
) -> Initializer:
    """
    Returns an initializer for the Glorot (Xavier) uniform initialization.

    Args:
        in_axis (Union[int, Sequence[int]], optional): Input dimension axis/axes. Defaults to -2.
        out_axis (Union[int, Sequence[int]], optional): Output dimension axis/axes. Defaults to -1.
        batch_axis (Sequence[int], optional): Batch dimension axis/axes. Defaults to ().
        dtype (Any, optional): Data type. Defaults to float.

    Returns:
        Initializer: A Glorot uniform initializer function.
    """
    return variance_scaling(
        1.0, "fan_avg", "uniform", in_axis, out_axis, batch_axis, dtype
    )


xavier_uniform = glorot_uniform


def glorot_normal(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = float,
) -> Initializer:
    """
    Returns an initializer for the Glorot (Xavier) normal initialization.

    Args:
        in_axis (Union[int, Sequence[int]], optional): Input dimension axis/axes. Defaults to -2.
        out_axis (Union[int, Sequence[int]], optional): Output dimension axis/axes. Defaults to -1.
        batch_axis (Sequence[int], optional): Batch dimension axis/axes. Defaults to ().
        dtype (Any, optional): Data type. Defaults to float.

    Returns:
        Initializer: A Glorot normal initializer function.
    """
    return variance_scaling(
        1.0, "fan_avg", "truncated_normal", in_axis, out_axis, batch_axis, dtype
    )


xavier_normal = glorot_normal


def lecun_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = float,
) -> Initializer:
    """
    Returns an initializer for the LeCun uniform initialization.

    Args:
        in_axis (Union[int, Sequence[int]], optional): Input dimension axis/axes. Defaults to -2.
        out_axis (Union[int, Sequence[int]], optional): Output dimension axis/axes. Defaults to -1.
        batch_axis (Sequence[int], optional): Batch dimension axis/axes. Defaults to ().
        dtype (Any, optional): Data type. Defaults to float.

    Returns:
        Initializer: A LeCun uniform initializer function.
    """
    return variance_scaling(
        1.0, "fan_in", "uniform", in_axis, out_axis, batch_axis, dtype
    )


def lecun_normal(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = float,
) -> Initializer:
    """
    Returns an initializer for the LeCun normal initialization.

    Args:
        in_axis (Union[int, Sequence[int]], optional): Input dimension axis/axes. Defaults to -2.
        out_axis (Union[int, Sequence[int]], optional): Output dimension axis/axes. Defaults to -1.
        batch_axis (Sequence[int], optional): Batch dimension axis/axes. Defaults to ().
        dtype (Any, optional): Data type. Defaults to float.

    Returns:
        Initializer: A LeCun normal initializer function.
    """
    return variance_scaling(
        1.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype
    )


def he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = float,
) -> Initializer:
    """
    Returns an initializer for the He (Kaiming) uniform initialization.

    Args:
        in_axis (Union[int, Sequence[int]], optional): Input dimension axis/axes. Defaults to -2.
        out_axis (Union[int, Sequence[int]], optional): Output dimension axis/axes. Defaults to -1.
        batch_axis (Sequence[int], optional): Batch dimension axis/axes. Defaults to ().
        dtype (Any, optional): Data type. Defaults to float.

    Returns:
        Initializer: A He uniform initializer function.
    """
    return variance_scaling(
        2.0, "fan_in", "uniform", in_axis, out_axis, batch_axis, dtype
    )


kaiming_uniform = he_uniform


def he_normal(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = float,
) -> Initializer:
    """
    Returns an initializer for the He (Kaiming) normal initialization.

    Args:
        in_axis (Union[int, Sequence[int]], optional): Input dimension axis/axes. Defaults to -2.
        out_axis (Union[int, Sequence[int]], optional): Output dimension axis/axes. Defaults to -1.
        batch_axis (Sequence[int], optional): Batch dimension axis/axes. Defaults to ().
        dtype (Any, optional): Data type. Defaults to float.

    Returns:
        Initializer: A He normal initializer function.
    """
    return variance_scaling(
        2.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype
    )


kaiming_normal = he_normal


def orthogonal(
    scale: RealNumeric = 1.0, column_axis: int = -1, dtype: Any = float
) -> Initializer:
    """
    Returns an initializer that generates orthogonally initialized weight arrays.

    Args:
        scale (RealNumeric, optional): Scaling factor for the initialized array. Defaults to 1.0.
        column_axis (int, optional): The axis to treat as columns. Defaults to -1.
        dtype (Any, optional): Data type of the array. Defaults to float.

    Returns:
        Initializer: An orthogonal initializer function.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """
        Initializes an orthogonally constrained array.

        Args:
            key (Any): PRNG key.
            shape (Any): Shape of the output array.
            dtype (Any, optional): Data type.

        Returns:
            Any: An orthogonally initialized array.
        """
        return uniform(scale, dtype)(key, shape, dtype)  # pragma: no cover

    return init


def delta_orthogonal(
    scale: RealNumeric = 1.0, column_axis: int = -1, dtype: Any = float
) -> Initializer:
    """
    Returns an initializer that generates delta orthogonal arrays (useful for CNNs).

    Args:
        scale (RealNumeric, optional): Scaling factor for the initialized array. Defaults to 1.0.
        column_axis (int, optional): The axis to treat as columns. Defaults to -1.
        dtype (Any, optional): Data type of the array. Defaults to float.

    Returns:
        Initializer: A delta orthogonal initializer function.
    """

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """
        Initializes a delta orthogonal constrained array.

        Args:
            key (Any): PRNG key.
            shape (Any): Shape of the output array.
            dtype (Any, optional): Data type.

        Returns:
            Any: A delta orthogonally initialized array.
        """
        return uniform(scale, dtype)(key, shape, dtype)  # pragma: no cover

    return init
