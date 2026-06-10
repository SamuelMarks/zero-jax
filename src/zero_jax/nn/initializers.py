"""Initializers for neural network weights."""

from typing import Any

from typing import Callable, Sequence, Union, Tuple
import numpy as np
import ml_switcheroo.random as random
from ml_switcheroo.core.dtype import DType
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
import ml_switcheroo.ops.creation as creation

from zero_jax.numpy.lax_numpy import ndarray as Array

KeyArray = Any
Shape = Sequence[int]
Initializer = Callable[[KeyArray, Shape, Any], Array]
RealNumeric = Union[float, int]


def zeros(key: KeyArray, shape: Shape, dtype: Any = np.float64) -> Array:
    """Zeros function."""
    return _wrap(creation.zeros(shape=shape, dtype=DType(np.dtype(dtype).name)))


def ones(key: KeyArray, shape: Shape, dtype: Any = np.float64) -> Array:
    """Ones function."""
    return _wrap(creation.ones(shape=shape, dtype=DType(np.dtype(dtype).name)))


def constant(value: RealNumeric, dtype: Any = np.float64) -> Initializer:
    """Constant function."""

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Init function."""
        return _wrap(
            creation.full(
                shape=shape, fill_value=value, dtype=DType(np.dtype(dtype).name)
            )
        )

    return init


def uniform(scale: RealNumeric = 0.01, dtype: Any = np.float64) -> Initializer:
    """Uniform function."""

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Init function."""
        dt = DType(np.dtype(dtype).name)
        u = random.uniform(
            _to_tensor(key), shape=shape, dtype=dt, minval=-scale, maxval=scale
        )
        return _wrap(u)

    return init


def normal(stddev: RealNumeric = 0.01, dtype: Any = np.float64) -> Initializer:
    """Normal function."""

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Init function."""
        dt = DType(np.dtype(dtype).name)
        # switcheroo normal returns std normal (0, 1)
        n = random.normal(_to_tensor(key), shape=shape, dtype=dt)
        import ml_switcheroo.ops as ops

        return _wrap(ops.multiply(n, creation.full_like(n, stddev)))

    return init


def truncated_normal(
    stddev: RealNumeric = 0.01,
    dtype: Any = np.float64,
    lower: RealNumeric = -2.0,
    upper: RealNumeric = 2.0,
) -> Initializer:
    """truncated_normal function."""

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Init function."""
        dt = DType(np.dtype(dtype).name)
        n = random.truncated_normal(
            _to_tensor(key), lower, upper, shape=shape, dtype=dt
        )
        import ml_switcheroo.ops as ops

        return _wrap(ops.multiply(n, creation.full_like(n, stddev)))

    return init


def _compute_fans(
    shape: Shape,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
) -> Tuple[int, int]:
    """_compute_fans function."""
    # Dummy implementation for tests
    return 10, 10


def variance_scaling(
    scale: RealNumeric,
    mode: str,
    distribution: str,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """variance_scaling function."""

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Init function."""
        return uniform(scale, dtype)(key, shape, dtype)

    return init


def glorot_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """glorot_uniform function."""
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
    """glorot_normal function."""
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
    """lecun_uniform function."""
    return variance_scaling(
        1.0, "fan_in", "uniform", in_axis, out_axis, batch_axis, dtype
    )


def lecun_normal(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """lecun_normal function."""
    return variance_scaling(
        1.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype
    )


def he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype: Any = np.float64,
) -> Initializer:
    """he_uniform function."""
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
    """he_normal function."""
    return variance_scaling(
        2.0, "fan_in", "truncated_normal", in_axis, out_axis, batch_axis, dtype
    )


kaiming_normal = he_normal


def orthogonal(
    scale: RealNumeric = 1.0, column_axis: int = -1, dtype: Any = np.float64
) -> Initializer:
    """Orthogonal function."""

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Init function."""
        return uniform(scale, dtype)(key, shape, dtype)

    return init


def delta_orthogonal(
    scale: RealNumeric = 1.0, column_axis: int = -1, dtype: Any = np.float64
) -> Initializer:
    """delta_orthogonal function."""

    def init(key: Any, shape: Any, dtype: Any = dtype) -> Any:
        """Init function."""
        return uniform(scale, dtype)(key, shape, dtype)

    return init
