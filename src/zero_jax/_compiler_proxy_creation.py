"""Static creation operations routing to ml_switcheroo_compiler.ops.creation."""

from __future__ import annotations

from collections.abc import Sequence

import ml_switcheroo_compiler.ops.creation as _original_creation
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor

from zero_jax._types import RealScalar, ShapeSpec, ShapeTuple


def _validate_shape(shape: ShapeSpec) -> int | ShapeTuple:
    """Validate and normalize a shape specification into an ``int`` or ``tuple[int, ...]``.

    Args:
        shape: Shape specification (integer or sequence of integers).

    Returns:
        int | ShapeTuple: Validated shape specification.

    Raises:
        TypeError: If ``shape`` is a string, bytes, or contains non-integers.
    """
    if isinstance(shape, (str, bytes)):
        raise TypeError(f"Shape must be an int or sequence of ints, got {type(shape).__name__}")
    if isinstance(shape, int):
        return shape
    if isinstance(shape, Sequence):
        validated: list[int] = []
        for dim in shape:
            if isinstance(dim, (str, bytes)) or not isinstance(dim, int):
                raise TypeError(f"Shape dimensions must be integers, got {type(dim).__name__}")
            validated.append(int(dim))
        return tuple(validated)
    raise TypeError(f"Shape must be an int or sequence of ints, got {type(shape).__name__}")


def zeros(
    shape: ShapeSpec,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Create a tensor filled with zeros.

    Args:
        shape: Shape of the new tensor.
        dtype: Desired data type of the tensor.
        device: Target device for the tensor.

    Returns:
        Tensor: A tensor filled with zeros.
    """
    return _original_creation.zeros(shape=_validate_shape(shape), dtype=dtype, device=device)


def ones(
    shape: ShapeSpec,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Create a tensor filled with ones.

    Args:
        shape: Shape of the new tensor.
        dtype: Desired data type of the tensor.
        device: Target device for the tensor.

    Returns:
        Tensor: A tensor filled with ones.
    """
    return _original_creation.ones(shape=_validate_shape(shape), dtype=dtype, device=device)


def full(
    shape: ShapeSpec,
    fill_value: RealScalar,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Create a tensor filled with a scalar value.

    Args:
        shape: Shape of the new tensor.
        fill_value: Value to fill the tensor with.
        dtype: Desired data type of the tensor.
        device: Target device for the tensor.

    Returns:
        Tensor: A tensor filled with ``fill_value``.
    """
    return _original_creation.full(shape=_validate_shape(shape), fill_value=fill_value, dtype=dtype, device=device)


def zeros_like(
    input: Tensor,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Create a tensor of zeros with the same shape and type as ``input``.

    Args:
        input: Reference tensor for shape and default dtype.
        dtype: Optional override data type.
        device: Optional override device.

    Returns:
        Tensor: A tensor of zeros with the same shape as ``input``.
    """
    return _original_creation.zeros_like(input=input, dtype=dtype, device=device)


def ones_like(
    input: Tensor,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Create a tensor of ones with the same shape and type as ``input``.

    Args:
        input: Reference tensor for shape and default dtype.
        dtype: Optional override data type.
        device: Optional override device.

    Returns:
        Tensor: A tensor of ones with the same shape as ``input``.
    """
    return _original_creation.ones_like(input=input, dtype=dtype, device=device)


def full_like(
    input: Tensor,
    fill_value: RealScalar,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Tensor:
    """Create a tensor filled with ``fill_value`` with the same shape as ``input``.

    Args:
        input: Reference tensor for shape and default dtype.
        fill_value: Value to fill the tensor with.
        dtype: Optional override data type.
        device: Optional override device.

    Returns:
        Tensor: A tensor filled with ``fill_value``.
    """
    return _original_creation.full_like(input=input, fill_value=fill_value, dtype=dtype, device=device)


def arange(
    start: RealScalar = 0,
    stop: RealScalar | None = None,
    step: RealScalar = 1,
    dtype: DType | None = None,
    device: Device | None = None,
) -> TensorOrScalar:
    """Create a 1-D tensor with evenly spaced values within a given interval.

    Args:
        start: Start of interval (or stop if ``stop`` is None).
        stop: End of interval.
        step: Spacing between values.
        dtype: Desired data type of the output tensor.
        device: Target device for the output tensor.

    Returns:
        TensorOrScalar: 1-D tensor of evenly spaced values.
    """
    return _original_creation.arange(start=start, stop=stop, step=step, dtype=dtype, device=device)


def eye(
    n: int,
    m: int | None = None,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> TensorOrScalar:
    """Create a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n: Number of rows in the output.
        m: Number of columns in the output (defaults to ``n``).
        k: Index of the diagonal.
        dtype: Data type of the returned tensor.
        device: Target device for the returned tensor.

    Returns:
        TensorOrScalar: 2-D identity-like tensor.
    """
    return _original_creation.eye(n=n, m=m, k=k, dtype=dtype, device=device)


def identity(
    n: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> TensorOrScalar:
    """Create a square identity matrix of size ``n x n``.

    Args:
        n: Number of rows and columns in the output.
        dtype: Data type of the returned tensor.
        device: Target device for the returned tensor.

    Returns:
        TensorOrScalar: 2-D square identity tensor.
    """
    return _original_creation.identity(n=n, dtype=dtype, device=device)


def linspace(
    start: RealScalar,
    stop: RealScalar,
    steps: int,
    dtype: DType | None = None,
    device: Device | None = None,
) -> TensorOrScalar:
    """Create a 1-D tensor of ``steps`` evenly spaced points between ``start`` and ``stop``.

    Args:
        start: Starting value of the sequence.
        stop: End value of the sequence.
        steps: Number of samples to generate.
        dtype: Data type of the output tensor.
        device: Target device for the output tensor.

    Returns:
        TensorOrScalar: 1-D tensor of evenly spaced values.
    """
    return _original_creation.linspace(start=start, stop=stop, steps=steps, dtype=dtype, device=device)


__all__ = [
    "_validate_shape",
    "arange",
    "eye",
    "full",
    "full_like",
    "identity",
    "linspace",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
]
