"""Missing activation functions and utilities for jax.nn."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLike, AxisSpec
from zero_jax.nn.activation import sigmoid
from zero_jax.numpy import (
    abs as _abs,
)
from zero_jax.numpy import (
    asarray,
    exp,
    log1p,
    split,
    sqrt,
    var,
    where,
)
from zero_jax.numpy import (
    mean as jnp_mean,
)
from zero_jax.numpy import (
    sum as jnp_sum,
)
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, ndarray


def _as_float(x: ArrayLike) -> ndarray:
    """Convert ``x`` to a floating-point ``ndarray`` preserving ``float64`` if active.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Floating-point ``ndarray``.
    """
    arr = asarray(x)
    if "float" in str(arr.dtype):
        return arr
    from zero_jax.numpy import float32

    return asarray(arr, dtype=float32)


def glu(x: ArrayLike, axis: int = -1) -> ndarray:
    """Gated linear unit activation function.

    Args:
        x: Input array whose size along ``axis`` is even.
        axis: Axis along which the input is split in half.

    Returns:
        ndarray: ``a * sigmoid(b)`` where ``a, b = split(x, 2, axis=axis)``.
    """
    a, b = split(x, 2, axis=axis)
    return a * sigmoid(b)


def hard_silu(x: ArrayLike) -> ndarray:
    """Hard SiLU (Hard Swish) activation function ``x * relu6(x + 3) / 6``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Activated array.
    """
    return _wrap(ops.hard_silu(_to_tensor(x)))


def hard_swish(x: ArrayLike) -> ndarray:
    """Hard Swish activation function (alias of :func:`hard_silu`).

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Activated array.
    """
    return hard_silu(x)


def identity(x: ArrayLike) -> ndarray:
    """Identity activation function returning ``x`` as an ``ndarray``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Array representation of ``x``.
    """
    return asarray(x)


def leaky_relu(x: ArrayLike, negative_slope: float = 0.01) -> ndarray:
    """Leaky rectified linear unit activation function.

    Args:
        x: Input array-like value.
        negative_slope: Multiplicative slope for negative inputs.

    Returns:
        ndarray: Activated array.
    """
    arr = asarray(x)
    return where(arr >= 0, arr, arr * negative_slope)


def mish(x: ArrayLike) -> ndarray:
    """Mish activation function ``x * tanh(softplus(x))``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Activated array.
    """
    return _wrap(ops.mish(_to_tensor(x)))


def normalize(
    x: ArrayLike,
    axis: AxisSpec = -1,
    mean: ArrayLike | None = None,
    variance: ArrayLike | None = None,
    epsilon: float = 1e-5,
) -> ndarray:
    """L2-normalize (or mean/variance standardize when ``mean``/``variance`` are provided) ``x`` along ``axis``.

    Args:
        x: Input array-like value.
        axis: Axis or axes along which to normalize.
        mean: Optional precomputed mean (if provided, delegates to :func:`standardize`).
        variance: Optional precomputed variance (if provided, delegates to :func:`standardize`).
        epsilon: Small constant added for numerical stability.

    Returns:
        ndarray: Normalized array.
    """
    arr = _as_float(x)
    if mean is not None or variance is not None:
        return standardize(arr, axis=axis, mean=mean, variance=variance, epsilon=epsilon)
    norm_sq = jnp_sum(arr * arr, axis=axis, keepdims=True)
    return arr / sqrt(norm_sq + epsilon)


def soft_sign(x: ArrayLike) -> ndarray:
    """Soft-sign activation function ``x / (1 + |x|)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Activated array.
    """
    arr = _as_float(x)
    return arr / (1.0 + _abs(arr))


def softplus(x: ArrayLike) -> ndarray:
    """Softplus activation function ``log(1 + exp(x))``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Activated array.
    """
    arr = _as_float(x)
    return log1p(exp(arr))


def sparse_plus(x: ArrayLike) -> ndarray:
    """Sparse-plus activation function.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Activated array.
    """
    arr = _as_float(x)
    return where(arr <= -1.0, 0.0, where(arr >= 1.0, arr, 0.25 * (arr + 1.0) * (arr + 1.0)))


def sparse_sigmoid(x: ArrayLike) -> ndarray:
    """Sparse-sigmoid activation function.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Activated array.
    """
    arr = _as_float(x)
    return where(arr <= -1.0, 0.0, where(arr >= 1.0, 1.0, 0.5 * (arr + 1.0)))


def squareplus(x: ArrayLike, b: float = 4.0) -> ndarray:
    """Squareplus activation function ``0.5 * (x + sqrt(x^2 + b))``.

    Args:
        x: Input array-like value.
        b: Smoothness parameter.

    Returns:
        ndarray: Activated array.
    """
    arr = _as_float(x)
    return 0.5 * (arr + sqrt(arr * arr + b))


def standardize(
    x: ArrayLike,
    axis: AxisSpec = -1,
    mean: ArrayLike | None = None,
    variance: ArrayLike | None = None,
    epsilon: float = 1e-5,
    where: ArrayLike | None = None,
) -> ndarray:
    """Standardize input ``x`` to zero mean and unit variance along ``axis``.

    Args:
        x: Input array-like value.
        axis: Axis or axes along which to standardize.
        mean: Optional precomputed mean.
        variance: Optional precomputed variance.
        epsilon: Small constant added to variance before square root.
        where: Optional boolean mask for mean/variance computation.

    Returns:
        ndarray: Standardized array.
    """
    arr = _as_float(x)
    _ = where
    m = jnp_mean(arr, axis=axis, keepdims=True) if mean is None else _as_float(mean)
    v = var(arr, axis=axis, keepdims=True) if variance is None else _as_float(variance)
    return (arr - m) / sqrt(v + epsilon)


__all__ = [
    "glu",
    "hard_silu",
    "hard_swish",
    "identity",
    "leaky_relu",
    "mish",
    "normalize",
    "soft_sign",
    "softplus",
    "sparse_plus",
    "sparse_sigmoid",
    "squareplus",
    "standardize",
]
