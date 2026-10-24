"""Missing activation functions and utilities for jax.nn."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec


def glu(x: ArrayLikeSpec, axis: int = -1) -> ArrayLikeSpec:
    """Execute ``glu``.

    Args:
        x: Input parameter ``x``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    from zero_jax.nn import sigmoid
    from zero_jax.numpy import split

    a, b = split(x, 2, axis=axis)
    return a * sigmoid(b)


def hard_silu(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``hard_silu``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.hard_silu(_to_tensor(x)))


def hard_swish(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``hard_swish``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.hard_swish(_to_tensor(x)))


def leaky_relu(x: ArrayLikeSpec, negative_slope: float = 0.01) -> ArrayLikeSpec:
    """Execute ``leaky_relu``.

    Args:
        x: Input parameter ``x``.
        negative_slope: Input parameter ``negative_slope``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import where

    return where(x >= 0, x, x * negative_slope)


def mish(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``mish``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.mish(_to_tensor(x)))


def soft_sign(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``soft_sign``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import abs

    return x / (1.0 + abs(x))


def softplus(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``softplus``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import exp, log1p

    return log1p(exp(x))


def sparse_plus(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``sparse_plus``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.sparse_plus(_to_tensor(x)))


def sparse_sigmoid(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``sparse_sigmoid``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.sparse_sigmoid(_to_tensor(x)))


def squareplus(x: ArrayLikeSpec, b: float = 4.0) -> ArrayLikeSpec:
    """Execute ``squareplus``.

    Args:
        x: Input parameter ``x``.
        b: Input parameter ``b``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.squareplus(_to_tensor(x), b))


def standardize(
    x: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    mean: ArrayLikeSpec = None,
    variance: ArrayLikeSpec = None,
    epsilon: float = 1e-5,
) -> ArrayLikeSpec:
    """Execute ``standardize``.

    Args:
        x: Input parameter ``x``.
        axis: Input parameter ``axis``.
        mean: Input parameter ``mean``.
        variance: Input parameter ``variance``.
        epsilon: Input parameter ``epsilon``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import mean as jnp_mean
    from zero_jax.numpy import sqrt, var

    if mean is None:
        mean = jnp_mean(x, axis=axis, keepdims=True)
    if variance is None:
        variance = var(x, axis=axis, keepdims=True)
    return (x - mean) / sqrt(variance + epsilon)
