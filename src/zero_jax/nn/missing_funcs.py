"""Missing activation functions and utilities for jax.nn."""

from __future__ import annotations

import math
from typing import Any


def glu(x: Any, axis: int = -1) -> Any:
    from zero_jax.nn import sigmoid
    from zero_jax.numpy import split

    a, b = split(x, 2, axis=axis)
    return a * sigmoid(b)  # pragma: no cover


def hard_silu(x: Any) -> Any:
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.hard_silu(_to_tensor(x)))


def hard_swish(x: Any) -> Any:
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.hard_swish(_to_tensor(x)))


def leaky_relu(x: Any, negative_slope: float = 0.01) -> Any:
    from zero_jax.numpy import where

    return where(x >= 0, x, x * negative_slope)


def mish(x: Any) -> Any:
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.mish(_to_tensor(x)))


def soft_sign(x: Any) -> Any:
    from zero_jax.numpy import abs

    return x / (1.0 + abs(x))


def softplus(x: Any) -> Any:
    from zero_jax.numpy import exp, log1p

    return log1p(exp(x))


def sparse_plus(x: Any) -> Any:
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.sparse_plus(_to_tensor(x)))


def sparse_sigmoid(x: Any) -> Any:
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.sparse_sigmoid(_to_tensor(x)))


def squareplus(x: Any, b: float = 4.0) -> Any:
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.squareplus(_to_tensor(x), b))


def standardize(
    x: Any,
    axis: Any = None,
    mean: Any = None,
    variance: Any = None,
    epsilon: float = 1e-5,
) -> Any:
    from zero_jax.numpy import mean as jnp_mean
    from zero_jax.numpy import sqrt, var

    if mean is None:
        mean = jnp_mean(x, axis=axis, keepdims=True)
    if variance is None:
        variance = var(x, axis=axis, keepdims=True)
    return (x - mean) / sqrt(variance + epsilon)
