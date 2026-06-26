"""Missing activation functions and utilities for jax.nn."""

from __future__ import annotations
from typing import Any
import math


def glu(x: Any, axis: int = -1) -> Any:
    from zero_jax.numpy import split
    from zero_jax.nn import sigmoid

    a, b = split(x, 2, axis=axis)
    return a * sigmoid(b)


def hard_silu(x: Any) -> Any:
    from zero_jax.numpy import where, minimum, maximum

    return x * maximum(0.0, minimum(1.0, (x + 3.0) / 6.0))


def hard_swish(x: Any) -> Any:
    return hard_silu(x)


def leaky_relu(x: Any, negative_slope: float = 0.01) -> Any:
    from zero_jax.numpy import where

    return where(x >= 0, x, x * negative_slope)


def mish(x: Any) -> Any:
    from zero_jax.numpy import tanh, exp, log1p

    # log1p(exp(x)) is softplus
    return x * tanh(log1p(exp(x)))


def soft_sign(x: Any) -> Any:
    from zero_jax.numpy import abs

    return x / (1.0 + abs(x))


def softplus(x: Any) -> Any:
    from zero_jax.numpy import log1p, exp

    return log1p(exp(x))


def sparse_plus(x: Any) -> Any:
    from zero_jax.numpy import where

    return where(x <= -1.0, 0.0, where(x >= 1.0, x, 0.25 * (x + 1.0) ** 2))


def sparse_sigmoid(x: Any) -> Any:
    from zero_jax.numpy import where

    return where(x <= -1.0, 0.0, where(x >= 1.0, 1.0, 0.5 * x + 0.5))


def squareplus(x: Any, b: float = 4.0) -> Any:
    from zero_jax.numpy import sqrt

    return 0.5 * (x + sqrt(x * x + b))


def standardize(
    x: Any,
    axis: Any = None,
    mean: Any = None,
    variance: Any = None,
    epsilon: float = 1e-5,
) -> Any:
    from zero_jax.numpy import mean as jnp_mean, var, sqrt

    if mean is None:
        mean = jnp_mean(x, axis=axis, keepdims=True)
    if variance is None:
        variance = var(x, axis=axis, keepdims=True)
    return (x - mean) / sqrt(variance + epsilon)
