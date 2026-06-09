"""Activation functions and related utilities.

This module implements basic neural network primitives like gelu, softmax, and one_hot.
"""

from typing import Any, Optional, Union, Tuple
ArrayLike = Any
from typing import Any, TypeVar

import numpy as np
import math


def _erf(x: np.ndarray) -> np.ndarray:
    """Computes the error function element-wise.

    Args:
        x: Input array.

    Returns:
        Array with error function applied element-wise.
    """
    return np.vectorize(math.erf)(x)


def gelu(x: ArrayLike, approximate: bool = True) -> np.ndarray:
    """Gaussian error linear unit activation function.

    Args:
        x: Input array.
        approximate: Whether to use the approximate formulation.

    Returns:
        The gelu activation applied to the input.
    """
    x = np.asarray(x)
    if approximate:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    else:
        return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def logsumexp(
    a: ArrayLike,
    axis: Any = None,
    b: Optional[ArrayLike] = None,
    keepdims: bool = False,
    return_sign: bool = False,
    where: Optional[ArrayLike] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Log-sum-exp reduction.

    Args:
        a: Input array.
        axis: Axis or axes over which the sum is taken.
        b: Scaling factors for the elements of `a`.
        keepdims: If True, the axes which are reduced are left in the result as dimensions with size one.
        return_sign: If True, returns the sign of the result.
        where: Elements to include in the reduction.

    Returns:
        The log-sum-exp reduction.
    """
    a = np.asarray(a)
    if b is not None:
        b = np.asarray(b)
    if where is not None:
        where = np.asarray(where)
        a = np.where(where, a, -np.inf)

    a_max = np.max(a, axis=axis, keepdims=True)
    # Handle infinite max cases (e.g. all elements are -inf)
    a_max = np.where(np.isinf(a_max), 0.0, a_max)

    tmp = a - a_max
    if b is not None:
        exp_a = b * np.exp(tmp)
    else:
        exp_a = np.exp(tmp)

    if where is not None:
        exp_a = np.where(where, exp_a, 0.0)

    sum_exp = np.sum(exp_a, axis=axis, keepdims=keepdims)

    out = (
        np.log(np.abs(sum_exp)) + np.squeeze(a_max, axis=axis)
        if not keepdims and axis is not None
        else np.log(np.abs(sum_exp)) + a_max
    )

    if return_sign:
        return out, np.sign(sum_exp)
    return out


def one_hot(
    x: Any, num_classes: int, *, dtype: Any = np.float32, axis: Any = -1
) -> np.ndarray:
    """One-hot encodes the given indices.

    Args:
        x: Array of indices.
        num_classes: Number of classes.
        dtype: Output data type.
        axis: Axis along which the one-hot encoding is added.

    Returns:
        One-hot encoded array.
    """
    x = np.asarray(x)
    shape = list(x.shape)
    if axis < 0:
        axis = len(shape) + 1 + axis
    shape.insert(axis, num_classes)

    out = np.zeros(shape, dtype=dtype)

    indices = []
    for i, dim in enumerate(shape):
        if i == axis:
            indices.append(x)
        else:
            idx = i if i < axis else i - 1
            # Create indexing array for this dimension
            shape_idx = [1] * len(x.shape)
            shape_idx[idx] = x.shape[idx]
            indices.append(np.arange(x.shape[idx]).reshape(shape_idx))

    out[tuple(indices)] = 1
    return out


def softmax(
    x: ArrayLike,
    axis: Any = -1,
    where: Optional[Any] = None,
    initial: Any = None,
) -> np.ndarray:
    """Softmax function.

    Args:
        x: Input array.
        axis: Axis or axes along which the softmax is computed.
        where: Elements to include in the softmax.
        initial: Initial value for the reduction.

    Returns:
        The softmax activation.
    """
    x = np.asarray(x)
    if where is not None:
        where = np.asarray(where)
        x_max = np.max(np.where(where, x, -np.inf), axis=axis, keepdims=True)
    else:
        x_max = np.max(x, axis=axis, keepdims=True)

    unnormalized = np.exp(x - x_max)

    if where is not None:
        unnormalized = np.where(where, unnormalized, 0.0)

    denominator = np.sum(unnormalized, axis=axis, keepdims=True)

    return unnormalized / denominator

def sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))

def log_sigmoid(x: Any) -> Any:
    return -np.logaddexp(0.0, -x)
