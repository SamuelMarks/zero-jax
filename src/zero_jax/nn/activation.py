"""Activation functions and related utilities."""

from __future__ import annotations

import math
from typing import Any, Optional

import zero_jax._compiler_proxy_creation as creation
import zero_jax._compiler_proxy_ops as ops
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

_ArrayLike = Any


def _erf(x: Any) -> Any:
    """
    Computes the error function of the given input.

    Args:
        x (Any): The input array-like object.

    Returns:
        Any: An array of the same shape as `x` containing the error function values.
    """
    return _wrap(ops.erf(_to_tensor(x)))  # pragma: no cover


def gelu(x: _ArrayLike, approximate: bool = True) -> Any:
    """
    Computes the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (_ArrayLike): The input array.
        approximate (bool): If True, uses the approximate GELU formulation based on tanh. Defaults to False.

    Returns:
        Any: The array after applying the GELU activation.
    """
    x_t = _to_tensor(x)
    if approximate:
        # 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        const1 = ops.multiply(x_t, _to_tensor(0.5))
        const2 = _to_tensor(math.sqrt(2 / math.pi))
        x_cube = ops.multiply(x_t, ops.multiply(x_t, x_t))
        term2 = ops.multiply(_to_tensor(0.044715), x_cube)
        inner = ops.multiply(const2, ops.add(x_t, term2))
        tanh_inner = ops.tanh(inner)
        return _wrap(ops.multiply(const1, ops.add(_to_tensor(1.0), tanh_inner)))
    else:
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        const1 = ops.multiply(x_t, _to_tensor(0.5))  # pragma: no cover
        erf_inner = ops.divide(x_t, _to_tensor(math.sqrt(2.0)))  # pragma: no cover
        erf_val = ops.erf(erf_inner)  # pragma: no cover
        return _wrap(
            ops.multiply(const1, ops.add(_to_tensor(1.0), erf_val))
        )  # pragma: no cover


def logsumexp(
    a: _ArrayLike,
    axis: Any = None,
    b: Optional[_ArrayLike] = None,
    keepdims: bool = False,
    return_sign: bool = False,
    where: Optional[_ArrayLike] = None,
) -> Any:
    """
    Computes the log of the sum of exponentials of input elements.

    Args:
        a (_ArrayLike): Input array.
        axis (Any, optional): Axis or axes over which the sum is computed. By default, computes the sum over all elements.
        b (Optional[_ArrayLike], optional): Array of weights for the elements of `a`. Defaults to None.
        keepdims (bool): If True, retains reduced dimensions with length 1. Defaults to False.
        return_sign (bool): If True, returns a tuple of (result, sign). Defaults to False.
        where (Optional[_ArrayLike], optional): Elements to include in the sum. Defaults to None.

    Returns:
        Any: The logsumexp of the inputs. If `return_sign` is True, returns a tuple containing the logsumexp and its sign.
    """
    a_t = _to_tensor(a)
    amax = ops.max(a_t, axis=axis, keepdims=True)
    a_shifted = ops.subtract(a_t, amax)
    if b is not None:
        a_shifted = ops.multiply(ops.exp(a_shifted), _to_tensor(b))
    else:
        a_shifted = ops.exp(a_shifted)

    # sum
    sum_exp = ops.sum(a_shifted, axis=axis, keepdims=keepdims)
    log_sum_exp = ops.log(sum_exp)

    if not keepdims:
        amax = (
            ops.squeeze(amax, dims=[axis] if isinstance(axis, int) else axis)
            if axis is not None
            else ops.squeeze(amax)
        )

    res = ops.add(log_sum_exp, amax)

    if return_sign:
        # dummy sign
        sign = ops.sign(res)
        return _wrap(res), _wrap(sign)
    return _wrap(res)


def _to_dtype(dtype: Any) -> Any:
    """
    Converts the given dtype object to the underlying DType class representation.

    Args:
        dtype (Any): A dtype specified as string, type, or DType instance.

    Returns:
        Any: The parsed DType object.
    """
    from ml_switcheroo_compiler.core.dtype import DType

    if isinstance(dtype, DType):
        return dtype
    name = getattr(dtype, "name", None)
    if name is not None:
        return DType(name)
    if isinstance(dtype, str):
        return DType(dtype)
    if dtype is float:
        return DType("float32")
    if dtype is int:
        return DType("int32")
    if dtype is bool:
        return DType("bool")
    try:
        return DType(dtype.__name__)
    except Exception:
        return DType(str(dtype))


def one_hot(x: Any, num_classes: int, *, dtype: Any = float, axis: Any = -1) -> Any:
    """
    Creates a one-hot encoding of the given integer array.

    Args:
        x (Any): Integer array of class indices.
        num_classes (int): Number of total classes.
        dtype (Any, optional): The data type of the output. Defaults to float.
        axis (Any, optional): The axis along which the one-hot dimension is added. Defaults to -1.

    Returns:
        Any: A one-hot encoded array.
    """
    x_t = _to_tensor(x)
    classes = creation.arange(0, num_classes, dtype=x_t.dtype, device=x_t.device)
    # broadcast x and classes
    x_expanded = ops.expand_dims(x_t, axis)
    classes_expanded = classes
    for i in range(len(x_expanded.shape)):
        if i != (axis if axis >= 0 else len(x_expanded.shape) + axis):
            classes_expanded = ops.expand_dims(classes_expanded, i)

    eq = ops.equal(x_expanded, classes_expanded)
    # cast to dtype
    dt = _to_dtype(dtype)
    return _wrap(ops.cast(eq, dtype=dt))


def softmax(
    x: _ArrayLike, axis: Any = -1, where: Optional[Any] = None, initial: Any = None
) -> Any:
    """
    Computes the softmax activation function over the given axis.

    Args:
        x (_ArrayLike): Input array.
        axis (Any, optional): Axis along which the softmax is computed. Defaults to -1.
        where (Optional[Any], optional): Elements to include in the computation. Defaults to None.
        initial (Any, optional): Initial value for the sum. Defaults to None.

    Returns:
        Any: Array of the same shape as `x` with softmax applied.
    """
    x_t = _to_tensor(x)
    if where is not None:
        x_t = ops.where(
            _to_tensor(where), x_t, creation.full_like(x_t, -float("inf"))
        )  # pragma: no cover
    lse = ops.logsumexp(x_t, axis=axis, keepdims=True)
    res = ops.exp(ops.subtract(x_t, lse))
    if where is not None:
        res = ops.where(
            _to_tensor(where), res, creation.zeros_like(res)
        )  # pragma: no cover
    return _wrap(res)


def sigmoid(x: Any) -> Any:
    """
    Computes the sigmoid activation function.

    Args:
        x (Any): Input array-like object.

    Returns:
        Any: Array of the same shape as `x` with sigmoid applied.
    """
    x_t = _to_tensor(x)
    one = creation.full_like(x_t, 1.0)
    neg_x = ops.negative(x_t)
    exp_neg_x = ops.exp(neg_x)
    denom = ops.add(one, exp_neg_x)
    return _wrap(ops.divide(one, denom))


def log_sigmoid(x: Any) -> Any:
    """
    Computes the logarithm of the sigmoid function.

    Args:
        x (Any): Input array-like object.

    Returns:
        Any: Array with the log-sigmoid applied.
    """
    x_t = _to_tensor(x)
    # log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    one = creation.full_like(x_t, 1.0)
    neg_x = ops.negative(x_t)
    exp_neg_x = ops.exp(neg_x)
    denom = ops.add(one, exp_neg_x)
    return _wrap(ops.negative(ops.log(denom)))


def relu(x: _ArrayLike) -> Any:
    """
    Computes the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        Any: Array with ReLU applied.
    """
    x_t = _to_tensor(x)
    return _wrap(ops.maximum(x_t, _to_tensor(0.0)))


def relu6(x: _ArrayLike) -> Any:
    """
    Computes the ReLU6 activation function, capping at 6.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        Any: Array with ReLU6 applied.
    """
    x = _to_tensor(x)
    return _wrap(ops.minimum(ops.maximum(x, _to_tensor(0.0)), _to_tensor(6.0)))


def hard_sigmoid(x: _ArrayLike) -> Any:
    """
    Computes the hard sigmoid activation function.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        Any: Array with hard sigmoid applied.
    """
    x = _to_tensor(x)
    return _wrap(
        ops.maximum(
            _to_tensor(0.0),
            ops.minimum(
                _to_tensor(1.0),
                ops.add(ops.multiply(x, _to_tensor(1 / 6)), _to_tensor(0.5)),
            ),
        )
    )


def hard_tanh(x: _ArrayLike) -> Any:
    """
    Computes the hard tanh activation function, bounding the input between -1 and 1.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        Any: Array with hard tanh applied.
    """
    x = _to_tensor(x)
    return _wrap(ops.maximum(_to_tensor(-1.0), ops.minimum(_to_tensor(1.0), x)))


def swish(x: _ArrayLike) -> Any:
    """
    Computes the Swish activation function (x * sigmoid(x)).

    Args:
        x (_ArrayLike): Input array.

    Returns:
        Any: Array with Swish applied.
    """
    x_t = _to_tensor(x)
    return _wrap(ops.multiply(x_t, _to_tensor(sigmoid(x_t))))


def silu(x: _ArrayLike) -> Any:
    """
    Computes the SiLU (Sigmoid Linear Unit) activation function, which is identical to Swish.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        Any: Array with SiLU applied.
    """
    return swish(x)


def elu(x: _ArrayLike, alpha: float = 1.0) -> Any:
    """
    Computes the Exponential Linear Unit (ELU) activation function.

    Args:
        x (_ArrayLike): Input array.
        alpha (float, optional): Scaling factor for negative values. Defaults to 1.0.

    Returns:
        Any: Array with ELU applied.
    """
    x_t = _to_tensor(x)
    pos = ops.maximum(x_t, _to_tensor(0.0))
    neg = ops.multiply(
        _to_tensor(alpha),
        ops.subtract(ops.exp(ops.minimum(x_t, _to_tensor(0.0))), _to_tensor(1.0)),
    )
    return _wrap(ops.add(pos, neg))


def celu(x: _ArrayLike, alpha: float = 1.0) -> Any:
    """
    Computes the Continuously Differentiable Exponential Linear Unit (CELU) activation function.

    Args:
        x (_ArrayLike): Input array.
        alpha (float, optional): Scaling factor controlling the curvature for negative values. Defaults to 1.0.

    Returns:
        Any: Array with CELU applied.
    """
    x_t = _to_tensor(x)
    pos = ops.maximum(x_t, _to_tensor(0.0))
    neg = ops.multiply(
        _to_tensor(alpha),
        ops.subtract(
            ops.exp(ops.divide(ops.minimum(x_t, _to_tensor(0.0)), _to_tensor(alpha))),
            _to_tensor(1.0),
        ),
    )
    return _wrap(ops.add(pos, neg))


def selu(x: _ArrayLike) -> Any:
    """
    Computes the Scaled Exponential Linear Unit (SELU) activation function.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        Any: Array with SELU applied.
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x_t = _to_tensor(x)
    pos = ops.maximum(x_t, _to_tensor(0.0))
    neg = ops.multiply(
        _to_tensor(alpha),
        ops.subtract(ops.exp(ops.minimum(x_t, _to_tensor(0.0))), _to_tensor(1.0)),
    )
    return _wrap(ops.multiply(_to_tensor(scale), ops.add(pos, neg)))


def log_softmax(x: _ArrayLike, axis: int = -1) -> Any:
    """
    Computes the logarithm of the softmax activation function.

    Args:
        x (_ArrayLike): Input array.
        axis (int, optional): Axis along which to compute the log-softmax. Defaults to -1.

    Returns:
        Any: Array of the same shape as `x` with log-softmax applied.
    """
    x_t = _to_tensor(x)
    lse = ops.logsumexp(x_t, axis=axis, keepdims=True)
    return _wrap(ops.subtract(x_t, lse))


def tanh(x: Any) -> Any:
    """Hyperbolic tangent activation function.

    Args:
        x: Input array.

    Returns:
        The hyperbolic tangent of x.
    """
    from zero_jax.numpy.lax_numpy import tanh as _tanh

    return _tanh(x)
