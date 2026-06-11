"""Activation functions and related utilities."""

from typing import Any
import ml_switcheroo

from typing import Optional
import math
from zero_jax.numpy.lax_numpy import _wrap, _to_tensor
import ml_switcheroo.ops as ops
import ml_switcheroo.ops.creation as creation
import ml_switcheroo.nn as nn

ArrayLike = Any


def _erf(x: Any) -> Any:
    """
    Computes the error function of the given input.

    Args:
        x (Any): The input array-like object.

    Returns:
        Any: An array of the same shape as `x` containing the error function values.
    """
    return _wrap(ops.erf(_to_tensor(x)))


def gelu(x: ArrayLike, approximate: bool = False) -> Any:
    """
    Computes the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (ArrayLike): The input array.
        approximate (bool): If True, uses the approximate GELU formulation based on tanh. Defaults to False.

    Returns:
        Any: The array after applying the GELU activation.
    """
    return _wrap(nn.gelu(_to_tensor(x), approximate="tanh" if approximate else "none"))


def logsumexp(
    a: ArrayLike,
    axis: Any = None,
    b: Optional[ArrayLike] = None,
    keepdims: bool = False,
    return_sign: bool = False,
    where: Optional[ArrayLike] = None,
) -> Any:
    """
    Computes the log of the sum of exponentials of input elements.

    Args:
        a (ArrayLike): Input array.
        axis (Any, optional): Axis or axes over which the sum is computed. By default, computes the sum over all elements.
        b (Optional[ArrayLike], optional): Array of weights for the elements of `a`. Defaults to None.
        keepdims (bool): If True, retains reduced dimensions with length 1. Defaults to False.
        return_sign (bool): If True, returns a tuple of (result, sign). Defaults to False.
        where (Optional[ArrayLike], optional): Elements to include in the sum. Defaults to None.

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
    from ml_switcheroo.core.dtype import DType

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
    x_expanded = ops.unsqueeze(x_t, axis)
    classes_expanded = classes
    for i in range(len(x_expanded.shape)):
        if i != (axis if axis >= 0 else len(x_expanded.shape) + axis):
            classes_expanded = ops.unsqueeze(classes_expanded, i)

    eq = ops.equal(x_expanded, classes_expanded)
    # cast to dtype
    dt = _to_dtype(dtype)
    return _wrap(ops.cast(eq, dt))


def softmax(
    x: ArrayLike, axis: Any = -1, where: Optional[Any] = None, initial: Any = None
) -> Any:
    """
    Computes the softmax activation function over the given axis.

    Args:
        x (ArrayLike): Input array.
        axis (Any, optional): Axis along which the softmax is computed. Defaults to -1.
        where (Optional[Any], optional): Elements to include in the computation. Defaults to None.
        initial (Any, optional): Initial value for the sum. Defaults to None.

    Returns:
        Any: Array of the same shape as `x` with softmax applied.
    """
    x_t = _to_tensor(x)
    amax = ops.max(x_t, axis=axis, keepdims=True)
    shifted = ops.subtract(x_t, amax)
    if where is not None:
        shifted = ops.where(
            _to_tensor(where), shifted, creation.full_like(shifted, -float("inf"))
        )

    exp_x = ops.exp(shifted)
    if where is not None:
        exp_x = ops.where(_to_tensor(where), exp_x, creation.zeros_like(exp_x))

    sum_exp = ops.sum(exp_x, axis=axis, keepdims=True)
    return _wrap(ops.divide(exp_x, sum_exp))


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


def relu(x: ArrayLike) -> Any:
    """
    Computes the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (ArrayLike): Input array.

    Returns:
        Any: Array with ReLU applied.
    """
    return _wrap(nn.relu(_to_tensor(x)))


def relu6(x: ArrayLike) -> Any:
    """
    Computes the ReLU6 activation function, capping at 6.

    Args:
        x (ArrayLike): Input array.

    Returns:
        Any: Array with ReLU6 applied.
    """
    x = _to_tensor(x)
    return _wrap(ops.minimum(ops.maximum(x, _to_tensor(0.0)), _to_tensor(6.0)))


def hard_sigmoid(x: ArrayLike) -> Any:
    """
    Computes the hard sigmoid activation function.

    Args:
        x (ArrayLike): Input array.

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


def hard_tanh(x: ArrayLike) -> Any:
    """
    Computes the hard tanh activation function, bounding the input between -1 and 1.

    Args:
        x (ArrayLike): Input array.

    Returns:
        Any: Array with hard tanh applied.
    """
    x = _to_tensor(x)
    return _wrap(ops.maximum(_to_tensor(-1.0), ops.minimum(_to_tensor(1.0), x)))


def swish(x: ArrayLike) -> Any:
    """
    Computes the Swish activation function (x * sigmoid(x)).

    Args:
        x (ArrayLike): Input array.

    Returns:
        Any: Array with Swish applied.
    """
    return _wrap(nn.swish(_to_tensor(x)))


def silu(x: ArrayLike) -> Any:
    """
    Computes the SiLU (Sigmoid Linear Unit) activation function, which is identical to Swish.

    Args:
        x (ArrayLike): Input array.

    Returns:
        Any: Array with SiLU applied.
    """
    return _wrap(nn.swish(_to_tensor(x)))


def elu(x: ArrayLike, alpha: float = 1.0) -> Any:
    """
    Computes the Exponential Linear Unit (ELU) activation function.

    Args:
        x (ArrayLike): Input array.
        alpha (float, optional): Scaling factor for negative values. Defaults to 1.0.

    Returns:
        Any: Array with ELU applied.
    """
    return _wrap(nn.elu(_to_tensor(x), alpha=alpha))


def celu(x: ArrayLike, alpha: float = 1.0) -> Any:
    """
    Computes the Continuously Differentiable Exponential Linear Unit (CELU) activation function.

    Args:
        x (ArrayLike): Input array.
        alpha (float, optional): Scaling factor controlling the curvature for negative values. Defaults to 1.0.

    Returns:
        Any: Array with CELU applied.
    """
    return _wrap(nn.celu(_to_tensor(x), alpha=alpha))


def selu(x: ArrayLike) -> Any:
    """
    Computes the Scaled Exponential Linear Unit (SELU) activation function.

    Args:
        x (ArrayLike): Input array.

    Returns:
        Any: Array with SELU applied.
    """
    return _wrap(nn.selu(_to_tensor(x)))


def log_softmax(x: ArrayLike, axis: int = -1) -> Any:
    """
    Computes the logarithm of the softmax activation function.

    Args:
        x (ArrayLike): Input array.
        axis (int, optional): Axis along which to compute the log-softmax. Defaults to -1.

    Returns:
        Any: Array of the same shape as `x` with log-softmax applied.
    """
    return _wrap(nn.log_softmax(_to_tensor(x), dim=axis))
