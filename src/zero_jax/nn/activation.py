"""Module docstring."""

from typing import Any
import ml_switcheroo

"""Activation functions and related utilities."""

from typing import Optional
import numpy as np
import math
from zero_jax.numpy.lax_numpy import _wrap, _to_tensor
import ml_switcheroo.ops as ops
import ml_switcheroo.ops.creation as creation
import ml_switcheroo.nn as nn

ArrayLike = Any


def _erf(x: Any) -> Any:
    """_erf function."""
    return _wrap(ops.erf(_to_tensor(x)))


def gelu(x: ArrayLike, approximate: bool = False) -> Any:
    """Gelu function."""
    return _wrap(nn.gelu(_to_tensor(x), approximate="tanh" if approximate else "none"))


def logsumexp(
    a: ArrayLike,
    axis: Any = None,
    b: Optional[ArrayLike] = None,
    keepdims: bool = False,
    return_sign: bool = False,
    where: Optional[ArrayLike] = None,
) -> Any:
    """Logsumexp function."""
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


def one_hot(
    x: Any, num_classes: int, *, dtype: Any = np.float32, axis: Any = -1
) -> Any:
    """one_hot function."""
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
    from ml_switcheroo.core.dtype import DType

    dt = DType(np.dtype(dtype).name)
    return _wrap(ops.cast(eq, dt))


def softmax(
    x: ArrayLike, axis: Any = -1, where: Optional[Any] = None, initial: Any = None
) -> Any:
    """Softmax function."""
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
    """Sigmoid function."""
    x_t = _to_tensor(x)
    one = creation.full_like(x_t, 1.0)
    neg_x = ops.negative(x_t)
    exp_neg_x = ops.exp(neg_x)
    denom = ops.add(one, exp_neg_x)
    return _wrap(ops.divide(one, denom))


def log_sigmoid(x: Any) -> Any:
    """log_sigmoid function."""
    x_t = _to_tensor(x)
    # log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    one = creation.full_like(x_t, 1.0)
    neg_x = ops.negative(x_t)
    exp_neg_x = ops.exp(neg_x)
    denom = ops.add(one, exp_neg_x)
    return _wrap(ops.negative(ops.log(denom)))


def relu(x: ArrayLike) -> Any:
    """Relu function."""
    return _wrap(nn.relu(_to_tensor(x)))


def relu6(x: ArrayLike) -> Any:
    """relu6 function."""
    x = _to_tensor(x)
    return _wrap(ops.minimum(ops.maximum(x, _to_tensor(0.0)), _to_tensor(6.0)))


def hard_sigmoid(x: ArrayLike) -> Any:
    """hard_sigmoid function."""
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
    """hard_tanh function."""
    x = _to_tensor(x)
    return _wrap(ops.maximum(_to_tensor(-1.0), ops.minimum(_to_tensor(1.0), x)))


def swish(x: ArrayLike) -> Any:
    """Swish function."""
    return _wrap(nn.swish(_to_tensor(x)))


def silu(x: ArrayLike) -> Any:
    """Silu function."""
    return _wrap(nn.swish(_to_tensor(x)))


def elu(x: ArrayLike, alpha: float = 1.0) -> Any:
    """Elu function."""
    return _wrap(nn.elu(_to_tensor(x), alpha=alpha))


def celu(x: ArrayLike, alpha: float = 1.0) -> Any:
    """Celu function."""
    return _wrap(nn.celu(_to_tensor(x), alpha=alpha))


def selu(x: ArrayLike) -> Any:
    """Selu function."""
    return _wrap(nn.selu(_to_tensor(x)))


def log_softmax(x: ArrayLike, axis: int = -1) -> Any:
    """log_softmax function."""
    return _wrap(nn.log_softmax(_to_tensor(x), dim=axis))
