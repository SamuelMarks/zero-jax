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

ArrayLike = Any


def _erf(x: Any) -> Any:
    """_erf function."""
    return _wrap(ops.erf(_to_tensor(x)))


def gelu(x: ArrayLike, approximate: bool = False) -> Any:
    """Gelu function."""
    x_t = _to_tensor(x)
    # x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
    half = creation.full_like(x_t, 0.5)
    one = creation.full_like(x_t, 1.0)
    sqrt2 = creation.full_like(x_t, math.sqrt(2.0))

    x_div_sqrt2 = ops.divide(x_t, sqrt2)
    erf_val = ops.erf(x_div_sqrt2)
    one_plus_erf = ops.add(one, erf_val)
    x_half = ops.multiply(x_t, half)
    return _wrap(ops.multiply(x_half, one_plus_erf))


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
    x = _to_tensor(x)
    return _wrap(ops.maximum(x, _to_tensor(0.0)))


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
    # x * sigmoid(x)
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(sigmoid(x))))


def silu(x: ArrayLike) -> Any:
    """Silu function."""
    return swish(x)


def elu(x: ArrayLike, alpha: float = 1.0) -> Any:
    """Elu function."""
    x = _to_tensor(x)
    # alpha * (exp(x) - 1) for x < 0, x for x >= 0
    # ONNX has Elu natively or we compose
    if hasattr(ops, "elu"):  # pragma: no cover
        return _wrap(ops.elu(x, alpha=alpha))
    import numpy as np

    if hasattr(x, "data") and isinstance(x.data, np.ndarray):
        res = np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1.0))
        return _wrap(ml_switcheroo.Tensor(res, res.shape, x.dtype, x.device))
    raise NotImplementedError()  # pragma: no cover


def celu(x: ArrayLike, alpha: float = 1.0) -> Any:
    """Celu function."""
    x = _to_tensor(x)
    if hasattr(ops, "celu"):  # pragma: no cover
        return _wrap(ops.celu(x, alpha=alpha))
    import numpy as np

    if hasattr(x, "data") and isinstance(x.data, np.ndarray):
        res = np.maximum(0, x.data) + np.minimum(
            0, alpha * (np.exp(x.data / alpha) - 1.0)
        )
        return _wrap(ml_switcheroo.Tensor(res, res.shape, x.dtype, x.device))
    raise NotImplementedError()  # pragma: no cover


def selu(x: ArrayLike) -> Any:
    """Selu function."""
    x = _to_tensor(x)
    if hasattr(ops, "selu"):  # pragma: no cover
        return _wrap(ops.selu(x))
    import numpy as np

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    if hasattr(x, "data") and isinstance(x.data, np.ndarray):
        res = scale * np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1.0))
        return _wrap(ml_switcheroo.Tensor(res, res.shape, x.dtype, x.device))
    raise NotImplementedError()  # pragma: no cover


def log_softmax(x: ArrayLike, axis: int = -1) -> Any:
    """log_softmax function."""
    x = _to_tensor(x)
    if hasattr(ops, "log_softmax"):  # pragma: no cover
        return _wrap(ops.log_softmax(x, axis=axis))
    # log(softmax(x)) eagerly
    import numpy as np
    import scipy.special

    if hasattr(x, "data") and isinstance(x.data, np.ndarray):
        res = scipy.special.log_softmax(x.data, axis=axis)
        return _wrap(ml_switcheroo.Tensor(res, res.shape, x.dtype, x.device))
    raise NotImplementedError()  # pragma: no cover
