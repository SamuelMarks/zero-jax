"""Activation functions and related utilities."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_creation as creation
import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

_ArrayLike = ArrayLikeSpec


def _erf(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the error function of the given input.

    Args:
        x (ArrayLikeSpec): The input array-like object.

    Returns:
        ArrayLikeSpec: An array of the same shape as `x` containing the error function values.
    """
    return _wrap(ops.erf(_to_tensor(x)))


def gelu(x: _ArrayLike, approximate: bool = True) -> ArrayLikeSpec:
    """Computes the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (_ArrayLike): The input array.
        approximate (bool): If True, uses the approximate GELU formulation based on tanh. Defaults to False.

    Returns:
        ArrayLikeSpec: The array after applying the GELU activation.
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
    # 0.5 * x * (1 + erf(x / sqrt(2)))
    const1 = ops.multiply(x_t, _to_tensor(0.5))
    erf_inner = ops.divide(x_t, _to_tensor(math.sqrt(2.0)))
    erf_val = ops.erf(erf_inner)
    return _wrap(ops.multiply(const1, ops.add(_to_tensor(1.0), erf_val)))


def logsumexp(
    a: _ArrayLike,
    axis: ArrayLikeSpec = None,
    b: _ArrayLike | None = None,
    keepdims: bool = False,
    return_sign: bool = False,
    where: _ArrayLike | None = None,
) -> ArrayLikeSpec:
    """Computes the log of the sum of exponentials of input elements.

    Args:
        a (_ArrayLike): Input array.
        axis (ArrayLikeSpec, optional): Axis or axes over which the sum is computed. By default, computes the sum over all elements.
        b (Optional[_ArrayLike], optional): Array of weights for the elements of `a`. Defaults to None.
        keepdims (bool): If True, retains reduced dimensions with length 1. Defaults to False.
        return_sign (bool): If True, returns a tuple of (result, sign). Defaults to False.
        where (Optional[_ArrayLike], optional): Elements to include in the sum. Defaults to None.

    Returns:
        ArrayLikeSpec: The logsumexp of the inputs. If `return_sign` is True, returns a tuple containing the logsumexp and its sign.
    """
    a_t = _to_tensor(a)
    if where is not None:
        a_t = ops.where(_to_tensor(where), a_t, creation.full_like(a_t, -float("inf")))
    amax = ops.max(a_t, axis=axis, keepdims=True)
    a_shifted = ops.subtract(a_t, amax)
    terms = ops.multiply(ops.exp(a_shifted), _to_tensor(b)) if b is not None else ops.exp(a_shifted)
    if where is not None:
        terms = ops.where(_to_tensor(where), terms, creation.zeros_like(terms))

    # sum
    sum_exp = ops.sum(terms, axis=axis, keepdims=keepdims)
    sign = ops.sign(sum_exp)
    log_sum_exp = ops.log(ops.abs(sum_exp) if return_sign else sum_exp)

    if not keepdims:
        amax = ops.squeeze(amax, dims=[axis] if isinstance(axis, int) else axis) if axis is not None else ops.squeeze(amax)

    res = ops.add(log_sum_exp, amax)

    if return_sign:
        return _wrap(res), _wrap(sign)
    return _wrap(res)


def _to_dtype(dtype: ArrayLikeSpec) -> ArrayLikeSpec:
    """Converts the given dtype object to the underlying DType class representation.

    Args:
        dtype (ArrayLikeSpec): A dtype specified as string, type, or DType instance.

    Returns:
        ArrayLikeSpec: The parsed DType object.
    """
    from ml_switcheroo_compiler.core.dtype import DType

    from zero_jax._types import NamedProtocol

    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, str):
        return DType(dtype)
    if dtype is float:
        return DType("float32")
    if dtype is int:
        return DType("int32")
    if dtype is bool:
        return DType("bool")
    if isinstance(dtype, type):
        return DType(dtype.__name__)
    if isinstance(dtype, NamedProtocol):
        return DType(dtype.name)
    return DType(str(dtype))


def one_hot(
    x: ArrayLikeSpec,
    num_classes: int,
    *,
    dtype: ArrayLikeSpec = float,
    axis: ArrayLikeSpec = -1,
    out_sharding: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Creates a one-hot encoding of the given integer array.

    Args:
        x (ArrayLikeSpec): Integer array of class indices.
        num_classes (int): Number of total classes.
        dtype (ArrayLikeSpec, optional): The data type of the output. Defaults to float.
        axis (ArrayLikeSpec, optional): The axis along which the one-hot dimension is added. Defaults to -1.
        out_sharding (ArrayLikeSpec, optional): Optional output sharding specification. Defaults to None.

    Returns:
        ArrayLikeSpec: A one-hot encoded array.
    """
    _ = out_sharding
    x_t = _to_tensor(x)
    classes = creation.arange(0, num_classes, dtype=x_t.dtype)
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
    x: _ArrayLike,
    axis: ArrayLikeSpec = -1,
    where: ArrayLikeSpec | None = None,
) -> ArrayLikeSpec:
    """Computes the softmax activation function over the given axis.

    Args:
        x (_ArrayLike): Input array.
        axis (ArrayLikeSpec, optional): Axis along which the softmax is computed. Defaults to -1.
        where (Optional[ArrayLikeSpec], optional): Elements to include in the computation. Defaults to None.

    Returns:
        ArrayLikeSpec: Array of the same shape as `x` with softmax applied.
    """
    x_t = _to_tensor(x)
    if where is not None:
        x_t = ops.where(_to_tensor(where), x_t, creation.full_like(x_t, -float("inf")))
    lse = ops.logsumexp(x_t, axis=axis, keepdims=True)
    res = ops.exp(ops.subtract(x_t, lse))
    if where is not None:
        res = ops.where(_to_tensor(where), res, creation.zeros_like(res))
    return _wrap(res)


def dot_product_attention(
    query: ArrayLikeSpec,
    key: ArrayLikeSpec,
    value: ArrayLikeSpec,
    bias: ArrayLikeSpec = None,
    mask: ArrayLikeSpec = None,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    query_seq_lengths: ArrayLikeSpec = None,
    key_value_seq_lengths: ArrayLikeSpec = None,
    local_window_size: int | tuple[int, int] | None = None,
    return_residual: bool = False,
    implementation: str | None = None,
) -> ArrayLikeSpec:
    """Compute scaled dot-product attention ``softmax((Q K^T) * scale + bias) V``.

    Args:
        query: Query tensor of shape ``(..., T, N, H)`` or ``(..., T, H)``.
        key: Key tensor of shape ``(..., S, K, H)`` or ``(..., S, H)``.
        value: Value tensor of shape matching ``key``.
        bias: Optional additive attention bias.
        mask: Optional boolean mask where ``True`` keeps logits and ``False`` masks them out.
        scale: Multiplicative scale for attention logits (defaults to ``1 / sqrt(H)``).
        is_causal: Whether to apply a lower-triangular causal mask.
        query_seq_lengths: Optional sequence lengths for queries.
        key_value_seq_lengths: Optional sequence lengths for keys and values.
        local_window_size: Optional local attention window size.
        return_residual: Whether to also return the logsumexp residual.
        implementation: Optional backend implementation string.

    Returns:
        ArrayLikeSpec: Attention output array, or ``(output, residual)`` if ``return_residual=True``.
    """
    _ = (query_seq_lengths, key_value_seq_lengths, local_window_size, implementation)
    from zero_jax.numpy import arange, asarray, einsum, moveaxis, tril, where

    q = asarray(query)
    k = asarray(key)
    v = asarray(value)
    head_dim = q.shape[-1]
    eff_scale = (1.0 / math.sqrt(float(head_dim))) if scale is None else float(scale)

    if q.ndim >= 3 and k.ndim >= 3 and q.shape[-2] == k.shape[-2]:
        q_m = moveaxis(q, -2, -3)
        k_m = moveaxis(k, -2, -3)
        v_m = moveaxis(v, -2, -3)
        logits = einsum("...th,...sh->...ts", q_m, k_m) * eff_scale
        if bias is not None:
            logits = logits + asarray(bias)
        if is_causal:
            t_len = q.shape[-3]
            s_len = k.shape[-3]
            causal = arange(t_len)[:, None] >= arange(s_len)[None, :]
            logits = where(causal, logits, -1e9)
        if mask is not None:
            logits = where(asarray(mask), logits, -1e9)
        weights = softmax(logits, axis=-1)
        out = moveaxis(einsum("...ts,...sh->...th", weights, v_m), -3, -2)
        if return_residual:
            return out, logsumexp(logits, axis=-1)
        return out

    logits = einsum("...th,...sh->...ts", q, k) * eff_scale
    if bias is not None:
        logits = logits + asarray(bias)
    if is_causal:
        causal_mask = tril(creation.ones((logits.shape[-2], logits.shape[-1])))
        logits = where(_wrap(causal_mask) > 0, logits, -1e9)
    if mask is not None:
        logits = where(asarray(mask), logits, -1e9)
    weights = softmax(logits, axis=-1)
    out = einsum("...ts,...sh->...th", weights, v)
    if return_residual:
        return out, logsumexp(logits, axis=-1)
    return out


def get_scaled_dot_general_config(
    mode: str = "nvfp4",
    global_scale: ArrayLikeSpec = None,
) -> dict[str, ArrayLikeSpec]:
    """Return a block-scaling configuration dictionary for :func:`scaled_dot_general`.

    Args:
        mode: Block scaling mode (``"nvfp4"``, ``"mxfp8"``, or ``"mxfp4"``).
        global_scale: Optional global scaling tensor.

    Returns:
        dict[str, ArrayLikeSpec]: Configuration dictionary describing the scaling mode.
    """
    return {"mode": mode, "global_scale": global_scale}


def scaled_dot_general(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    preferred_element_type: ArrayLikeSpec = "float32",
    configs: ArrayLikeSpec = None,
    implementation: str | None = None,
) -> ArrayLikeSpec:
    """Compute a generalized dot product with optional block-scale configurations.

    Args:
        lhs: Left-hand side array.
        rhs: Right-hand side array.
        dimension_numbers: Contracting and batch dimension numbers ``((lhs_c, rhs_c), (lhs_b, rhs_b))``.
        preferred_element_type: Accumulation and output dtype.
        configs: Optional list of scaling configurations.
        implementation: Optional backend implementation name.

    Returns:
        ArrayLikeSpec: Generalized dot product output.
    """
    _ = (configs, implementation)
    from zero_jax.lax import dot_general

    return dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        preferred_element_type=preferred_element_type,
    )


def scaled_matmul(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    lhs_scales: ArrayLikeSpec = None,
    rhs_scales: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = "float32",
) -> ArrayLikeSpec:
    """Compute block-scaled matrix multiplication of ``lhs`` and ``rhs``.

    Args:
        lhs: Left-hand side operand.
        rhs: Right-hand side operand.
        lhs_scales: Scale factors for ``lhs``.
        rhs_scales: Scale factors for ``rhs``.
        preferred_element_type: Output element type.

    Returns:
        ArrayLikeSpec: Scaled matrix product.
    """
    from zero_jax.numpy import asarray, astype, matmul

    l_arr = asarray(lhs)
    r_arr = asarray(rhs)
    if lhs_scales is not None:
        l_arr = l_arr * asarray(lhs_scales)
    if rhs_scales is not None:
        r_arr = r_arr * asarray(rhs_scales)
    res = matmul(l_arr, r_arr)
    if preferred_element_type is not None:
        return astype(res, preferred_element_type)
    return res


def sigmoid(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the sigmoid activation function.

    Args:
        x (ArrayLikeSpec): Input array-like object.

    Returns:
        ArrayLikeSpec: Array of the same shape as `x` with sigmoid applied.
    """
    x_t = _to_tensor(x)
    one = creation.full_like(x_t, 1.0)
    neg_x = ops.negative(x_t)
    exp_neg_x = ops.exp(neg_x)
    denom = ops.add(one, exp_neg_x)
    return _wrap(ops.divide(one, denom))


def log_sigmoid(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the logarithm of the sigmoid function.

    Args:
        x (ArrayLikeSpec): Input array-like object.

    Returns:
        ArrayLikeSpec: Array with the log-sigmoid applied.
    """
    x_t = _to_tensor(x)
    # log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    one = creation.full_like(x_t, 1.0)
    neg_x = ops.negative(x_t)
    exp_neg_x = ops.exp(neg_x)
    denom = ops.add(one, exp_neg_x)
    return _wrap(ops.negative(ops.log(denom)))


def relu(x: _ArrayLike) -> ArrayLikeSpec:
    """Computes the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        ArrayLikeSpec: Array with ReLU applied.
    """
    x_t = _to_tensor(x)
    return _wrap(ops.maximum(x_t, _to_tensor(0.0)))


def relu6(x: _ArrayLike) -> ArrayLikeSpec:
    """Computes the ReLU6 activation function, capping at 6.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        ArrayLikeSpec: Array with ReLU6 applied.
    """
    x = _to_tensor(x)
    return _wrap(ops.minimum(ops.maximum(x, _to_tensor(0.0)), _to_tensor(6.0)))


def hard_sigmoid(x: _ArrayLike) -> ArrayLikeSpec:
    """Computes the hard sigmoid activation function.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        ArrayLikeSpec: Array with hard sigmoid applied.
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


def hard_tanh(x: _ArrayLike) -> ArrayLikeSpec:
    """Computes the hard tanh activation function, bounding the input between -1 and 1.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        ArrayLikeSpec: Array with hard tanh applied.
    """
    x = _to_tensor(x)
    return _wrap(ops.maximum(_to_tensor(-1.0), ops.minimum(_to_tensor(1.0), x)))


def swish(x: _ArrayLike) -> ArrayLikeSpec:
    """Computes the Swish activation function (x * sigmoid(x)).

    Args:
        x (_ArrayLike): Input array.

    Returns:
        ArrayLikeSpec: Array with Swish applied.
    """
    x_t = _to_tensor(x)
    return _wrap(ops.multiply(x_t, _to_tensor(sigmoid(x_t))))


def silu(x: _ArrayLike) -> ArrayLikeSpec:
    """Computes the SiLU (Sigmoid Linear Unit) activation function, which is identical to Swish.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        ArrayLikeSpec: Array with SiLU applied.
    """
    return swish(x)


def elu(x: _ArrayLike, alpha: float = 1.0) -> ArrayLikeSpec:
    """Computes the Exponential Linear Unit (ELU) activation function.

    Args:
        x (_ArrayLike): Input array.
        alpha (float, optional): Scaling factor for negative values. Defaults to 1.0.

    Returns:
        ArrayLikeSpec: Array with ELU applied.
    """
    x_t = _to_tensor(x)
    pos = ops.maximum(x_t, _to_tensor(0.0))
    neg = ops.multiply(
        _to_tensor(alpha),
        ops.subtract(ops.exp(ops.minimum(x_t, _to_tensor(0.0))), _to_tensor(1.0)),
    )
    return _wrap(ops.add(pos, neg))


def celu(x: _ArrayLike, alpha: float = 1.0) -> ArrayLikeSpec:
    """Computes the Continuously Differentiable Exponential Linear Unit (CELU) activation function.

    Args:
        x (_ArrayLike): Input array.
        alpha (float, optional): Scaling factor controlling the curvature for negative values. Defaults to 1.0.

    Returns:
        ArrayLikeSpec: Array with CELU applied.
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


def selu(x: _ArrayLike) -> ArrayLikeSpec:
    """Computes the Scaled Exponential Linear Unit (SELU) activation function.

    Args:
        x (_ArrayLike): Input array.

    Returns:
        ArrayLikeSpec: Array with SELU applied.
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


def log_softmax(x: _ArrayLike, axis: int = -1) -> ArrayLikeSpec:
    """Computes the logarithm of the softmax activation function.

    Args:
        x (_ArrayLike): Input array.
        axis (int, optional): Axis along which to compute the log-softmax. Defaults to -1.

    Returns:
        ArrayLikeSpec: Array of the same shape as `x` with log-softmax applied.
    """
    x_t = _to_tensor(x)
    lse = ops.logsumexp(x_t, axis=axis, keepdims=True)
    return _wrap(ops.subtract(x_t, lse))


def tanh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Hyperbolic tangent activation function.

    Args:
        x: Input array.

    Returns:
        The hyperbolic tangent of x.
    """
    from zero_jax.numpy.lax_numpy import tanh as _tanh

    return _tanh(x)
