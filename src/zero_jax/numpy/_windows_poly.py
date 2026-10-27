"""Window functions, polynomial operations, contractions, and linear algebra products for zero_jax.numpy."""

from __future__ import annotations

import builtins
from collections.abc import Callable

import ml_switcheroo_compiler as _ml_switcheroo_compiler
from ml_switcheroo_compiler import Tensor
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import TensorConfig as _TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import (
    ArrayLikeSpec,
    ArrayProtocol,
    ConfigValue,
    DTypeSpec,
    VariadicCallable,
)
from zero_jax.numpy._creation import arange, array, asarray, ones
from zero_jax.numpy._dtypes import float32
from zero_jax.numpy._indexing_set import where
from zero_jax.numpy._math import cos, i0, maximum, sqrt
from zero_jax.numpy._ndarray import _to_tensor, _wrap
from zero_jax.numpy._reductions import sum
from zero_jax.numpy.polynomial import poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub, polyval, roots
from zero_jax.numpy.tensor_utils import to_array as _to_array


def bartlett(M: int) -> ArrayLikeSpec:
    """Return the Bartlett window.

    Args:
        M: Number of points in the output window.

    Returns:
        The window.
    """
    if M < 1:
        return array([], dtype=float32)
    if M == 1:
        return ones((1,), dtype=float32)
    n = arange(0, M, dtype=float32)
    half = (M - 1) / 2.0
    return where(n <= half, 2.0 * n / (M - 1), 2.0 - 2.0 * n / (M - 1))


def blackman(M: int) -> ArrayLikeSpec:
    """Return the Blackman window.

    Args:
        M: Number of points in the output window.

    Returns:
        The window.
    """
    if M < 1:
        return array([], dtype=float32)
    if M == 1:
        return ones((1,), dtype=float32)
    import math

    n = arange(0, M, dtype=float32)
    factor = 2.0 * math.pi / (M - 1)
    return 0.42 - 0.5 * cos(factor * n) + 0.08 * cos(2.0 * factor * n)


def hamming(M: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the Hamming window of length ``M``.

    Args:
        M: Number of points in the output window.

    Returns:
        ArrayLikeSpec: The Hamming window array.
    """
    m_int = int(M)
    if m_int < 1:
        return array([], dtype=float32)
    if m_int == 1:
        return ones((1,), dtype=float32)
    import math

    n = arange(0, m_int, dtype=float32)
    return 0.54 - 0.46 * cos(2.0 * math.pi * n / (m_int - 1))


def hanning(M: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the Hanning (Hann) window of length ``M``.

    Args:
        M: Number of points in the output window.

    Returns:
        ArrayLikeSpec: The Hanning window array.
    """
    m_int = int(M)
    if m_int < 1:
        return array([], dtype=float32)
    if m_int == 1:
        return ones((1,), dtype=float32)
    import math

    n = arange(0, m_int, dtype=float32)
    return 0.5 - 0.5 * cos(2.0 * math.pi * n / (m_int - 1))


def kaiser(M: int, beta: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the Kaiser window.

    Args:
        M: Number of points in the output window.
        beta: Shape parameter for window.

    Returns:
        The window.
    """
    if M < 1:
        return array([], dtype=float32)
    if M == 1:
        return ones((1,), dtype=float32)
    n = arange(0, M, dtype=float32)
    alpha = (M - 1) / 2.0
    ratio = (n - alpha) / alpha
    arg = beta * sqrt(maximum(1.0 - ratio * ratio, 0.0))
    return i0(arg) / i0(beta)


def convolve(
    a: ArrayLikeSpec,
    v: ArrayLikeSpec,
    mode: str = "full",
    *,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Returns the discrete, linear convolution of two one-dimensional sequences.

    Args:
        a: First input sequence.
        v: Second input sequence.
        mode: 'full', 'valid', or 'same'.
        precision: Numerical precision.
        preferred_element_type: Optional dtype.

    Returns:
        Discrete, linear convolution of a and v.
    """
    # Pass kwargs supported by JAX if applicable, otherwise base ops
    return _wrap(ops.convolve(_to_tensor(a), _to_tensor(v), mode=mode))


def correlate(
    a: ArrayLikeSpec,
    v: ArrayLikeSpec,
    mode: str = "valid",
    *,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        a: First input sequence.
        v: Second input sequence.
        mode: 'valid', 'same', or 'full'.
        precision: Numerical precision.
        preferred_element_type: Optional dtype.

    Returns:
        Discrete cross-correlation of a and v.
    """
    return _wrap(ops.correlate(_to_tensor(a), _to_tensor(v), mode=mode))


def cov(
    m: ArrayLikeSpec,
    y: ArrayLikeSpec = None,
    rowvar: ArrayLikeSpec = True,
    bias: ArrayLikeSpec = False,
    ddof: ArrayLikeSpec = None,
    fweights: ArrayLikeSpec = None,
    aweights: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Estimate a covariance matrix, given data and weights.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
        y: An additional set of variables and observations.
        rowvar: If rowvar is True, each row represents a variable.
        bias: Default normalization is False.
        ddof: Degrees of freedom.
        fweights: Frequency weights.
        aweights: Observation vector weights.

    Returns:
        The covariance matrix of the variables.
    """
    y_tensor = _to_tensor(y) if y is not None else None
    return _wrap(
        ops.cov(
            _to_tensor(m),
            y=y_tensor,
            rowvar=rowvar,
            bias=bias,
            ddof=ddof,
            fweights=fweights,
            aweights=aweights,
        )
    )


def corrcoef(x: ArrayLikeSpec, y: ArrayLikeSpec = None, rowvar: ArrayLikeSpec = True) -> ArrayLikeSpec:
    """Return Pearson product-moment correlation coefficients.

    Args:
        x: A 1-D or 2-D array containing multiple variables and observations.
        y: An additional set of variables and observations.
        rowvar: If rowvar is True, each row represents a variable.

    Returns:
        The correlation coefficient matrix of the variables.
    """
    return _wrap(ops.corrcoef(_to_tensor(x), _to_tensor(y) if y is not None else None, rowvar=rowvar))


def einsum(subscripts: str, *operands: ArrayLikeSpec) -> ArrayLikeSpec:
    """Evaluate the Einstein summation convention on the operands.

    Args:
        subscripts: Argument subscripts.
        *operands: Argument operands.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    tensors = [_to_tensor(a) for a in operands]
    return _wrap(ops.einsum(subscripts, *tensors))


def einsum_path(subscripts: ArrayLikeSpec, *operands: ArrayLikeSpec, optimize: ArrayLikeSpec = "auto") -> ArrayLikeSpec:
    """Evaluates the lowest cost contraction order for an einsum expression.

    Args:
        subscripts: Specifies the subscripts for summation.
        *operands: These are the arrays for the operation.
        optimize: Choose the type of path.

    Returns:
        A tuple of (path, string_representation).
    """
    return ops.einsum_path(subscripts, *[_to_tensor(o) for o in operands], optimize=optimize)


def dot(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Dot product of two arrays.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    t_a = _to_tensor(a)
    t_b = _to_tensor(b)
    res_t = ops.dot(t_a, t_b)
    if isinstance(res_t.data, ProxyTensor) and (len(t_a.shape) >= 2 or len(t_b.shape) >= 2):
        out_shape = (*t_a.shape[:-1],) if len(t_b.shape) == 1 else (*t_a.shape[:-1], *t_b.shape[:-2], t_b.shape[-1])
        proxy = ProxyTensor(id=res_t.data.id, shape=out_shape, dtype=res_t.dtype.value)
        _tracer.active_graph.nodes[res_t.data.id].shape_metadata = out_shape
        res_t = Tensor(data=proxy, config=_TensorConfig(shape=out_shape, dtype=res_t.dtype, device=res_t.device))
    return _wrap(res_t)


def vdot(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the dot product of two vectors.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.vdot(_to_tensor(a), _to_tensor(b)))


def vecdot(x1: ArrayLikeSpec, x2: ArrayLikeSpec, /, *, axis: int = -1) -> ArrayLikeSpec:
    """Execute ``vecdot``.

    Args:
        x1: Input parameter ``x1``.
        x2: Input parameter ``x2``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    return sum(x1 * x2, axis=axis)


def inner(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Inner product of two arrays.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.inner(_to_tensor(a), _to_tensor(b)))


def outer(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the outer product of two vectors.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.outer(_to_tensor(a), _to_tensor(b)))


def matmul(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Matrix product of two arrays.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.matmul(_to_tensor(a), _to_tensor(b)))


def tensordot(a: ArrayLikeSpec, b: ArrayLikeSpec, axes: ArrayLikeSpec = 2) -> ArrayLikeSpec:
    """Compute tensor dot product along specified axes.

    Args:
        a (ArrayLikeSpec): Argument a.
        b (ArrayLikeSpec): Argument b.
        axes (ArrayLikeSpec): Argument axes.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.tensordot(_to_tensor(a), _to_tensor(b), axes=axes))


def kron(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """Kronecker product of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        ArrayLikeSpec: The Kronecker product of ``a`` and ``b``.
    """
    from zero_jax.numpy._manipulation import atleast_2d, reshape

    a_arr = asarray(a)
    b_arr = asarray(b)
    if a_arr.ndim == 1 and b_arr.ndim == 1:
        return reshape(outer(a_arr, b_arr), (int(a_arr.shape[0]) * int(b_arr.shape[0]),))
    a_2d = atleast_2d(a_arr)
    b_2d = atleast_2d(b_arr)
    r = int(a_2d.shape[0]) * int(b_2d.shape[0])
    c = int(a_2d.shape[1]) * int(b_2d.shape[1])
    prod = a_2d[:, None, :, None] * b_2d[None, :, None, :]
    return reshape(prod, (r, c))


def cross(a: ArrayLikeSpec, b: ArrayLikeSpec, axisa: int = -1, axisb: int = -1, axisc: int = -1, axis: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for cross.

    Args:
        a: Argument a.
        b: Argument b.
        axisa: Argument axisa.
        axisb: Argument axisb.
        axisc: Argument axisc.
        axis: Argument axis.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(
        ops.cross(
            _to_tensor(a),
            _to_tensor(b),
            axes={"axisa": axisa, "axisb": axisb, "axisc": axisc, "axis": axis},
        )
    )


def trace(
    a: ArrayLikeSpec,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: ArrayLikeSpec = None,
    out: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Execute ``trace``.

    Args:
        a: Input parameter ``a``.
        offset: Input parameter ``offset``.
        axis1: Input parameter ``axis1``.
        axis2: Input parameter ``axis2``.
        dtype: Input parameter ``dtype``.
        out: Input parameter ``out``.

    Returns:
        The operation result.
    """
    from ml_switcheroo_compiler.ops import trace as _trace

    return _trace(_to_tensor(a), offset=offset, axis1=axis1, axis2=axis2)


def matvec(x1: ArrayLikeSpec, x2: ArrayLikeSpec, /) -> ArrayLikeSpec:
    """Compute matrix-vector product of ``x1`` and ``x2``.

    Args:
        x1: Matrix array of shape ``(..., M, N)``.
        x2: Vector array of shape ``(..., N)``.

    Returns:
        ArrayLikeSpec: Matrix-vector product of shape ``(..., M)``.
    """
    return einsum("...ij,...j->...i", asarray(x1), asarray(x2))


def vecmat(x1: ArrayLikeSpec, x2: ArrayLikeSpec, /) -> ArrayLikeSpec:
    """Compute vector-matrix product of ``x1`` and ``x2``.

    Args:
        x1: Vector array of shape ``(..., M)``.
        x2: Matrix array of shape ``(..., M, N)``.

    Returns:
        ArrayLikeSpec: Vector-matrix product of shape ``(..., N)``.
    """
    return einsum("...i,...ij->...j", asarray(x1), asarray(x2))


__all__ = [
    "bartlett",
    "blackman",
    "convolve",
    "corrcoef",
    "correlate",
    "cov",
    "cross",
    "dot",
    "einsum",
    "einsum_path",
    "hamming",
    "hanning",
    "inner",
    "kaiser",
    "kron",
    "matmul",
    "matvec",
    "outer",
    "poly",
    "polyadd",
    "polyder",
    "polydiv",
    "polyfit",
    "polyint",
    "polymul",
    "polysub",
    "polyval",
    "roots",
    "tensordot",
    "trace",
    "vdot",
    "vecdot",
    "vecmat",
]
