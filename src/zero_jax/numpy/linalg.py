"""Module documentation."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

Deprecated = "Deprecated"


def cholesky(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for cholesky.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.cholesky(_to_tensor(a)))


def det(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for det.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.det(_to_tensor(a)))


def eigh(a: ArrayLikeSpec, UPLO: str = "L", symmetrize_input: bool = True) -> ArrayLikeSpec:
    """JAX API implementation for eigh.

    Args:
        a: Argument a.
        UPLO: Argument UPLO.
        symmetrize_input: Argument symmetrize_input.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.eigh(_to_tensor(a)))


def eigvals(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for eigvals.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.eigvals(_to_tensor(a)))


def eigvalsh(a: ArrayLikeSpec, UPLO: str = "L") -> ArrayLikeSpec:
    """JAX API implementation for eigvalsh.

    Args:
        a: Argument a.
        UPLO: Argument UPLO.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.eigvalsh(_to_tensor(a)))


def inv(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for inv.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.inv(_to_tensor(a)))


def matrix_power(a: ArrayLikeSpec, n: int) -> ArrayLikeSpec:
    """JAX API implementation for matrix_power.

    Args:
        a: Argument a.
        n: Argument n.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.matrix_power(_to_tensor(a), n))


def pinv(a: ArrayLikeSpec, rcond: ArrayLikeSpec = None, hermitian: bool = False) -> ArrayLikeSpec:
    """JAX API implementation for pinv.

    Args:
        a: Argument a.
        rcond: Argument rcond.
        hermitian: Argument hermitian.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.pinv(_to_tensor(a)))


def qr(a: ArrayLikeSpec, mode: str = "reduced") -> ArrayLikeSpec:
    """JAX API implementation for qr.

    Args:
        a: Argument a.
        mode: Argument mode.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.qr(_to_tensor(a)))


def slogdet(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for slogdet.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.slogdet(_to_tensor(a)))


def solve(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for solve.

    Args:
        a: Argument a.
        b: Argument b.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.solve(_to_tensor(a), _to_tensor(b)))


def svd(a: ArrayLikeSpec, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False) -> ArrayLikeSpec:
    """JAX API implementation for svd.

    Args:
        a: Argument a.
        full_matrices: Argument full_matrices.
        compute_uv: Argument compute_uv.
        hermitian: Argument hermitian.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.svd(_to_tensor(a)))


def cond(x: ArrayLikeSpec, p: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for cond.

    Args:
        x: Argument x.
        p: Argument p.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    from ml_switcheroo_compiler.ops.base import get_op

    return _wrap(get_op("Cond")()(_to_tensor(x), p))


def cross(x1: ArrayLikeSpec, x2: ArrayLikeSpec, axis: ArrayLikeSpec = -1) -> ArrayLikeSpec:
    """JAX API implementation for cross.

    Args:
        x1: Argument x1.
        x2: Argument x2.
        axis: Argument axis.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.cross(_to_tensor(x1), _to_tensor(x2)))


def diagonal(x: ArrayLikeSpec, offset: ArrayLikeSpec = 0) -> ArrayLikeSpec:
    """JAX API implementation for diagonal.

    Args:
        x: Argument x.
        offset: Argument offset.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.diagonal(_to_tensor(x), offset=offset))


def eig(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for eig.

    Args:
        a: Argument a.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.eig(_to_tensor(a)))


def lstsq(a: ArrayLikeSpec, b: ArrayLikeSpec, rcond: ArrayLikeSpec = None, numpy_resid: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """JAX API implementation for lstsq.

    Args:
        a: Argument a.
        b: Argument b.
        rcond: Argument rcond.
        numpy_resid: Argument numpy_resid.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.lstsq(_to_tensor(a), _to_tensor(b), rcond=rcond))


def matmul(x1: ArrayLikeSpec, x2: ArrayLikeSpec, precision: ArrayLikeSpec = None, preferred_element_type: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for matmul.

    Args:
        x1: Argument x1.
        x2: Argument x2.
        precision: Argument precision.
        preferred_element_type: Argument preferred_element_type.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.matmul(_to_tensor(x1), _to_tensor(x2)))


def matrix_norm(x: ArrayLikeSpec, keepdims: ArrayLikeSpec = False, ord: ArrayLikeSpec = "fro") -> ArrayLikeSpec:
    """JAX API implementation for matrix_norm.

    Args:
        x: Argument x.
        keepdims: Argument keepdims.
        ord: Argument ord.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.matrix_norm(_to_tensor(x), keepdims=keepdims))


def matrix_rank(M: ArrayLikeSpec, rtol: ArrayLikeSpec = None, tol: ArrayLikeSpec = Deprecated) -> ArrayLikeSpec:
    """JAX API implementation for matrix_rank.

    Args:
        M: Argument M.
        rtol: Argument rtol.
        tol: Argument tol.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.matrix_rank(_to_tensor(M), tol=rtol))


def matrix_transpose(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for matrix_transpose.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.matrix_transpose(_to_tensor(x)))


def multi_dot(arrays: ArrayLikeSpec, precision: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for multi_dot.

    Args:
        arrays: Argument arrays.
        precision: Argument precision.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.multi_dot([_to_tensor(arr) for arr in arrays]))


def norm(x: ArrayLikeSpec, ord: ArrayLikeSpec = None, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ArrayLikeSpec:
    """JAX API implementation for norm.

    Args:
        x: Argument x.
        ord: Argument ord.
        axis: Argument axis.
        keepdims: Argument keepdims.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.norm(_to_tensor(x), ord, axis, keepdims))


def outer(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for outer.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.outer(_to_tensor(x1), _to_tensor(x2)))


def svdvals(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for svdvals.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.svdvals(_to_tensor(x)))


def tensordot(
    x1: ArrayLikeSpec,
    x2: ArrayLikeSpec,
    axes: ArrayLikeSpec = 2,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for tensordot.

    Args:
        x1: Argument x1.
        x2: Argument x2.
        axes: Argument axes.
        precision: Argument precision.
        preferred_element_type: Argument preferred_element_type.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.tensordot(_to_tensor(x1), _to_tensor(x2), axes=axes))


def tensorinv(a: ArrayLikeSpec, ind: ArrayLikeSpec = 2) -> ArrayLikeSpec:
    """JAX API implementation for tensorinv.

    Args:
        a: Argument a.
        ind: Argument ind.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.tensorinv(_to_tensor(a), ind))


def tensorsolve(a: ArrayLikeSpec, b: ArrayLikeSpec, axes: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for tensorsolve.

    Args:
        a: Argument a.
        b: Argument b.
        axes: Argument axes.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.tensorsolve(_to_tensor(a), _to_tensor(b), axes))


def trace(x: ArrayLikeSpec, offset: ArrayLikeSpec = 0, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for trace.

    Args:
        x: Argument x.
        offset: Argument offset.
        dtype: Argument dtype.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.trace(_to_tensor(x), offset=offset))


def vecdot(
    x1: ArrayLikeSpec,
    x2: ArrayLikeSpec,
    axis: ArrayLikeSpec = -1,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for vecdot.

    Args:
        x1: Argument x1.
        x2: Argument x2.
        axis: Argument axis.
        precision: Argument precision.
        preferred_element_type: Argument preferred_element_type.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.vecdot(_to_tensor(x1), _to_tensor(x2), axis=axis))


def vector_norm(x: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False, ord: ArrayLikeSpec = 2) -> ArrayLikeSpec:
    """JAX API implementation for vector_norm.

    Args:
        x: Argument x.
        axis: Argument axis.
        keepdims: Argument keepdims.
        ord: Argument ord.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.vector_norm(_to_tensor(x), ord=ord, axis=axis, keepdims=keepdims))


__all__ = [
    "cholesky",
    "cond",
    "cross",
    "det",
    "diagonal",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "inv",
    "lstsq",
    "matmul",
    "matrix_norm",
    "matrix_power",
    "matrix_rank",
    "matrix_transpose",
    "multi_dot",
    "norm",
    "outer",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "svd",
    "svdvals",
    "tensordot",
    "tensorinv",
    "tensorsolve",
    "trace",
    "vecdot",
    "vector_norm",
]
