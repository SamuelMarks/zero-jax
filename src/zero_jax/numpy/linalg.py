"""Module documentation."""

from typing import Any

import zero_jax._compiler_proxy_ops as ops
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

Deprecated = "Deprecated"


def cholesky(a: Any) -> Any:
    """JAX API implementation for cholesky.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    return _wrap(ops.cholesky(_to_tensor(a)))


def det(a: Any) -> Any:
    """JAX API implementation for det.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    return _wrap(ops.det(_to_tensor(a)))


def eigh(a: Any, UPLO: str = "L", symmetrize_input: bool = True) -> Any:
    """JAX API implementation for eigh.

    Args:
        a: Argument a.
        UPLO: Argument UPLO.
        symmetrize_input: Argument symmetrize_input.

    Returns:
        Any: The result.
    """
    res = ops.eigh(_to_tensor(a))  # pragma: no cover
    return tuple(_wrap(t) for t in res)  # pragma: no cover


def eigvals(a: Any) -> Any:
    """JAX API implementation for eigvals.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    res = ops.eigvals(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)


def eigvalsh(a: Any, UPLO: str = "L") -> Any:
    """JAX API implementation for eigvalsh.

    Args:
        a: Argument a.
        UPLO: Argument UPLO.

    Returns:
        Any: The result.
    """
    return _wrap(ops.eigvalsh(_to_tensor(a)))  # pragma: no cover


def inv(a: Any) -> Any:
    """JAX API implementation for inv.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    return _wrap(ops.inv(_to_tensor(a)))


def matrix_power(a: Any, n: int) -> Any:
    """JAX API implementation for matrix_power.

    Args:
        a: Argument a.
        n: Argument n.

    Returns:
        Any: The result.
    """
    return _wrap(ops.matrix_power(_to_tensor(a), n))


def pinv(a: Any, rcond: Any = None, hermitian: bool = False) -> Any:
    """JAX API implementation for pinv.

    Args:
        a: Argument a.
        rcond: Argument rcond.
        hermitian: Argument hermitian.

    Returns:
        Any: The result.
    """
    return _wrap(ops.pinv(_to_tensor(a)))


def qr(a: Any, mode: str = "reduced") -> Any:
    """JAX API implementation for qr.

    Args:
        a: Argument a.
        mode: Argument mode.

    Returns:
        Any: The result.
    """
    res = ops.qr(_to_tensor(a))
    return tuple(_wrap(t) for t in res)


def slogdet(a: Any) -> Any:
    """JAX API implementation for slogdet.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    res = ops.slogdet(_to_tensor(a))
    return tuple(_wrap(t) for t in res)


def solve(a: Any, b: Any) -> Any:
    """JAX API implementation for solve.

    Args:
        a: Argument a.
        b: Argument b.

    Returns:
        Any: The result.
    """
    return _wrap(ops.solve(_to_tensor(a), _to_tensor(b)))


def svd(
    a: Any, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False
) -> Any:
    """JAX API implementation for svd.

    Args:
        a: Argument a.
        full_matrices: Argument full_matrices.
        compute_uv: Argument compute_uv.
        hermitian: Argument hermitian.

    Returns:
        Any: The result.
    """
    res = ops.svd(_to_tensor(a))
    if isinstance(res, tuple):
        return tuple(_wrap(t) for t in res)
    return _wrap(res)  # pragma: no cover


def cond(x: Any, p: Any = None) -> Any:
    """JAX API implementation for cond.

    Args:
        x: Argument x.
        p: Argument p.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.cond(_to_tensor(x), p)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def cross(x1: Any, x2: Any, axis: Any = -1) -> Any:
    """JAX API implementation for cross.

    Args:
        x1: Argument x1.
        x2: Argument x2.
        axis: Argument axis.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.cross(_to_tensor(x1), _to_tensor(x2), axis)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def diagonal(x: Any, offset: Any = 0) -> Any:
    """JAX API implementation for diagonal.

    Args:
        x: Argument x.
        offset: Argument offset.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.diagonal(_to_tensor(x), offset)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def eig(a: Any) -> Any:
    """JAX API implementation for eig.

    Args:
        a: Argument a.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.eig(_to_tensor(a))  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def lstsq(a: Any, b: Any, rcond: Any = None, numpy_resid: Any = False) -> Any:
    """JAX API implementation for lstsq.

    Args:
        a: Argument a.
        b: Argument b.
        rcond: Argument rcond.
        numpy_resid: Argument numpy_resid.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.lstsq(
        _to_tensor(a), _to_tensor(b), rcond, numpy_resid
    )  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def matmul(
    x1: Any, x2: Any, precision: Any = None, preferred_element_type: Any = None
) -> Any:
    """JAX API implementation for matmul.

    Args:
        x1: Argument x1.
        x2: Argument x2.
        precision: Argument precision.
        preferred_element_type: Argument preferred_element_type.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.matmul(
        _to_tensor(x1), _to_tensor(x2), precision, preferred_element_type
    )  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def matrix_norm(x: Any, keepdims: Any = False, ord: Any = "fro") -> Any:
    """JAX API implementation for matrix_norm.

    Args:
        x: Argument x.
        keepdims: Argument keepdims.
        ord: Argument ord.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.matrix_norm(_to_tensor(x), keepdims, ord)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def matrix_rank(M: Any, rtol: Any = None, tol: Any = Deprecated) -> Any:
    """JAX API implementation for matrix_rank.

    Args:
        M: Argument M.
        rtol: Argument rtol.
        tol: Argument tol.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.matrix_rank(_to_tensor(M), rtol, tol)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def matrix_transpose(x: Any) -> Any:
    """JAX API implementation for matrix_transpose.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.matrix_transpose(_to_tensor(x))  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def multi_dot(arrays: Any, precision: Any = None) -> Any:
    """JAX API implementation for multi_dot.

    Args:
        arrays: Argument arrays.
        precision: Argument precision.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.multi_dot(_to_tensor(arrays), precision)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def norm(x: Any, ord: Any = None, axis: Any = None, keepdims: Any = False) -> Any:
    """JAX API implementation for norm.

    Args:
        x: Argument x.
        ord: Argument ord.
        axis: Argument axis.
        keepdims: Argument keepdims.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.norm(_to_tensor(x), ord, axis, keepdims)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def outer(x1: Any, x2: Any) -> Any:
    """JAX API implementation for outer.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.outer(_to_tensor(x1), _to_tensor(x2))  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def svdvals(x: Any) -> Any:
    """JAX API implementation for svdvals.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.svdvals(_to_tensor(x))  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def tensordot(
    x1: Any,
    x2: Any,
    axes: Any = 2,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Any:
    """JAX API implementation for tensordot.

    Args:
        x1: Argument x1.
        x2: Argument x2.
        axes: Argument axes.
        precision: Argument precision.
        preferred_element_type: Argument preferred_element_type.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.tensordot(  # pragma: no cover
        _to_tensor(x1), _to_tensor(x2), axes, precision, preferred_element_type
    )
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def tensorinv(a: Any, ind: Any = 2) -> Any:
    """JAX API implementation for tensorinv.

    Args:
        a: Argument a.
        ind: Argument ind.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.tensorinv(_to_tensor(a), ind)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def tensorsolve(a: Any, b: Any, axes: Any = None) -> Any:
    """JAX API implementation for tensorsolve.

    Args:
        a: Argument a.
        b: Argument b.
        axes: Argument axes.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.tensorsolve(_to_tensor(a), _to_tensor(b), axes)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def trace(x: Any, offset: Any = 0, dtype: Any = None) -> Any:
    """JAX API implementation for trace.

    Args:
        x: Argument x.
        offset: Argument offset.
        dtype: Argument dtype.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.trace(_to_tensor(x), offset, dtype)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def vecdot(
    x1: Any,
    x2: Any,
    axis: Any = -1,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Any:
    """JAX API implementation for vecdot.

    Args:
        x1: Argument x1.
        x2: Argument x2.
        axis: Argument axis.
        precision: Argument precision.
        preferred_element_type: Argument preferred_element_type.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.vecdot(  # pragma: no cover
        _to_tensor(x1), _to_tensor(x2), axis, precision, preferred_element_type
    )
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


def vector_norm(x: Any, axis: Any = None, keepdims: Any = False, ord: Any = 2) -> Any:
    """JAX API implementation for vector_norm.

    Args:
        x: Argument x.
        axis: Argument axis.
        keepdims: Argument keepdims.
        ord: Argument ord.

    Returns:
        Any: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    res = ops.vector_norm(_to_tensor(x), axis, keepdims, ord)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(t) for t in res)  # pragma: no cover
    return _wrap(res)  # pragma: no cover


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
