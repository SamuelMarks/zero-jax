"""Linear algebra operations for ``zero_jax.numpy.linalg``."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, astype, ndarray

Deprecated = "Deprecated"


def cholesky(a: ArrayLikeSpec, *, upper: bool = False, symmetrize_input: bool = True) -> ndarray:
    """Compute the Cholesky decomposition of a Hermitian positive-definite matrix.

    Args:
        a: Input Hermitian positive-definite matrix.
        upper: If ``True``, return the upper-triangular Cholesky factor.
        symmetrize_input: Whether to symmetrize ``a`` before decomposition.

    Returns:
        ndarray: Lower (or upper) triangular Cholesky factor.
    """
    t = _to_tensor(a)
    if symmetrize_input:
        t = _to_tensor(0.5 * (_wrap(t) + matrix_transpose(_wrap(t))))
    res = _wrap(ops.cholesky(t))
    if upper:
        return matrix_transpose(res)
    return res


def det(a: ArrayLikeSpec) -> ndarray:
    """Compute the determinant of an array.

    Args:
        a: Input square matrix or batch of square matrices.

    Returns:
        ndarray: Determinant of ``a``.
    """
    return _wrap(ops.det(_to_tensor(a)))


def eigh(a: ArrayLikeSpec, UPLO: str = "L", symmetrize_input: bool = True) -> tuple[ndarray, ndarray]:
    """Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    Args:
        a: Complex Hermitian or real symmetric matrix.
        UPLO: Specifies whether the calculation uses the lower (``"L"``) or upper (``"U"``) triangle.
        symmetrize_input: Whether to symmetrize the input matrix before decomposition.

    Returns:
        tuple[ndarray, ndarray]: Tuple of ``(eigenvalues, eigenvectors)``.
    """
    t = _to_tensor(a)
    if symmetrize_input:
        t = _to_tensor(0.5 * (_wrap(t) + matrix_transpose(_wrap(t))))
    return _wrap(ops.eigh(t))


def eigvals(a: ArrayLikeSpec) -> ndarray:
    """Compute the eigenvalues of a general matrix.

    Args:
        a: Input square matrix or batch of matrices.

    Returns:
        ndarray: Eigenvalues of ``a``.
    """
    return _wrap(ops.eigvals(_to_tensor(a)))


def eigvalsh(a: ArrayLikeSpec, UPLO: str = "L") -> ndarray:
    """Compute the eigenvalues of a Hermitian or real symmetric matrix.

    Args:
        a: Input Hermitian or symmetric matrix.
        UPLO: Specifies whether the lower (``"L"``) or upper (``"U"``) triangle is used.

    Returns:
        ndarray: Eigenvalues in ascending order.
    """
    return _wrap(ops.eigvalsh(_to_tensor(a)))


def inv(a: ArrayLikeSpec) -> ndarray:
    """Compute the multiplicative inverse of a matrix.

    Args:
        a: Square matrix to invert.

    Returns:
        ndarray: Multiplicative inverse of ``a``.
    """
    return _wrap(ops.inv(_to_tensor(a)))


def matrix_power(a: ArrayLikeSpec, n: int) -> ndarray:
    """Raise a square matrix to the integer power ``n``.

    Args:
        a: Input square matrix.
        n: Integer exponent.

    Returns:
        ndarray: Matrix ``a`` raised to the power ``n``.
    """
    return _wrap(ops.matrix_power(_to_tensor(a), n))


def pinv(a: ArrayLikeSpec, rcond: ArrayLikeSpec = None, hermitian: bool = False) -> ndarray:
    """Compute the Moore-Penrose pseudoinverse of a matrix.

    Args:
        a: Matrix or stack of matrices to be pseudo-inverted.
        rcond: Cutoff for small singular values.
        hermitian: If ``True``, ``a`` is assumed Hermitian/symmetric.

    Returns:
        ndarray: Pseudoinverse of ``a``.
    """
    return _wrap(ops.pinv(_to_tensor(a)))


def qr(a: ArrayLikeSpec, mode: str = "reduced") -> tuple[ndarray, ndarray] | ndarray:
    """Compute the QR factorization of a matrix.

    Args:
        a: Matrix to factorize.
        mode: Factorization mode (``"reduced"``, ``"complete"``, ``"r"``, or ``"raw"``).

    Returns:
        tuple[ndarray, ndarray] | ndarray: Tuple ``(Q, R)`` or matrix ``R`` when ``mode == "r"``.
    """
    res = _wrap(ops.qr(_to_tensor(a)))
    if mode == "r" and isinstance(res, tuple) and len(res) == 2:
        return res[1]
    return res


def slogdet(a: ArrayLikeSpec) -> tuple[ndarray, ndarray]:
    """Compute the sign and natural logarithm of the determinant of an array.

    Args:
        a: Input square matrix or batch of square matrices.

    Returns:
        tuple[ndarray, ndarray]: Tuple of ``(sign, logabsdet)``.
    """
    return _wrap(ops.slogdet(_to_tensor(a)))


def solve(a: ArrayLikeSpec, b: ArrayLikeSpec) -> ndarray:
    """Solve a linear matrix equation ``a @ x = b``.

    Args:
        a: Coefficient square matrix.
        b: Ordinate or dependent variable values.

    Returns:
        ndarray: Solution to the system ``a @ x = b``.
    """
    return _wrap(ops.solve(_to_tensor(a), _to_tensor(b)))


def svd(a: ArrayLikeSpec, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False) -> tuple[ndarray, ndarray, ndarray] | ndarray:
    """Compute the Singular Value Decomposition of a matrix.

    Args:
        a: Matrix to decompose.
        full_matrices: Whether ``u`` and ``vh`` have full or reduced shapes.
        compute_uv: Whether to compute ``u`` and ``vh`` in addition to singular values ``s``.
        hermitian: Whether ``a`` is known to be Hermitian.

    Returns:
        tuple[ndarray, ndarray, ndarray] | ndarray: Tuple ``(u, s, vh)`` or singular values ``s`` if ``compute_uv=False``.
    """
    if not compute_uv:
        return svdvals(a)
    res = _wrap(ops.svd(_to_tensor(a)))
    if not full_matrices and isinstance(res, tuple) and len(res) == 3:
        u, s, vh = res
        k = s.shape[-1]
        return u[..., :k], s, vh[..., :k, :]
    return res


def cond(x: ArrayLikeSpec, p: ArrayLikeSpec = None) -> ndarray:
    """Compute the condition number of a matrix.

    Args:
        x: Matrix whose condition number is computed.
        p: Order of the norm used in the condition number computation.

    Returns:
        ndarray: Condition number of the matrix.
    """
    from ml_switcheroo_compiler.ops.base import get_op

    return _wrap(get_op("Cond")()(_to_tensor(x), p))


def cross(x1: ArrayLikeSpec, x2: ArrayLikeSpec, axis: ArrayLikeSpec = -1) -> ndarray:
    """Compute the cross product of two (arrays of) 3D vectors.

    Args:
        x1: First vector array.
        x2: Second vector array.
        axis: Axis defining the vectors.

    Returns:
        ndarray: Vector cross product.
    """
    return _wrap(ops.cross(_to_tensor(x1), _to_tensor(x2)))


def diagonal(x: ArrayLikeSpec, offset: ArrayLikeSpec = 0) -> ndarray:
    """Extract specified diagonals from a matrix or stack of matrices.

    Args:
        x: Input array with ``ndim >= 2``.
        offset: Offset of the diagonal from the main diagonal.

    Returns:
        ndarray: Array of diagonals.
    """
    return _wrap(ops.diagonal(_to_tensor(x), offset=offset))


def eig(a: ArrayLikeSpec) -> tuple[ndarray, ndarray]:
    """Compute the eigenvalues and right eigenvectors of a square array.

    Args:
        a: Matrices for which the eigenvalues and right eigenvectors will be computed.

    Returns:
        tuple[ndarray, ndarray]: Tuple of ``(eigenvalues, eigenvectors)``.
    """
    return _wrap(ops.eig(_to_tensor(a)))


def lstsq(a: ArrayLikeSpec, b: ArrayLikeSpec, rcond: ArrayLikeSpec = None, numpy_resid: ArrayLikeSpec = False) -> tuple[ndarray, ...]:
    """Return the least-squares solution to a linear matrix equation.

    Args:
        a: Coefficient matrix.
        b: Ordinate or dependent variable values.
        rcond: Cutoff ratio for small singular values.
        numpy_resid: Whether to compute residuals in NumPy-compatible format.

    Returns:
        tuple[ndarray, ...]: Least-squares solution tuple.
    """
    return _wrap(ops.lstsq(_to_tensor(a), _to_tensor(b), rcond=rcond))


def matmul(x1: ArrayLikeSpec, x2: ArrayLikeSpec, precision: ArrayLikeSpec = None, preferred_element_type: ArrayLikeSpec = None) -> ndarray:
    """Compute the matrix product of two arrays.

    Args:
        x1: First input array.
        x2: Second input array.
        precision: Optional XLA precision configuration.
        preferred_element_type: Optional accumulation/output dtype.

    Returns:
        ndarray: Matrix product of ``x1`` and ``x2``.
    """
    res = _wrap(ops.matmul(_to_tensor(x1), _to_tensor(x2)))
    if preferred_element_type is not None:
        return astype(res, preferred_element_type)
    return res


def matrix_norm(x: ArrayLikeSpec, keepdims: ArrayLikeSpec = False, ord: ArrayLikeSpec = "fro") -> ndarray:
    """Compute the matrix norm of a matrix or stack of matrices.

    Args:
        x: Input array with ``ndim >= 2``.
        keepdims: Whether reduced axes are left with size one.
        ord: Order of the norm (e.g., ``"fro"``, ``"nuc"``, ``1``, ``2``, ``inf``).

    Returns:
        ndarray: Norm of the matrix.
    """
    return _wrap(ops.matrix_norm(_to_tensor(x), keepdims=keepdims))


def matrix_rank(M: ArrayLikeSpec, rtol: ArrayLikeSpec = None, tol: ArrayLikeSpec = Deprecated) -> ndarray:
    """Return matrix rank of array using SVD method.

    Args:
        M: Input vector or matrix.
        rtol: Relative threshold for singular values.
        tol: Deprecated alias for ``rtol``.

    Returns:
        ndarray: Rank of ``M``.
    """
    effective_tol = rtol if rtol is not None else (None if tol == Deprecated else tol)
    return _wrap(ops.matrix_rank(_to_tensor(M), tol=effective_tol))


def matrix_transpose(x: ArrayLikeSpec) -> ndarray:
    """Transpose each matrix in a stack of matrices (swapping the last two dimensions).

    Args:
        x: Input array with ``ndim >= 2``.

    Returns:
        ndarray: Matrix-transposed array.
    """
    return _wrap(ops.matrix_transpose(_to_tensor(x)))


def multi_dot(arrays: ArrayLikeSpec, precision: ArrayLikeSpec = None) -> ndarray:
    """Compute the dot product of two or more arrays in a single function call.

    Args:
        arrays: Sequence of 2D or 1D arrays to multiply.
        precision: Optional XLA precision specification.

    Returns:
        ndarray: Chained matrix product.
    """
    return _wrap(ops.multi_dot([_to_tensor(arr) for arr in arrays]))


def norm(x: ArrayLikeSpec, ord: ArrayLikeSpec = None, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False) -> ndarray:
    """Compute a matrix or vector norm.

    Args:
        x: Input array.
        ord: Order of the norm.
        axis: Axis or tuple of axes along which to compute the norm.
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        ndarray: Norm of ``x``.
    """
    return _wrap(ops.norm(_to_tensor(x), ord, axis, keepdims))


def outer(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ndarray:
    """Compute the outer product of two 1D vectors.

    Args:
        x1: First 1D input vector.
        x2: Second 1D input vector.

    Returns:
        ndarray: Outer product matrix.
    """
    return _wrap(ops.outer(_to_tensor(x1), _to_tensor(x2)))


def svdvals(x: ArrayLikeSpec) -> ndarray:
    """Compute the singular values of a matrix or stack of matrices.

    Args:
        x: Input matrix or batch of matrices.

    Returns:
        ndarray: Singular values in descending order.
    """
    return _wrap(ops.svdvals(_to_tensor(x)))


def tensordot(
    x1: ArrayLikeSpec,
    x2: ArrayLikeSpec,
    axes: ArrayLikeSpec = 2,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ndarray:
    """Compute tensor dot product along specified axes.

    Args:
        x1: First tensor operand.
        x2: Second tensor operand.
        axes: Number of axes to sum over or pair of axis sequences.
        precision: Optional XLA precision setting.
        preferred_element_type: Optional output dtype.

    Returns:
        ndarray: Contracted tensor result.
    """
    res = _wrap(ops.tensordot(_to_tensor(x1), _to_tensor(x2), axes=axes))
    if preferred_element_type is not None:
        return astype(res, preferred_element_type)
    return res


def tensorinv(a: ArrayLikeSpec, ind: ArrayLikeSpec = 2) -> ndarray:
    """Compute the multiplicative inverse of a tensor.

    Args:
        a: Tensor to invert.
        ind: Number of first indices that are involved in the inverse sum.

    Returns:
        ndarray: Tensor inverse of ``a``.
    """
    return _wrap(ops.tensorinv(_to_tensor(a), ind))


def tensorsolve(a: ArrayLikeSpec, b: ArrayLikeSpec, axes: ArrayLikeSpec = None) -> ndarray:
    """Solve the tensor equation ``a x = b`` for ``x``.

    Args:
        a: Coefficient tensor.
        b: Right-hand side tensor.
        axes: Axes of ``a`` to reorder to the right before inversion.

    Returns:
        ndarray: Solution tensor ``x``.
    """
    return _wrap(ops.tensorsolve(_to_tensor(a), _to_tensor(b), axes))


def trace(x: ArrayLikeSpec, offset: ArrayLikeSpec = 0, dtype: ArrayLikeSpec = None) -> ndarray:
    """Return the sum along diagonals of the array.

    Args:
        x: Input array from which the diagonal is taken.
        offset: Offset of the diagonal from the main diagonal.
        dtype: Optional data type of the returned array.

    Returns:
        ndarray: Sum along the diagonal.
    """
    res = _wrap(ops.trace(_to_tensor(x), offset=offset))
    if dtype is not None:
        return astype(res, dtype)
    return res


def vecdot(
    x1: ArrayLikeSpec,
    x2: ArrayLikeSpec,
    axis: ArrayLikeSpec = -1,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ndarray:
    """Compute the (complex-conjugated) vector dot product of two arrays.

    Args:
        x1: First vector array.
        x2: Second vector array.
        axis: Axis along which the dot product is computed.
        precision: Optional XLA precision setting.
        preferred_element_type: Optional output dtype.

    Returns:
        ndarray: Vector dot product.
    """
    res = _wrap(ops.vecdot(_to_tensor(x1), _to_tensor(x2), axis=axis))
    if preferred_element_type is not None:
        return astype(res, preferred_element_type)
    return res


def vector_norm(x: ArrayLikeSpec, axis: ArrayLikeSpec = None, keepdims: ArrayLikeSpec = False, ord: ArrayLikeSpec = 2) -> ndarray:
    """Compute the vector norm of a vector or batch of vectors.

    Args:
        x: Input array.
        axis: Axis or tuple of axes along which to compute the vector norm.
        keepdims: Whether reduced axes are retained with size 1.
        ord: Order of the norm (default ``2``).

    Returns:
        ndarray: Vector norm of ``x``.
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
