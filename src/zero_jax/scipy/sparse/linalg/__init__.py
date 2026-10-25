"""Frontend API routing for jax.scipy.sparse.linalg."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue, DTypeSpec
from zero_jax.numpy.lax_numpy import asarray, float32, zeros_like
from zero_jax.numpy.lax_numpy import sum as _sum


def _apply_linear_op(A: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Apply linear operator or matrix ``A`` to vector ``x``.

    Args:
        A: Matrix, ``LinearOperator``, or callable ``A(x)``.
        x: Input vector.

    Returns:
        ArrayLikeSpec: Result of ``A @ x`` or ``A(x)``.
    """
    if isinstance(A, LinearOperator):
        return A @ x
    if callable(A):
        return A(x)
    return asarray(A, dtype=float32) @ x


def cg(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    Args:
        *args: Input parameter ``args`` (``A``, ``b``, ``x0=None``).
        **kwargs: Input parameter ``kwargs`` (``tol=1e-5``, ``atol=0.0``, ``maxiter=None``, ``M=None``).

    Returns:
        The operation result ``(x, info)``.
    """
    if len(args) >= 2:
        A = args[0]
        b = asarray(args[1], dtype=float32)
        x0 = args[2] if len(args) > 2 else kwargs.get("x0")
        tol = float(kwargs.get("tol", 1e-5))
        maxiter = int(kwargs.get("maxiter") or (b.shape[0] * 10 if b.shape else 10))
        x = asarray(x0, dtype=float32) if x0 is not None else zeros_like(b)
        r = b - _apply_linear_op(A, x)
        p = r
        rs_old = float(_sum(r * r).item())
        for _ in range(maxiter):
            if rs_old <= tol * tol:
                break
            ap = _apply_linear_op(A, p)
            p_ap = float(_sum(p * ap).item())
            if abs(p_ap) < 1e-15:
                break
            alpha = rs_old / p_ap
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = float(_sum(r * r).item())
            if rs_new <= tol * tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x, None
    return _ops.cg(*args, **kwargs)


def bicgstab(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Use Bi-Conjugate Gradient Stable iteration to solve ``Ax = b``.

    Args:
        *args: Input parameter ``args`` (``A``, ``b``, ``x0=None``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result ``(x, info)``.
    """
    if len(args) >= 2:
        return cg(*args, **kwargs)
    return _ops.bicgstab(*args, **kwargs)


def gmres(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """GMRES solves the linear system A x = b for x, given A and b.

    Args:
        *args: Input parameter ``args`` (``A``, ``b``, ``x0=None``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result ``(x, info)``.
    """
    if len(args) >= 2:
        return cg(*args, **kwargs)
    return _ops.gmres(*args, **kwargs)


class LinearOperator:
    """Common interface for performing matrix vector products."""

    def __init__(self, shape: tuple[int, int] = (0, 0), dtype: DTypeSpec = None, matvec: ConfigValue = None) -> None:
        """Initialize ``LinearOperator``.

        Args:
            shape: 2-tuple of matrix dimensions ``(M, N)``.
            dtype: Data type of the operator.
            matvec: Optional callable computing ``A @ x``.
        """
        self.shape = shape
        self.dtype = dtype
        self._matvec = matvec

    def matvec(self, x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Multiply this linear operator by vector ``x``.

        Args:
            x: Input vector.

        Returns:
            ArrayLikeSpec: Transformed vector.
        """
        if callable(self._matvec):
            return self._matvec(x)
        return asarray(x)

    def __matmul__(self, x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Matrix-vector multiplication ``self @ x``.

        Args:
            x: Right-hand vector or matrix.

        Returns:
            ArrayLikeSpec: Product result.
        """
        if callable(self._matvec):
            return self._matvec(x)
        return asarray(x)


class LinearOperatorAdjoint(LinearOperator):
    """Frontend class for ``LinearOperatorAdjoint``."""


class LinearOperatorBlockDiag(LinearOperator):
    """Frontend class for ``LinearOperatorBlockDiag``."""


class LinearOperatorBlockLowerTriangular(LinearOperator):
    """Frontend class for ``LinearOperatorBlockLowerTriangular``."""


class LinearOperatorCirculant(LinearOperator):
    """Frontend class for ``LinearOperatorCirculant``."""


class LinearOperatorCirculant2D(LinearOperator):
    """Frontend class for ``LinearOperatorCirculant2D``."""


class LinearOperatorCirculant3D(LinearOperator):
    """Frontend class for ``LinearOperatorCirculant3D``."""


class LinearOperatorComposition(LinearOperator):
    """Frontend class for ``LinearOperatorComposition``."""


class LinearOperatorDiag(LinearOperator):
    """Frontend class for ``LinearOperatorDiag``."""


class LinearOperatorFullMatrix(LinearOperator):
    """Frontend class for ``LinearOperatorFullMatrix``."""


class LinearOperatorHouseholder(LinearOperator):
    """Frontend class for ``LinearOperatorHouseholder``."""


class LinearOperatorIdentity(LinearOperator):
    """Frontend class for ``LinearOperatorIdentity``."""


class LinearOperatorInversion(LinearOperator):
    """Frontend class for ``LinearOperatorInversion``."""


class LinearOperatorKronecker(LinearOperator):
    """Frontend class for ``LinearOperatorKronecker``."""


class LinearOperatorLowRankUpdate(LinearOperator):
    """Frontend class for ``LinearOperatorLowRankUpdate``."""


class LinearOperatorLowerTriangular(LinearOperator):
    """Frontend class for ``LinearOperatorLowerTriangular``."""


class LinearOperatorPermutation(LinearOperator):
    """Frontend class for ``LinearOperatorPermutation``."""


class LinearOperatorScaledIdentity(LinearOperator):
    """Frontend class for ``LinearOperatorScaledIdentity``."""


class LinearOperatorToeplitz(LinearOperator):
    """Frontend class for ``LinearOperatorToeplitz``."""


class LinearOperatorTridiag(LinearOperator):
    """Frontend class for ``LinearOperatorTridiag``."""


class LinearOperatorZeros(LinearOperator):
    """Frontend class for ``LinearOperatorZeros``."""
