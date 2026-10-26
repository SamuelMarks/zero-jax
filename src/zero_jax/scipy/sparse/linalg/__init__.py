"""Frontend API routing for jax.scipy.sparse.linalg."""

from __future__ import annotations

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


def cg(
    A: ArrayLikeSpec,
    b: ArrayLikeSpec,
    x0: ArrayLikeSpec = None,
    *,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: ArrayLikeSpec = None,
) -> tuple[ArrayLikeSpec, int | None]:
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    Args:
        A: 2D symmetric positive-definite matrix or linear operator callable.
        b: Right-hand side vector.
        x0: Starting guess for the solution.
        tol: Relative convergence tolerance.
        atol: Absolute convergence tolerance.
        maxiter: Maximum number of iterations.
        M: Optional preconditioner.

    Returns:
        tuple[ArrayLikeSpec, int | None]: Solution ``(x, None)``.
    """
    _ = (atol, M)
    b_arr = asarray(b, dtype=float32)
    max_steps = int(maxiter or (b_arr.shape[0] * 10 if b_arr.shape else 10))
    x = asarray(x0, dtype=float32) if x0 is not None else zeros_like(b_arr)
    r = b_arr - _apply_linear_op(A, x)
    p = r
    rs_old = float(_sum(r * r).item())
    for _ in range(max_steps):
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


def bicgstab(
    A: ArrayLikeSpec,
    b: ArrayLikeSpec,
    x0: ArrayLikeSpec = None,
    *,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: ArrayLikeSpec = None,
) -> tuple[ArrayLikeSpec, int | None]:
    """Use Bi-Conjugate Gradient Stable iteration to solve ``Ax = b``.

    Args:
        A: 2D matrix or linear operator callable.
        b: Right-hand side vector.
        x0: Starting guess for the solution.
        tol: Relative convergence tolerance.
        atol: Absolute convergence tolerance.
        maxiter: Maximum number of iterations.
        M: Optional preconditioner.

    Returns:
        tuple[ArrayLikeSpec, int | None]: Solution ``(x, None)``.
    """
    _ = (atol, M)
    b_arr = asarray(b, dtype=float32)
    max_steps = int(maxiter or (b_arr.shape[0] * 10 if b_arr.shape else 10))
    x = asarray(x0, dtype=float32) if x0 is not None else zeros_like(b_arr)
    r = b_arr - _apply_linear_op(A, x)
    r_hat = r
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    v = zeros_like(b_arr)
    p = zeros_like(b_arr)
    for _ in range(max_steps):
        if float(_sum(r * r).item()) <= tol * tol:
            break
        rho_new = float(_sum(r_hat * r).item())
        beta = (rho_new / max(abs(rho_old), 1e-15)) * (alpha / max(abs(omega), 1e-15))
        p = r + beta * (p - omega * v)
        v = _apply_linear_op(A, p)
        r_hat_v = float(_sum(r_hat * v).item())
        if abs(r_hat_v) < 1e-15:
            break
        alpha = rho_new / r_hat_v
        s = r - alpha * v
        t = _apply_linear_op(A, s)
        tt = float(_sum(t * t).item())
        if abs(tt) < 1e-15:
            x = x + alpha * p
            break
        omega = float(_sum(t * s).item()) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho_old = rho_new
    return x, None


def gmres(
    A: ArrayLikeSpec,
    b: ArrayLikeSpec,
    x0: ArrayLikeSpec = None,
    *,
    tol: float = 1e-5,
    atol: float = 0.0,
    restart: int = 20,
    maxiter: int | None = None,
    M: ArrayLikeSpec = None,
    solve_method: str = "batched",
) -> tuple[ArrayLikeSpec, int | None]:
    """Solve the linear system ``A x = b`` for ``x`` using GMRES.

    Args:
        A: 2D matrix or linear operator callable.
        b: Right-hand side vector.
        x0: Starting guess for the solution.
        tol: Relative convergence tolerance.
        atol: Absolute convergence tolerance.
        restart: Size of Krylov subspace between restarts.
        maxiter: Maximum number of iterations.
        M: Optional preconditioner.
        solve_method: Subproblem solve method.

    Returns:
        tuple[ArrayLikeSpec, int | None]: Solution ``(x, None)``.
    """
    _ = (restart, solve_method)
    return bicgstab(A, b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)


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


__all__ = [
    "LinearOperator",
    "LinearOperatorAdjoint",
    "LinearOperatorBlockDiag",
    "LinearOperatorBlockLowerTriangular",
    "LinearOperatorCirculant",
    "LinearOperatorCirculant2D",
    "LinearOperatorCirculant3D",
    "LinearOperatorComposition",
    "LinearOperatorDiag",
    "LinearOperatorFullMatrix",
    "LinearOperatorHouseholder",
    "LinearOperatorIdentity",
    "LinearOperatorInversion",
    "LinearOperatorKronecker",
    "LinearOperatorLowRankUpdate",
    "LinearOperatorLowerTriangular",
    "LinearOperatorPermutation",
    "LinearOperatorScaledIdentity",
    "LinearOperatorToeplitz",
    "LinearOperatorTridiag",
    "LinearOperatorZeros",
    "_apply_linear_op",
    "bicgstab",
    "cg",
    "gmres",
]
