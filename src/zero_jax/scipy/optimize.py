"""SciPy optimize submodule."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from ml_switcheroo_compiler.ops.registry import get_op

from zero_jax._types import ArrayLike, VariadicCallable
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, ndarray


class OptimizeResults(typing.NamedTuple):
    """Result of optimization.

    Attributes:
        x: The solution of the optimization.
        success: Whether or not the optimizer exited successfully.
        status: Termination status of the optimizer.
        fun: Values of objective function.
        jac: Values of objective function Jacobian.
        hess_inv: Inverse of objective function Hessian.
        nfev: Number of evaluations of the objective functions.
        njev: Number of evaluations of the Jacobian.
        nit: Number of iterations performed by the optimizer.
    """

    x: ndarray
    success: ndarray
    status: ndarray
    fun: ndarray
    jac: ndarray
    hess_inv: ndarray
    nfev: ndarray
    njev: ndarray
    nit: ndarray


def minimize(
    fun: VariadicCallable[ArrayLike],
    x0: ArrayLike,
    args: tuple[ArrayLike, ...] = (),
    *,
    method: str,
    tol: float | None = None,
    options: Mapping[str, float | int] | None = None,
) -> OptimizeResults:
    """Minimization of scalar function of one or more variables.

    Args:
        fun: The objective function to be minimized.
        x0: Initial guess.
        args: Extra arguments passed to the objective function and its derivatives.
        method: Type of solver.
        tol: Tolerance for termination.
        options: A dictionary of solver options.

    Returns:
        The optimization result represented as a ``OptimizeResults`` object.
    """
    op = get_op("ScipyOptimizeMinimize")
    x0_t = _to_tensor(x0)

    res = op(fun, x0_t, args, method, tol, options)

    return OptimizeResults(
        x=_wrap(res[0]),
        success=_wrap(res[1]),
        status=_wrap(res[2]),
        fun=_wrap(res[3]),
        jac=_wrap(res[4]),
        hess_inv=_wrap(res[5]),
        nfev=_wrap(res[6]),
        njev=_wrap(res[7]),
        nit=_wrap(res[8]),
    )


__all__ = ["OptimizeResults", "minimize"]
