"""SciPy optimize submodule."""

import typing
from collections.abc import Mapping
from typing import Any, Callable, Optional

from ml_switcheroo_compiler.ops.registry import get_op

from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


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

    x: Any
    success: Any
    status: Any
    fun: Any
    jac: Any
    hess_inv: Any
    nfev: Any
    njev: Any
    nit: Any


def minimize(
    fun: Callable[..., Any],
    x0: Any,
    args: tuple[Any, ...] = (),
    *,
    method: str,
    tol: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
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
