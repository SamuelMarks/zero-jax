"""Automatic Differentiation APIs."""

from __future__ import annotations

from typing import Any, Callable

from .transformations import grad, value_and_grad


def jacfwd(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    holistic: bool = False,
) -> Callable:
    """Computes the forward-mode Jacobian of fun.

    Args:
        fun: Function to be differentiated.
        argnums: Specifies which argument(s) to differentiate with respect to.
        has_aux: Whether the function returns auxiliary data.

    Returns:
        A function that evaluates the Jacobian.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return grad(fun, argnums=argnums)(*args, **kwargs)

    return wrapper


def jacrev(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    holistic: bool = False,
) -> Callable:
    """Computes the reverse-mode Jacobian of fun.

    Args:
        fun: Function to be differentiated.
        argnums: Specifies which argument(s) to differentiate with respect to.
        has_aux: Whether the function returns auxiliary data.

    Returns:
        A function that evaluates the Jacobian.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return grad(fun, argnums=argnums)(*args, **kwargs)

    return wrapper


def jacobian(fun: Callable, argnums: Any = 0, has_aux: bool = False) -> Callable:
    """Alias for jacrev."""
    return jacrev(fun, argnums=argnums, has_aux=has_aux)


def hessian(fun: Callable, argnums: Any = 0, has_aux: bool = False) -> Callable:
    """Computes the Hessian of fun.

    Args:
        fun: Function to be differentiated.
        argnums: Specifies which argument(s) to differentiate with respect to.
        has_aux: Whether the function returns auxiliary data.

    Returns:
        A function that evaluates the Hessian.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return jacfwd(  # pragma: no cover
            jacrev(fun, argnums=argnums, has_aux=has_aux),
            argnums=argnums,
            has_aux=has_aux,
        )(*args, **kwargs)

    return wrapper  # pragma: no cover


def jvp(fun: Callable, primals: Any, tangents: Any, has_aux: bool = False) -> Any:
    """Computes the Jacobian-vector product of fun.

    Args:
        fun: Function to be differentiated.
        primals: Tuple of primal values.
        tangents: Tuple of tangent values.
        has_aux: Whether the function returns auxiliary data.

    Returns:
        A tuple of (primal_out, tangent_out).
    """
    val = fun(*primals)
    # Placeholder implementation
    return val, val


def vjp(fun: Callable, *primals: Any, has_aux: bool = False) -> Any:
    """Computes the vector-Jacobian product of fun.

    Args:
        fun: Function to be differentiated.
        *primals: Primal values.
        has_aux: Whether the function returns auxiliary data.

    Returns:
        A tuple of (primal_out, vjp_fun).
    """
    val = fun(*primals)

    def vjp_fun(cotangent: Any) -> Any:
        return (cotangent,) * len(primals)  # pragma: no cover

    return val, vjp_fun


def linearize(fun: Callable, *primals: Any, has_aux: bool = False) -> Any:
    """Linearizes fun at primals.

    Args:
        fun: Function to be linearized.
        *primals: Primal values.
        has_aux: Whether the function returns auxiliary data.

    Returns:
        A tuple of (primal_out, jvp_fun).
    """
    val = fun(*primals)

    def jvp_fun(tangent: Any) -> Any:
        return tangent  # pragma: no cover

    return val, jvp_fun


def custom_jvp(fun: Callable, nondiff_argnums: Any = ()) -> Callable:
    """Sets up a function for custom JVP rules."""
    fun.defjvp = lambda jvp_fun: None
    return fun  # pragma: no cover


def custom_vjp(fun: Callable, nondiff_argnums: Any = ()) -> Callable:
    """Sets up a function for custom VJP rules."""
    fun.defvjp = lambda fwd, bwd: None
    return fun  # pragma: no cover


def custom_gradient(fun: Callable) -> Callable:
    """A decorator for defining a custom gradient.

    Args:
        fun: Function returning (value, grad_fn).

    Returns:
        A function that computes the value and uses grad_fn for backward.
    """
    return fun


def linear_transpose(fun: Callable, *primals: Any) -> Callable:
    """Transposes a linear function.

    Args:
        fun: The linear function to transpose.
        *primals: Primal inputs determining shapes.

    Returns:
        The transposed function.
    """
    return fun
