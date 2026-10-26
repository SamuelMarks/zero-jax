"""Frontend API routing for jax.custom_derivatives."""

from __future__ import annotations

from collections.abc import Callable

from zero_jax._types import PyTree, VariadicCallable
from zero_jax.api.ad import CustomVJPWrapper
from zero_jax.api.ad import custom_gradient as _api_custom_gradient
from zero_jax.api.misc import closure_convert as _api_closure_convert
from zero_jax.lax.mock_p import MockPrimitive


class CustomVJPPrimal:
    """Primal value wrapper passed to a ``custom_vjp`` forward rule when ``symbolic_zeros=True``."""

    def __init__(self, value: PyTree = None, perturbed: bool = True) -> None:
        """Initialize ``CustomVJPPrimal``.

        Args:
            value: Underlying primal value.
            perturbed: Whether the primal has a non-zero tangent/perturbation.
        """
        self.value: PyTree = value
        self.perturbed: bool = bool(perturbed)


class SymbolicZero:
    """Representation of a symbolic zero tangent or cotangent in custom derivative rules."""

    def __init__(self, aval: PyTree = None) -> None:
        """Initialize ``SymbolicZero``.

        Args:
            aval: Abstract value (shape/dtype descriptor) of the zero tangent.
        """
        self.aval: PyTree = aval
        self.value: int = 0


def closure_convert(
    fun: VariadicCallable[PyTree],
    *example_args: PyTree,
) -> tuple[VariadicCallable[PyTree], list[PyTree]]:
    """Closure conversion utility, for use with higher-order custom derivatives.

    Args:
        fun: Function to closure-convert.
        *example_args: Example positional arguments passed to ``fun``.

    Returns:
        tuple[VariadicCallable[PyTree], list[PyTree]]: ``(converted_fun, residual_args)``.
    """
    return _api_closure_convert(fun, *example_args)


def custom_gradient(fun: VariadicCallable[PyTree]) -> CustomVJPWrapper:
    """Convenience function for defining custom VJP rules (aka custom gradients).

    Args:
        fun: Callable returning ``(primal_out, vjp_fn)``.

    Returns:
        CustomVJPWrapper: Wrapped function with custom VJP rule.
    """
    return _api_custom_gradient(fun)


class custom_jvp:
    """Set up a JAX-transformable function for a custom JVP rule definition."""

    def __init__(
        self,
        fun: VariadicCallable[PyTree] | None = None,
        nondiff_argnums: tuple[int, ...] = (),
    ) -> None:
        """Initialize ``custom_jvp``.

        Args:
            fun: Target primal function to wrap.
            nondiff_argnums: Tuple of non-differentiable argument indices.
        """
        self.fun: VariadicCallable[PyTree] | None = fun
        self.nondiff_argnums: tuple[int, ...] = tuple(nondiff_argnums)
        self.jvp: VariadicCallable[PyTree] | None = None

    def defjvp(
        self,
        jvp_rule: VariadicCallable[PyTree],
        symbolic_zeros: bool = False,
    ) -> VariadicCallable[PyTree]:
        """Define a custom JVP rule for this function.

        Args:
            jvp_rule: Custom JVP rule callable.
            symbolic_zeros: Whether symbolic zeros are passed.

        Returns:
            VariadicCallable[PyTree]: Registered ``jvp_rule``.
        """
        _ = symbolic_zeros
        self.jvp = jvp_rule
        return jvp_rule

    def __call__(self, *call_args: PyTree, **call_kwargs: PyTree) -> PyTree:
        """Invoke the wrapped function.

        Args:
            *call_args: Positional arguments.
            **call_kwargs: Keyword arguments.

        Returns:
            PyTree: Function output.
        """
        if self.fun is not None:
            return self.fun(*call_args, **call_kwargs)
        return None


def custom_jvp_call_jaxpr_p(name: str = "custom_jvp_call_jaxpr") -> MockPrimitive:
    """Return the primitive descriptor for ``custom_jvp_call_jaxpr_p``.

    Args:
        name: Primitive identifier string.

    Returns:
        MockPrimitive: Primitive descriptor.
    """
    return MockPrimitive(name)


def custom_jvp_call_p(name: str = "custom_jvp_call") -> MockPrimitive:
    """Return the primitive descriptor for ``custom_jvp_call_p``.

    Args:
        name: Primitive identifier string.

    Returns:
        MockPrimitive: Primitive descriptor.
    """
    return MockPrimitive(name)


class custom_vjp:
    """Set up a JAX-transformable function for a custom VJP rule definition."""

    def __init__(
        self,
        fun: VariadicCallable[PyTree] | None = None,
        nondiff_argnums: tuple[int, ...] = (),
    ) -> None:
        """Initialize ``custom_vjp``.

        Args:
            fun: Target primal function to wrap.
            nondiff_argnums: Tuple of non-differentiable argument indices.
        """
        self.fun: VariadicCallable[PyTree] | None = fun
        self.nondiff_argnums: tuple[int, ...] = tuple(nondiff_argnums)
        self.fwd: VariadicCallable[PyTree] | None = None
        self.bwd: VariadicCallable[PyTree] | None = None

    def defvjp(
        self,
        fwd: VariadicCallable[PyTree],
        bwd: VariadicCallable[PyTree],
        symbolic_zeros: bool = False,
    ) -> None:
        """Define custom forward and backward VJP rules.

        Args:
            fwd: Forward pass callable.
            bwd: Backward pass callable.
            symbolic_zeros: Whether symbolic zeros are passed.
        """
        _ = symbolic_zeros
        self.fwd = fwd
        self.bwd = bwd

    def __call__(self, *call_args: PyTree, **call_kwargs: PyTree) -> PyTree:
        """Invoke the wrapped function.

        Args:
            *call_args: Positional arguments.
            **call_kwargs: Keyword arguments.

        Returns:
            PyTree: Function output.
        """
        if self.fun is not None:
            return self.fun(*call_args, **call_kwargs)
        return None


def custom_vjp_call_jaxpr_p(name: str = "custom_vjp_call_jaxpr") -> MockPrimitive:
    """Return the primitive descriptor for ``custom_vjp_call_jaxpr_p``.

    Args:
        name: Primitive identifier string.

    Returns:
        MockPrimitive: Primitive descriptor.
    """
    return MockPrimitive(name)


def custom_vjp_call_p(name: str = "custom_vjp_call") -> MockPrimitive:
    """Return the primitive descriptor for ``custom_vjp_call_p``.

    Args:
        name: Primitive identifier string.

    Returns:
        MockPrimitive: Primitive descriptor.
    """
    return MockPrimitive(name)


def custom_vjp_primal_tree_values(tree: PyTree) -> PyTree:
    """Strip perturbation metadata from ``CustomVJPPrimal`` leaves in ``tree``.

    Args:
        tree: PyTree potentially containing :class:`CustomVJPPrimal` objects.

    Returns:
        PyTree: Unwrapped primal values.
    """
    if isinstance(tree, CustomVJPPrimal):
        return tree.value
    return tree


def linear_call(
    fun: VariadicCallable[PyTree],
    fun_transpose: VariadicCallable[PyTree],
    *args: PyTree,
    **kwargs: PyTree,
) -> PyTree:
    """Call a linear function ``fun``, with a custom implementation ``fun_transpose`` for its transpose.

    Args:
        fun: Linear primal callable.
        fun_transpose: Custom transpose callable.
        *args: Positional arguments to ``fun``.
        **kwargs: Keyword arguments to ``fun``.

    Returns:
        PyTree: Result of invoking ``fun(*args, **kwargs)``.
    """
    _ = fun_transpose
    return fun(*args, **kwargs)


__all__ = [
    "CustomVJPPrimal",
    "SymbolicZero",
    "closure_convert",
    "custom_gradient",
    "custom_jvp",
    "custom_jvp_call_jaxpr_p",
    "custom_jvp_call_p",
    "custom_vjp",
    "custom_vjp_call_jaxpr_p",
    "custom_vjp_call_p",
    "custom_vjp_primal_tree_values",
    "linear_call",
]
