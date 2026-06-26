"""Frontend API routing for jax.custom_derivatives."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class CustomVJPPrimal:
    """Primal to a ``custom_vjp``'s forward rule when ``symbolic_zeros`` is set"""

    pass


class SymbolicZero:
    """Mock implementation for SymbolicZero."""

    pass


def closure_convert(*args: Any, **kwargs: Any) -> Any:
    """Closure conversion utility, for use with higher-order custom derivatives."""
    return getattr(_ops, "closure_convert")(*args, **kwargs)


def custom_gradient(*args: Any, **kwargs: Any) -> Any:
    """Convenience function for defining custom VJP rules (aka custom gradients)."""
    return getattr(_ops, "custom_gradient")(*args, **kwargs)


class custom_jvp:
    """Set up a JAX-transformable function for a custom JVP rule definition."""

    pass


custom_jvp_call_jaxpr_p: Any = None

custom_jvp_call_p: Any = None


class custom_vjp:
    """Set up a JAX-transformable function for a custom VJP rule definition."""

    pass


custom_vjp_call_jaxpr_p: Any = None

custom_vjp_call_p: Any = None


def custom_vjp_primal_tree_values(*args: Any, **kwargs: Any) -> Any:
    """Strips away perturbation information from forward rule arguments."""
    return getattr(_ops, "custom_vjp_primal_tree_values")(*args, **kwargs)


def linear_call(*args: Any, **kwargs: Any) -> Any:
    """Call a linear function, with a custom implementation for its transpose."""
    return getattr(_ops, "linear_call")(*args, **kwargs)
