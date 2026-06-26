"""Frontend API routing for jax.custom_derivatives."""

from typing import Any


class CustomVJPPrimal:
    """Primal to a ``custom_vjp``'s forward rule when ``symbolic_zeros`` is set"""

    pass


class SymbolicZero:
    """Mock implementation for SymbolicZero."""

    pass


def closure_convert(*args: Any, **kwargs: Any) -> Any:
    """Closure conversion utility, for use with higher-order custom derivatives."""
    raise NotImplementedError("closure_convert not yet implemented in zero-jax")


def custom_gradient(*args: Any, **kwargs: Any) -> Any:
    """Convenience function for defining custom VJP rules (aka custom gradients)."""
    raise NotImplementedError("custom_gradient not yet implemented in zero-jax")


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
    raise NotImplementedError(
        "custom_vjp_primal_tree_values not yet implemented in zero-jax"
    )


def linear_call(*args: Any, **kwargs: Any) -> Any:
    """Call a linear function, with a custom implementation for its transpose."""
    raise NotImplementedError("linear_call not yet implemented in zero-jax")
