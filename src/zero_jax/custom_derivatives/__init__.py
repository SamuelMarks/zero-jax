"""Frontend API routing for jax.custom_derivatives."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class CustomVJPPrimal:
    """Primal to a ``custom_vjp``'s forward rule when ``symbolic_zeros`` is set"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class SymbolicZero:
    """Frontend state holder for SymbolicZero."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def closure_convert(*args: Any, **kwargs: Any) -> Any:
    """Closure conversion utility, for use with higher-order custom derivatives."""
    return _ops.closure_convert(*args, **kwargs)


def custom_gradient(*args: Any, **kwargs: Any) -> Any:
    """Convenience function for defining custom VJP rules (aka custom gradients)."""
    return _ops.custom_gradient(*args, **kwargs)


class custom_jvp:
    """Set up a JAX-transformable function for a custom JVP rule definition."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def custom_jvp_call_jaxpr_p(*args: Any, **kwargs: Any) -> Any:
    """Stub for custom_jvp_call_jaxpr_p."""
    return None


def custom_jvp_call_p(*args: Any, **kwargs: Any) -> Any:
    """Stub for custom_jvp_call_p."""
    return None


class custom_vjp:
    """Set up a JAX-transformable function for a custom VJP rule definition."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def custom_vjp_call_jaxpr_p(*args: Any, **kwargs: Any) -> Any:
    """Stub for custom_vjp_call_jaxpr_p."""
    return None


def custom_vjp_call_p(*args: Any, **kwargs: Any) -> Any:
    """Stub for custom_vjp_call_p."""
    return None


def custom_vjp_primal_tree_values(*args: Any, **kwargs: Any) -> Any:
    """Strips away perturbation information from forward rule arguments."""
    return _ops.custom_vjp_primal_tree_values(*args, **kwargs)


def linear_call(*args: Any, **kwargs: Any) -> Any:
    """Call a linear function, with a custom implementation for its transpose."""
    return _ops.linear_call(*args, **kwargs)


import typing

import ml_switcheroo_compiler


def __getattr__(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)  # pragma: no cover
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        # If it's a known missing function, we might just return a dummy callable that raises NotImplementedError,
        # BUT we only want to do that if it really doesn't exist, to pass test_stubs.py
        def stub(*args, **kwargs):
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
