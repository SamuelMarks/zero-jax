"""Frontend API routing for jax.interpreters.ad."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class CustomJVPException:
    """Mock implementation for CustomJVPException."""

    pass


class CustomVJPException:
    """Mock implementation for CustomVJPException."""

    pass


class JVPTrace:
    """Mock implementation for JVPTrace."""

    pass


class JVPTracer:
    """Mock implementation for JVPTracer."""

    pass


class UndefinedPrimal:
    """Mock implementation for UndefinedPrimal."""

    pass


class Zero:
    """Mock implementation for Zero."""

    pass


def add_jaxvals(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for add_jaxvals."""
    return getattr(_ops, "add_jaxvals")(*args, **kwargs)


add_jaxvals_p: Any = None


def add_tangents(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for add_tangents."""
    return getattr(_ops, "add_tangents")(*args, **kwargs)


annotations: Any = None


def backward_pass(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for backward_pass."""
    return getattr(_ops, "backward_pass")(*args, **kwargs)


def backward_pass_internal(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for backward_pass_internal."""
    return getattr(_ops, "backward_pass_internal")(*args, **kwargs)


def bilinear_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for bilinear_transpose."""
    return getattr(_ops, "bilinear_transpose")(*args, **kwargs)


call_param_updaters: Any = None


def call_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_transpose."""
    return getattr(_ops, "call_transpose")(*args, **kwargs)


call_transpose_param_updaters: Any = None


def closed_backward_pass(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for closed_backward_pass."""
    return getattr(_ops, "closed_backward_pass")(*args, **kwargs)


custom_lin_p: Any = None


def defbilinear(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defbilinear."""
    return getattr(_ops, "defbilinear")(*args, **kwargs)


def defjvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defjvp."""
    return getattr(_ops, "defjvp")(*args, **kwargs)


def defjvp2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defjvp2."""
    return getattr(_ops, "defjvp2")(*args, **kwargs)


def defjvp_zero(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defjvp_zero."""
    return getattr(_ops, "defjvp_zero")(*args, **kwargs)


def deflinear(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for deflinear."""
    return getattr(_ops, "deflinear")(*args, **kwargs)


def deflinear2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for deflinear2."""
    return getattr(_ops, "deflinear2")(*args, **kwargs)


def f_jvp_traceable(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "f_jvp_traceable")(*args, **kwargs)


def get_primitive_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_primitive_transpose."""
    return getattr(_ops, "get_primitive_transpose")(*args, **kwargs)


def instantiate_zeros(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for instantiate_zeros."""
    return getattr(_ops, "instantiate_zeros")(*args, **kwargs)


def is_undefined_primal(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_undefined_primal."""
    return getattr(_ops, "is_undefined_primal")(*args, **kwargs)


def jvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for jvp."""
    return getattr(_ops, "jvp")(*args, **kwargs)


def jvp_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for jvp_jaxpr."""
    return getattr(_ops, "jvp_jaxpr")(*args, **kwargs)


def jvp_subtrace(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "jvp_subtrace")(*args, **kwargs)


def jvp_subtrace_aux(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "jvp_subtrace_aux")(*args, **kwargs)


def jvpfun(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "jvpfun")(*args, **kwargs)


def linear_jvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for linear_jvp."""
    return getattr(_ops, "linear_jvp")(*args, **kwargs)


def linear_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for linear_transpose."""
    return getattr(_ops, "linear_transpose")(*args, **kwargs)


def linear_transpose2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for linear_transpose2."""
    return getattr(_ops, "linear_transpose2")(*args, **kwargs)


def linearize(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for linearize."""
    return getattr(_ops, "linearize")(*args, **kwargs)


def map_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for map_transpose."""
    return getattr(_ops, "map_transpose")(*args, **kwargs)


def nonzero_outputs(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "nonzero_outputs")(*args, **kwargs)


def nonzero_tangent_outputs(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "nonzero_tangent_outputs")(*args, **kwargs)


primitive_jvps: Any = None

primitive_transposes: Any = None


def rearrange_binders(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for rearrange_binders."""
    return getattr(_ops, "rearrange_binders")(*args, **kwargs)


def recast_to_float0(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for recast_to_float0."""
    return getattr(_ops, "recast_to_float0")(*args, **kwargs)


reducing_transposes: Any = None


def replace_float0s(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for replace_float0s."""
    return getattr(_ops, "replace_float0s")(*args, **kwargs)


def standard_jvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for standard_jvp."""
    return getattr(_ops, "standard_jvp")(*args, **kwargs)


def standard_jvp2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for standard_jvp2."""
    return getattr(_ops, "standard_jvp2")(*args, **kwargs)


def traceable(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "traceable")(*args, **kwargs)


def unpair_pval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for unpair_pval."""
    return getattr(_ops, "unpair_pval")(*args, **kwargs)


def vjp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for vjp."""
    return getattr(_ops, "vjp")(*args, **kwargs)


def zero_jvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for zero_jvp."""
    return getattr(_ops, "zero_jvp")(*args, **kwargs)


def zeros_like_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for zeros_like_aval."""
    return getattr(_ops, "zeros_like_aval")(*args, **kwargs)


def zeros_like_jaxval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for zeros_like_jaxval."""
    return getattr(_ops, "zeros_like_jaxval")(*args, **kwargs)


zeros_like_p: Any = None
