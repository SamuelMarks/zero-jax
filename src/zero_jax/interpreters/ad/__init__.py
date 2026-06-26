"""Frontend API routing for jax.interpreters.ad."""

from typing import Any


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
    raise NotImplementedError("add_jaxvals not yet implemented in zero-jax")


add_jaxvals_p: Any = None


def add_tangents(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for add_tangents."""
    raise NotImplementedError("add_tangents not yet implemented in zero-jax")


annotations: Any = None


def backward_pass(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for backward_pass."""
    raise NotImplementedError("backward_pass not yet implemented in zero-jax")


def backward_pass_internal(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for backward_pass_internal."""
    raise NotImplementedError("backward_pass_internal not yet implemented in zero-jax")


def bilinear_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for bilinear_transpose."""
    raise NotImplementedError("bilinear_transpose not yet implemented in zero-jax")


call_param_updaters: Any = None


def call_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_transpose."""
    raise NotImplementedError("call_transpose not yet implemented in zero-jax")


call_transpose_param_updaters: Any = None


def closed_backward_pass(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for closed_backward_pass."""
    raise NotImplementedError("closed_backward_pass not yet implemented in zero-jax")


custom_lin_p: Any = None


def defbilinear(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defbilinear."""
    raise NotImplementedError("defbilinear not yet implemented in zero-jax")


def defjvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defjvp."""
    raise NotImplementedError("defjvp not yet implemented in zero-jax")


def defjvp2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defjvp2."""
    raise NotImplementedError("defjvp2 not yet implemented in zero-jax")


def defjvp_zero(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defjvp_zero."""
    raise NotImplementedError("defjvp_zero not yet implemented in zero-jax")


def deflinear(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for deflinear."""
    raise NotImplementedError("deflinear not yet implemented in zero-jax")


def deflinear2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for deflinear2."""
    raise NotImplementedError("deflinear2 not yet implemented in zero-jax")


def f_jvp_traceable(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("f_jvp_traceable not yet implemented in zero-jax")


def get_primitive_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_primitive_transpose."""
    raise NotImplementedError("get_primitive_transpose not yet implemented in zero-jax")


def instantiate_zeros(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for instantiate_zeros."""
    raise NotImplementedError("instantiate_zeros not yet implemented in zero-jax")


def is_undefined_primal(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_undefined_primal."""
    raise NotImplementedError("is_undefined_primal not yet implemented in zero-jax")


def jvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for jvp."""
    raise NotImplementedError("jvp not yet implemented in zero-jax")


def jvp_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for jvp_jaxpr."""
    raise NotImplementedError("jvp_jaxpr not yet implemented in zero-jax")


def jvp_subtrace(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("jvp_subtrace not yet implemented in zero-jax")


def jvp_subtrace_aux(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("jvp_subtrace_aux not yet implemented in zero-jax")


def jvpfun(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("jvpfun not yet implemented in zero-jax")


def linear_jvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for linear_jvp."""
    raise NotImplementedError("linear_jvp not yet implemented in zero-jax")


def linear_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for linear_transpose."""
    raise NotImplementedError("linear_transpose not yet implemented in zero-jax")


def linear_transpose2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for linear_transpose2."""
    raise NotImplementedError("linear_transpose2 not yet implemented in zero-jax")


def linearize(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for linearize."""
    raise NotImplementedError("linearize not yet implemented in zero-jax")


def map_transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for map_transpose."""
    raise NotImplementedError("map_transpose not yet implemented in zero-jax")


def nonzero_outputs(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("nonzero_outputs not yet implemented in zero-jax")


def nonzero_tangent_outputs(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("nonzero_tangent_outputs not yet implemented in zero-jax")


primitive_jvps: Any = None

primitive_transposes: Any = None


def rearrange_binders(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for rearrange_binders."""
    raise NotImplementedError("rearrange_binders not yet implemented in zero-jax")


def recast_to_float0(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for recast_to_float0."""
    raise NotImplementedError("recast_to_float0 not yet implemented in zero-jax")


reducing_transposes: Any = None


def replace_float0s(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for replace_float0s."""
    raise NotImplementedError("replace_float0s not yet implemented in zero-jax")


def standard_jvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for standard_jvp."""
    raise NotImplementedError("standard_jvp not yet implemented in zero-jax")


def standard_jvp2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for standard_jvp2."""
    raise NotImplementedError("standard_jvp2 not yet implemented in zero-jax")


def traceable(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("traceable not yet implemented in zero-jax")


def unpair_pval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for unpair_pval."""
    raise NotImplementedError("unpair_pval not yet implemented in zero-jax")


def vjp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for vjp."""
    raise NotImplementedError("vjp not yet implemented in zero-jax")


def zero_jvp(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for zero_jvp."""
    raise NotImplementedError("zero_jvp not yet implemented in zero-jax")


def zeros_like_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for zeros_like_aval."""
    raise NotImplementedError("zeros_like_aval not yet implemented in zero-jax")


def zeros_like_jaxval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for zeros_like_jaxval."""
    raise NotImplementedError("zeros_like_jaxval not yet implemented in zero-jax")


zeros_like_p: Any = None
