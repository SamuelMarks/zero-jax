"""Frontend API routing for jax.interpreters.ad."""

from dataclasses import dataclass
from typing import Any, Optional

import zero_jax._compiler_proxy_ops as _ops


class CustomJVPException(Exception):
    """Exception for CustomJVPException.

    Attributes:
        msg (str): The error message.
    """

    def __init__(self, msg: str = "Default CustomJVPException message") -> None:
        super().__init__(msg)
        self.msg = msg


class CustomVJPException(Exception):
    """Exception for CustomVJPException.

    Attributes:
        msg (str): The error message.
    """

    def __init__(self, msg: str = "Default CustomVJPException message") -> None:
        super().__init__(msg)
        self.msg = msg


@dataclass
class JVPTrace:
    """Data structure for JVPTrace.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "JVPTrace"
    value: Optional[Any] = None


@dataclass
class JVPTracer:
    """Data structure for JVPTracer.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "JVPTracer"
    value: Optional[Any] = None


@dataclass
class UndefinedPrimal:
    """Data structure for UndefinedPrimal.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "UndefinedPrimal"
    value: Optional[Any] = None


@dataclass
class Zero:
    """Data structure for Zero.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "Zero"
    value: Optional[Any] = None


def add_jaxvals(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for add_jaxvals.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.add_jaxvals(*args, **kwargs)


def add_jaxvals_p(*args: Any, **kwargs: Any) -> Any:
    return None


def add_tangents(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for add_tangents.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.add_tangents(*args, **kwargs)


annotations: Any = None


def backward_pass(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for backward_pass.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.backward_pass(*args, **kwargs)


def backward_pass_internal(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for backward_pass_internal.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.backward_pass_internal(*args, **kwargs)


def bilinear_transpose(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for bilinear_transpose.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.bilinear_transpose(*args, **kwargs)


def call_param_updaters(*args: Any, **kwargs: Any) -> Any:
    return None


def call_transpose(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for call_transpose.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.call_transpose(*args, **kwargs)


def call_transpose_param_updaters(*args: Any, **kwargs: Any) -> Any:
    return None


def closed_backward_pass(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for closed_backward_pass.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.closed_backward_pass(*args, **kwargs)


def custom_lin_p(*args: Any, **kwargs: Any) -> Any:
    return None


def defbilinear(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for defbilinear.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.defbilinear(*args, **kwargs)


def defjvp(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for defjvp.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.defjvp(*args, **kwargs)


def defjvp2(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for defjvp2.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.defjvp2(*args, **kwargs)


def defjvp_zero(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for defjvp_zero.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.defjvp_zero(*args, **kwargs)


def deflinear(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for deflinear.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.deflinear(*args, **kwargs)


def deflinear2(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for deflinear2.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.deflinear2(*args, **kwargs)


def f_jvp_traceable(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.f_jvp_traceable(*args, **kwargs)


def get_primitive_transpose(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for get_primitive_transpose.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.get_primitive_transpose(*args, **kwargs)


def instantiate_zeros(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for instantiate_zeros.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.instantiate_zeros(*args, **kwargs)


def is_undefined_primal(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for is_undefined_primal.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.is_undefined_primal(*args, **kwargs)


def jvp(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for jvp.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.jvp(*args, **kwargs)


def jvp_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for jvp_jaxpr.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.jvp_jaxpr(*args, **kwargs)


def jvp_subtrace(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.jvp_subtrace(*args, **kwargs)


def jvp_subtrace_aux(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.jvp_subtrace_aux(*args, **kwargs)


def jvpfun(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.jvpfun(*args, **kwargs)


def linear_jvp(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for linear_jvp.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.linear_jvp(*args, **kwargs)


def linear_transpose(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for linear_transpose.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.linear_transpose(*args, **kwargs)


def linear_transpose2(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for linear_transpose2.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.linear_transpose2(*args, **kwargs)


def linearize(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for linearize.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.linearize(*args, **kwargs)


def map_transpose(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for map_transpose.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.map_transpose(*args, **kwargs)


def nonzero_outputs(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.nonzero_outputs(*args, **kwargs)


def nonzero_tangent_outputs(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.nonzero_tangent_outputs(*args, **kwargs)


def primitive_jvps(*args: Any, **kwargs: Any) -> Any:
    return None


def primitive_transposes(*args: Any, **kwargs: Any) -> Any:
    return None


def rearrange_binders(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for rearrange_binders.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.rearrange_binders(*args, **kwargs)


def recast_to_float0(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for recast_to_float0.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.recast_to_float0(*args, **kwargs)


def reducing_transposes(*args: Any, **kwargs: Any) -> Any:
    return None


def replace_float0s(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for replace_float0s.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.replace_float0s(*args, **kwargs)


def standard_jvp(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for standard_jvp.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.standard_jvp(*args, **kwargs)


def standard_jvp2(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for standard_jvp2.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.standard_jvp2(*args, **kwargs)


def traceable(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.traceable(*args, **kwargs)


def unpair_pval(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for unpair_pval.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.unpair_pval(*args, **kwargs)


def vjp(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for vjp.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.vjp(*args, **kwargs)


def zero_jvp(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for zero_jvp.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.zero_jvp(*args, **kwargs)


def zeros_like_aval(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for zeros_like_aval.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.zeros_like_aval(*args, **kwargs)


def zeros_like_jaxval(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for zeros_like_jaxval.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.zeros_like_jaxval(*args, **kwargs)


def zeros_like_p(*args: Any, **kwargs: Any) -> Any:
    return None


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
            raise NotImplementedError(f"Stub for {name} is not implemented in backend")

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
