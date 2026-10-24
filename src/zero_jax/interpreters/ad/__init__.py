"""Frontend API routing for jax.interpreters.ad."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class CustomJVPException(Exception):
    """Exception for CustomJVPException.

    Attributes:
        msg (str): The error message.
    """

    def __init__(self, msg: str = "Default CustomJVPException message") -> None:
        """Execute ``__init__``.

        Args:
            msg: Input parameter ``msg``.
        """
        super().__init__(msg)
        self.msg = msg


class CustomVJPException(Exception):
    """Exception for CustomVJPException.

    Attributes:
        msg (str): The error message.
    """

    def __init__(self, msg: str = "Default CustomVJPException message") -> None:
        """Execute ``__init__``.

        Args:
            msg: Input parameter ``msg``.
        """
        super().__init__(msg)
        self.msg = msg


@dataclass
class JVPTrace:
    """Data structure for JVPTrace.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "JVPTrace"
    value: ArrayLikeSpec | None = None


@dataclass
class JVPTracer:
    """Data structure for JVPTracer.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "JVPTracer"
    value: ArrayLikeSpec | None = None


@dataclass
class UndefinedPrimal:
    """Data structure for UndefinedPrimal.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "UndefinedPrimal"
    value: ArrayLikeSpec | None = None


@dataclass
class Zero:
    """Data structure for Zero.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "Zero"
    value: ArrayLikeSpec | None = None


def add_jaxvals(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for add_jaxvals.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.add_jaxvals(*args, **kwargs)


def add_jaxvals_p(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``add_jaxvals_p``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def add_tangents(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for add_tangents.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.add_tangents(*args, **kwargs)


annotations: ArrayLikeSpec = None


def backward_pass(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for backward_pass.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.backward_pass(*args, **kwargs)


def backward_pass_internal(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for backward_pass_internal.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.backward_pass_internal(*args, **kwargs)


def bilinear_transpose(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for bilinear_transpose.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.bilinear_transpose(*args, **kwargs)


def call_param_updaters(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``call_param_updaters``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def call_transpose(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for call_transpose.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.call_transpose(*args, **kwargs)


def call_transpose_param_updaters(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``call_transpose_param_updaters``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def closed_backward_pass(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for closed_backward_pass.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.closed_backward_pass(*args, **kwargs)


def custom_lin_p(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``custom_lin_p``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def defbilinear(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for defbilinear.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.defbilinear(*args, **kwargs)


def defjvp(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for defjvp.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.defjvp(*args, **kwargs)


def defjvp2(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for defjvp2.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.defjvp2(*args, **kwargs)


def defjvp_zero(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for defjvp_zero.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.defjvp_zero(*args, **kwargs)


def deflinear(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for deflinear.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.deflinear(*args, **kwargs)


def deflinear2(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for deflinear2.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.deflinear2(*args, **kwargs)


def f_jvp_traceable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.f_jvp_traceable(*args, **kwargs)


def get_primitive_transpose(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for get_primitive_transpose.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.get_primitive_transpose(*args, **kwargs)


def instantiate_zeros(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for instantiate_zeros.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.instantiate_zeros(*args, **kwargs)


def is_undefined_primal(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for is_undefined_primal.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.is_undefined_primal(*args, **kwargs)


def jvp(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for jvp.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.jvp(*args, **kwargs)


def jvp_jaxpr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for jvp_jaxpr.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.jvp_jaxpr(*args, **kwargs)


def jvp_subtrace(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jvp_subtrace(*args, **kwargs)


def jvp_subtrace_aux(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jvp_subtrace_aux(*args, **kwargs)


def jvpfun(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jvpfun(*args, **kwargs)


def linear_jvp(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for linear_jvp.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.linear_jvp(*args, **kwargs)


def linear_transpose(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for linear_transpose.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.linear_transpose(*args, **kwargs)


def linear_transpose2(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for linear_transpose2.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.linear_transpose2(*args, **kwargs)


def linearize(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for linearize.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.linearize(*args, **kwargs)


def map_transpose(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for map_transpose.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.map_transpose(*args, **kwargs)


def nonzero_outputs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.nonzero_outputs(*args, **kwargs)


def nonzero_tangent_outputs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.nonzero_tangent_outputs(*args, **kwargs)


def primitive_jvps(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``primitive_jvps``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def primitive_transposes(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``primitive_transposes``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def rearrange_binders(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for rearrange_binders.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.rearrange_binders(*args, **kwargs)


def recast_to_float0(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for recast_to_float0.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.recast_to_float0(*args, **kwargs)


def reducing_transposes(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``reducing_transposes``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def replace_float0s(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for replace_float0s.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.replace_float0s(*args, **kwargs)


def standard_jvp(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for standard_jvp.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.standard_jvp(*args, **kwargs)


def standard_jvp2(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for standard_jvp2.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.standard_jvp2(*args, **kwargs)


def traceable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.traceable(*args, **kwargs)


def unpair_pval(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for unpair_pval.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.unpair_pval(*args, **kwargs)


def vjp(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for vjp.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.vjp(*args, **kwargs)


def zero_jvp(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for zero_jvp.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.zero_jvp(*args, **kwargs)


def zeros_like_aval(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for zeros_like_aval.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.zeros_like_aval(*args, **kwargs)


def zeros_like_jaxval(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for zeros_like_jaxval.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.zeros_like_jaxval(*args, **kwargs)


def zeros_like_p(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``zeros_like_p``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


class JVP_rules:
    """Stub class for JVP_rules."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize JVP_rules.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


def call_p(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for call_p.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def closed_call_p(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for closed_call_p.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def primitive_ivps(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for primitive_ivps.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None
