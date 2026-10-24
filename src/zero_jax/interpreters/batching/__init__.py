"""Frontend API routing for jax.interpreters.batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class Array:
    """Array base class for JAX."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


def AxisSize(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Special type indicating an unconstrained type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.AxisSize(*args, **kwargs)


@dataclass
class BatchTrace:
    """Data structure for BatchTrace.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "BatchTrace"
    value: ArrayLikeSpec | None = None


@dataclass
class BatchTracer:
    """Data structure for BatchTracer.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "BatchTracer"
    value: ArrayLikeSpec | None = None


def BatchingRule(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for BatchingRule.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.BatchingRule(*args, **kwargs)


def Elt(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Special type indicating an unconstrained type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.Elt(*args, **kwargs)


def FromEltHandler(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for FromEltHandler.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.FromEltHandler(*args, **kwargs)


def GetIdx(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for GetIdx.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.GetIdx(*args, **kwargs)


class IndexedAxisSize:
    """IndexedAxisSize(idx: 'core.Var', lengths: 'Array | core.Var | Tracer')."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


class Jumble:
    """Jumble(aval: 'JumbleTy', data: 'Array')."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


@dataclass
class JumbleAxis:
    """Data structure for JumbleAxis.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "JumbleAxis"
    value: ArrayLikeSpec | None = None


class JumbleTy:
    """JumbleTy(binder: 'core.Var', length: 'int | Tracer | core.Var', elt_ty: 'core.DShapedArray')."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


def MakeIotaHandler(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for MakeIotaHandler.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.MakeIotaHandler(*args, **kwargs)


def MapSpec(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Special type indicating an unconstrained type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.MapSpec(*args, **kwargs)


@dataclass
class NotMapped:
    """Data structure for NotMapped.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "NotMapped"
    value: ArrayLikeSpec | None = None


not_mapped = NotMapped()


class RaggedAxis:
    """RaggedAxis(stacked_axis: 'int', ragged_axes: 'tuple[tuple[int, ArrayLikeSpec], ...]')."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


def ToEltHandler(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for ToEltHandler.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.ToEltHandler(*args, **kwargs)


def Vmappable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Special type indicating an unconstrained type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.Vmappable(*args, **kwargs)


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


@dataclass
class ZeroIfMapped:
    """Data structure for ZeroIfMapped.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "ZeroIfMapped"
    value: ArrayLikeSpec | None = None


def axis_primitive_batchers(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``axis_primitive_batchers``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def batch(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for batch.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.batch(*args, **kwargs)


def batch_custom_jvp_subtrace(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.batch_custom_jvp_subtrace(*args, **kwargs)


def batch_custom_vjp_bwd(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for batch_custom_vjp_bwd.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.batch_custom_vjp_bwd(*args, **kwargs)


def batch_jaxpr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for batch_jaxpr.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.batch_jaxpr(*args, **kwargs)


def batch_jaxpr2(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for batch_jaxpr2.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.batch_jaxpr2(*args, **kwargs)


def batch_jaxpr_axes(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for batch_jaxpr_axes.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.batch_jaxpr_axes(*args, **kwargs)


def batch_subtrace(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.batch_subtrace(*args, **kwargs)


def bdim_at_front(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for bdim_at_front.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.bdim_at_front(*args, **kwargs)


def broadcast(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for broadcast.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.broadcast(*args, **kwargs)


def broadcast_batcher(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Process a primitive with built-in broadcasting.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.broadcast_batcher(*args, **kwargs)


def defbroadcasting(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for defbroadcasting.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.defbroadcasting(*args, **kwargs)


def defreducer(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for defreducer.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.defreducer(*args, **kwargs)


def defvectorized(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for defvectorized.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.defvectorized(*args, **kwargs)


def flatten_fun_for_vmap(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.flatten_fun_for_vmap(*args, **kwargs)


def from_elt(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for from_elt.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.from_elt(*args, **kwargs)


def from_elt_handlers(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``from_elt_handlers``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def is_vmappable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for is_vmappable.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.is_vmappable(*args, **kwargs)


def jumble_axis(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``jumble_axis``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def make_iota(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for make_iota.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.make_iota(*args, **kwargs)


def make_iota_handlers(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``make_iota_handlers``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def matchaxis(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for matchaxis.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.matchaxis(*args, **kwargs)


def moveaxis(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for moveaxis.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.moveaxis(*args, **kwargs)


def reducer_batcher(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for reducer_batcher.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.reducer_batcher(*args, **kwargs)


def register_vmappable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for register_vmappable.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.register_vmappable(*args, **kwargs)


def spec_types(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``spec_types``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def spmd_axis_primitive_batchers(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``spmd_axis_primitive_batchers``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def to_elt(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for to_elt.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.to_elt(*args, **kwargs)


def to_elt_handlers(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``to_elt_handlers``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def unregister_vmappable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for unregister_vmappable.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.unregister_vmappable(*args, **kwargs)


def vectorized_batcher(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for vectorized_batcher.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.vectorized_batcher(*args, **kwargs)


def vmappables(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``vmappables``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def primitive_batchers(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``primitive_batchers``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def vtile(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for vtile.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.vtile(*args, **kwargs)


def zero_if_mapped(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``zero_if_mapped``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


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


def unbatcher_spec_types(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for unbatcher_spec_types.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None
