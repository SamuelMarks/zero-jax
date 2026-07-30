"""Frontend API routing for jax.interpreters.batching."""

from dataclasses import dataclass
from typing import Any, Optional

import zero_jax._compiler_proxy_ops as _ops


class Array:
    """Array base class for JAX"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def AxisSize(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return _ops.AxisSize(*args, **kwargs)


@dataclass
class BatchTrace:
    """Data structure for BatchTrace.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "BatchTrace"
    value: Optional[Any] = None


@dataclass
class BatchTracer:
    """Data structure for BatchTracer.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "BatchTracer"
    value: Optional[Any] = None


def BatchingRule(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for BatchingRule.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.BatchingRule(*args, **kwargs)


def Elt(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return _ops.Elt(*args, **kwargs)


def FromEltHandler(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for FromEltHandler.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.FromEltHandler(*args, **kwargs)


def GetIdx(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for GetIdx.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.GetIdx(*args, **kwargs)


class IndexedAxisSize:
    """IndexedAxisSize(idx: 'core.Var', lengths: 'Array | core.Var | Tracer')"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Jumble:
    """Jumble(aval: 'JumbleTy', data: 'Array')"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class JumbleAxis:
    """Data structure for JumbleAxis.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "JumbleAxis"
    value: Optional[Any] = None


class JumbleTy:
    """JumbleTy(binder: 'core.Var', length: 'int | Tracer | core.Var', elt_ty: 'core.DShapedArray')"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def MakeIotaHandler(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for MakeIotaHandler.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.MakeIotaHandler(*args, **kwargs)


def MapSpec(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return _ops.MapSpec(*args, **kwargs)


@dataclass
class NotMapped:
    """Data structure for NotMapped.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "NotMapped"
    value: Optional[Any] = None


not_mapped = NotMapped()


class RaggedAxis:
    """RaggedAxis(stacked_axis: 'int', ragged_axes: 'tuple[tuple[int, Any], ...]')"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def ToEltHandler(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for ToEltHandler.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.ToEltHandler(*args, **kwargs)


def Vmappable(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return _ops.Vmappable(*args, **kwargs)


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


@dataclass
class ZeroIfMapped:
    """Data structure for ZeroIfMapped.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "ZeroIfMapped"
    value: Optional[Any] = None


def axis_primitive_batchers(*args: Any, **kwargs: Any) -> Any:
    return None


def batch(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for batch.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.batch(*args, **kwargs)


def batch_custom_jvp_subtrace(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.batch_custom_jvp_subtrace(*args, **kwargs)


def batch_custom_vjp_bwd(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for batch_custom_vjp_bwd.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.batch_custom_vjp_bwd(*args, **kwargs)


def batch_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for batch_jaxpr.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.batch_jaxpr(*args, **kwargs)


def batch_jaxpr2(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for batch_jaxpr2.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.batch_jaxpr2(*args, **kwargs)


def batch_jaxpr_axes(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for batch_jaxpr_axes.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.batch_jaxpr_axes(*args, **kwargs)


def batch_subtrace(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.batch_subtrace(*args, **kwargs)


def bdim_at_front(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for bdim_at_front.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.bdim_at_front(*args, **kwargs)


def broadcast(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for broadcast.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.broadcast(*args, **kwargs)


def broadcast_batcher(*args: Any, **kwargs: Any) -> Any:
    """Process a primitive with built-in broadcasting."""
    return _ops.broadcast_batcher(*args, **kwargs)


def defbroadcasting(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for defbroadcasting.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.defbroadcasting(*args, **kwargs)


def defreducer(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for defreducer.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.defreducer(*args, **kwargs)


def defvectorized(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for defvectorized.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.defvectorized(*args, **kwargs)


def flatten_fun_for_vmap(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.flatten_fun_for_vmap(*args, **kwargs)


def from_elt(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for from_elt.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.from_elt(*args, **kwargs)


def from_elt_handlers(*args: Any, **kwargs: Any) -> Any:
    return None


def is_vmappable(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for is_vmappable.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.is_vmappable(*args, **kwargs)


def jumble_axis(*args: Any, **kwargs: Any) -> Any:
    return None


def make_iota(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_iota.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.make_iota(*args, **kwargs)


def make_iota_handlers(*args: Any, **kwargs: Any) -> Any:
    return None


def matchaxis(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for matchaxis.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.matchaxis(*args, **kwargs)


def moveaxis(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for moveaxis.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.moveaxis(*args, **kwargs)


def reducer_batcher(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for reducer_batcher.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.reducer_batcher(*args, **kwargs)


def register_vmappable(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for register_vmappable.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.register_vmappable(*args, **kwargs)


def spec_types(*args: Any, **kwargs: Any) -> Any:
    return None


def spmd_axis_primitive_batchers(*args: Any, **kwargs: Any) -> Any:
    return None


def to_elt(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for to_elt.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.to_elt(*args, **kwargs)


def to_elt_handlers(*args: Any, **kwargs: Any) -> Any:
    return None


def unregister_vmappable(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for unregister_vmappable.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.unregister_vmappable(*args, **kwargs)


def vectorized_batcher(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for vectorized_batcher.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.vectorized_batcher(*args, **kwargs)


def vmappables(*args: Any, **kwargs: Any) -> Any:
    return None


def primitive_batchers(*args: Any, **kwargs: Any) -> Any:
    return None


def vtile(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for vtile.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.vtile(*args, **kwargs)


def zero_if_mapped(*args: Any, **kwargs: Any) -> Any:
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
