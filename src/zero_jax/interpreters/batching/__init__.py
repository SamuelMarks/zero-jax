"""Frontend API routing for jax.interpreters.batching."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class Array:
    """Array base class for JAX"""

    pass


def AxisSize(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return getattr(_ops, "AxisSize")(*args, **kwargs)


class BatchTrace:
    """Mock implementation for BatchTrace."""

    pass


class BatchTracer:
    """Mock implementation for BatchTracer."""

    pass


def BatchingRule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for BatchingRule."""
    return getattr(_ops, "BatchingRule")(*args, **kwargs)


def Elt(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return getattr(_ops, "Elt")(*args, **kwargs)


def FromEltHandler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for FromEltHandler."""
    return getattr(_ops, "FromEltHandler")(*args, **kwargs)


def GetIdx(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for GetIdx."""
    return getattr(_ops, "GetIdx")(*args, **kwargs)


class IndexedAxisSize:
    """IndexedAxisSize(idx: 'core.Var', lengths: 'Array | core.Var | Tracer')"""

    pass


class Jumble:
    """Jumble(aval: 'JumbleTy', data: 'Array')"""

    pass


class JumbleAxis:
    """Mock implementation for JumbleAxis."""

    pass


class JumbleTy:
    """JumbleTy(binder: 'core.Var', length: 'int | Tracer | core.Var', elt_ty: 'core.DShapedArray')"""

    pass


def MakeIotaHandler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for MakeIotaHandler."""
    return getattr(_ops, "MakeIotaHandler")(*args, **kwargs)


def MapSpec(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return getattr(_ops, "MapSpec")(*args, **kwargs)


class NotMapped:
    """Mock implementation for NotMapped."""

    pass


class RaggedAxis:
    """RaggedAxis(stacked_axis: 'int', ragged_axes: 'tuple[tuple[int, Any], ...]')"""

    pass


def ToEltHandler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ToEltHandler."""
    return getattr(_ops, "ToEltHandler")(*args, **kwargs)


def Vmappable(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return getattr(_ops, "Vmappable")(*args, **kwargs)


class Zero:
    """Mock implementation for Zero."""

    pass


class ZeroIfMapped:
    """Mock implementation for ZeroIfMapped."""

    pass


axis_primitive_batchers: Any = None


def batch(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch."""
    return getattr(_ops, "batch")(*args, **kwargs)


def batch_custom_jvp_subtrace(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "batch_custom_jvp_subtrace")(*args, **kwargs)


def batch_custom_vjp_bwd(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch_custom_vjp_bwd."""
    return getattr(_ops, "batch_custom_vjp_bwd")(*args, **kwargs)


def batch_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch_jaxpr."""
    return getattr(_ops, "batch_jaxpr")(*args, **kwargs)


def batch_jaxpr2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch_jaxpr2."""
    return getattr(_ops, "batch_jaxpr2")(*args, **kwargs)


def batch_jaxpr_axes(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch_jaxpr_axes."""
    return getattr(_ops, "batch_jaxpr_axes")(*args, **kwargs)


def batch_subtrace(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "batch_subtrace")(*args, **kwargs)


def bdim_at_front(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for bdim_at_front."""
    return getattr(_ops, "bdim_at_front")(*args, **kwargs)


def broadcast(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for broadcast."""
    return getattr(_ops, "broadcast")(*args, **kwargs)


def broadcast_batcher(*args: Any, **kwargs: Any) -> Any:
    """Process a primitive with built-in broadcasting."""
    return getattr(_ops, "broadcast_batcher")(*args, **kwargs)


def defbroadcasting(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defbroadcasting."""
    return getattr(_ops, "defbroadcasting")(*args, **kwargs)


def defreducer(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defreducer."""
    return getattr(_ops, "defreducer")(*args, **kwargs)


def defvectorized(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defvectorized."""
    return getattr(_ops, "defvectorized")(*args, **kwargs)


def flatten_fun_for_vmap(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "flatten_fun_for_vmap")(*args, **kwargs)


def from_elt(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for from_elt."""
    return getattr(_ops, "from_elt")(*args, **kwargs)


from_elt_handlers: Any = None


def is_vmappable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_vmappable."""
    return getattr(_ops, "is_vmappable")(*args, **kwargs)


jumble_axis: Any = None


def make_iota(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_iota."""
    return getattr(_ops, "make_iota")(*args, **kwargs)


make_iota_handlers: Any = None


def matchaxis(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for matchaxis."""
    return getattr(_ops, "matchaxis")(*args, **kwargs)


def moveaxis(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for moveaxis."""
    return getattr(_ops, "moveaxis")(*args, **kwargs)


def reducer_batcher(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for reducer_batcher."""
    return getattr(_ops, "reducer_batcher")(*args, **kwargs)


def register_vmappable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_vmappable."""
    return getattr(_ops, "register_vmappable")(*args, **kwargs)


spec_types: Any = None

spmd_axis_primitive_batchers: Any = None


def to_elt(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for to_elt."""
    return getattr(_ops, "to_elt")(*args, **kwargs)


to_elt_handlers: Any = None


def unregister_vmappable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for unregister_vmappable."""
    return getattr(_ops, "unregister_vmappable")(*args, **kwargs)


def vectorized_batcher(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for vectorized_batcher."""
    return getattr(_ops, "vectorized_batcher")(*args, **kwargs)


vmappables: Any = None


def vtile(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for vtile."""
    return getattr(_ops, "vtile")(*args, **kwargs)


zero_if_mapped: Any = None
