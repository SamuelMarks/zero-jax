"""Frontend API routing for jax.interpreters.batching."""

from typing import Any


class Array:
    """Array base class for JAX"""

    pass


def AxisSize(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    raise NotImplementedError("AxisSize not yet implemented in zero-jax")


class BatchTrace:
    """Mock implementation for BatchTrace."""

    pass


class BatchTracer:
    """Mock implementation for BatchTracer."""

    pass


def BatchingRule(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for BatchingRule."""
    raise NotImplementedError("BatchingRule not yet implemented in zero-jax")


def Elt(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    raise NotImplementedError("Elt not yet implemented in zero-jax")


def FromEltHandler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for FromEltHandler."""
    raise NotImplementedError("FromEltHandler not yet implemented in zero-jax")


def GetIdx(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for GetIdx."""
    raise NotImplementedError("GetIdx not yet implemented in zero-jax")


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
    raise NotImplementedError("MakeIotaHandler not yet implemented in zero-jax")


def MapSpec(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    raise NotImplementedError("MapSpec not yet implemented in zero-jax")


class NotMapped:
    """Mock implementation for NotMapped."""

    pass


class RaggedAxis:
    """RaggedAxis(stacked_axis: 'int', ragged_axes: 'tuple[tuple[int, Any], ...]')"""

    pass


def ToEltHandler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for ToEltHandler."""
    raise NotImplementedError("ToEltHandler not yet implemented in zero-jax")


def Vmappable(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    raise NotImplementedError("Vmappable not yet implemented in zero-jax")


class Zero:
    """Mock implementation for Zero."""

    pass


class ZeroIfMapped:
    """Mock implementation for ZeroIfMapped."""

    pass


axis_primitive_batchers: Any = None


def batch(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch."""
    raise NotImplementedError("batch not yet implemented in zero-jax")


def batch_custom_jvp_subtrace(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError(
        "batch_custom_jvp_subtrace not yet implemented in zero-jax"
    )


def batch_custom_vjp_bwd(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch_custom_vjp_bwd."""
    raise NotImplementedError("batch_custom_vjp_bwd not yet implemented in zero-jax")


def batch_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch_jaxpr."""
    raise NotImplementedError("batch_jaxpr not yet implemented in zero-jax")


def batch_jaxpr2(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch_jaxpr2."""
    raise NotImplementedError("batch_jaxpr2 not yet implemented in zero-jax")


def batch_jaxpr_axes(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for batch_jaxpr_axes."""
    raise NotImplementedError("batch_jaxpr_axes not yet implemented in zero-jax")


def batch_subtrace(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("batch_subtrace not yet implemented in zero-jax")


def bdim_at_front(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for bdim_at_front."""
    raise NotImplementedError("bdim_at_front not yet implemented in zero-jax")


def broadcast(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for broadcast."""
    raise NotImplementedError("broadcast not yet implemented in zero-jax")


def broadcast_batcher(*args: Any, **kwargs: Any) -> Any:
    """Process a primitive with built-in broadcasting."""
    raise NotImplementedError("broadcast_batcher not yet implemented in zero-jax")


def defbroadcasting(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defbroadcasting."""
    raise NotImplementedError("defbroadcasting not yet implemented in zero-jax")


def defreducer(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defreducer."""
    raise NotImplementedError("defreducer not yet implemented in zero-jax")


def defvectorized(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for defvectorized."""
    raise NotImplementedError("defvectorized not yet implemented in zero-jax")


def flatten_fun_for_vmap(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("flatten_fun_for_vmap not yet implemented in zero-jax")


def from_elt(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for from_elt."""
    raise NotImplementedError("from_elt not yet implemented in zero-jax")


from_elt_handlers: Any = None


def is_vmappable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_vmappable."""
    raise NotImplementedError("is_vmappable not yet implemented in zero-jax")


jumble_axis: Any = None


def make_iota(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_iota."""
    raise NotImplementedError("make_iota not yet implemented in zero-jax")


make_iota_handlers: Any = None


def matchaxis(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for matchaxis."""
    raise NotImplementedError("matchaxis not yet implemented in zero-jax")


def moveaxis(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for moveaxis."""
    raise NotImplementedError("moveaxis not yet implemented in zero-jax")


def reducer_batcher(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for reducer_batcher."""
    raise NotImplementedError("reducer_batcher not yet implemented in zero-jax")


def register_vmappable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_vmappable."""
    raise NotImplementedError("register_vmappable not yet implemented in zero-jax")


spec_types: Any = None

spmd_axis_primitive_batchers: Any = None


def to_elt(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for to_elt."""
    raise NotImplementedError("to_elt not yet implemented in zero-jax")


to_elt_handlers: Any = None


def unregister_vmappable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for unregister_vmappable."""
    raise NotImplementedError("unregister_vmappable not yet implemented in zero-jax")


def vectorized_batcher(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for vectorized_batcher."""
    raise NotImplementedError("vectorized_batcher not yet implemented in zero-jax")


vmappables: Any = None


def vtile(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for vtile."""
    raise NotImplementedError("vtile not yet implemented in zero-jax")


zero_if_mapped: Any = None
