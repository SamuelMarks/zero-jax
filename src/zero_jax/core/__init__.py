"""Frontend API routing for jax.core."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


@dataclass
class AbstractToken:
    """Represents a token type in the abstract interpretation."""

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
class AbstractValue:
    """Base class for all abstract values in JAX."""

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


def Atom(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns an atom variable.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.Atom(*args, **kwargs)


def AxisSize(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Represents the size of a mapped axis.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.AxisSize(*args, **kwargs)


@dataclass
class CallPrimitive:
    """Primitive that calls a JAX computation."""

    name: str = ""


@dataclass
class ClosedJaxpr:
    """A Jaxpr with its environment closed over."""

    jaxpr: ArrayLikeSpec = None
    consts: list[ArrayLikeSpec] = field(default_factory=list)


@dataclass
class ConcreteArray:
    """An array with a known concrete value."""

    val: ArrayLikeSpec = None


class ConcretizationTypeError(Exception):
    """This error occurs when a JAX Tracer object is used in a context where a concrete value is needed."""

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
class DShapedArray:
    """A shaped array with dynamic shape dimensions."""

    shape: tuple[ArrayLikeSpec, ...] = field(default_factory=tuple)
    dtype: ArrayLikeSpec = None


@dataclass
class DropVar:
    """Represents a dropped variable that is not used."""

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
class Effect:
    """A generic side-effect."""

    name: str = ""


@dataclass
class Effects:
    """A set is a finite, iterable container representing multiple side-effects."""

    effects: set[Effect] = field(default_factory=set)


@dataclass
class EvalTrace:
    """A trace representing evaluation."""

    state: ArrayLikeSpec = None


@dataclass
class InDBIdx:
    """Input De Bruijn Index."""

    val: int = 0


class InconclusiveDimensionOperation(Exception):
    """Raised when we cannot conclusively compute with symbolic dimensions."""

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
class InputType:
    """Built-in immutable sequence representing input types."""

    types: tuple[ArrayLikeSpec, ...] = field(default_factory=tuple)


@dataclass
class JaxprDebugInfo:
    """Information for debugging a Jaxpr."""

    traced_for: ArrayLikeSpec = None
    func_src_info: str = ""
    arg_names: tuple[str, ...] = field(default_factory=tuple)
    result_paths: tuple[ArrayLikeSpec, ...] = field(default_factory=tuple)


@dataclass
class JaxprEqn:
    """Equation in a Jaxpr."""

    invars: list[ArrayLikeSpec] = field(default_factory=list)
    outvars: list[ArrayLikeSpec] = field(default_factory=list)
    primitive: ArrayLikeSpec = None
    params: dict[str, ArrayLikeSpec] = field(default_factory=dict)
    effects: ArrayLikeSpec = None
    source_info: ArrayLikeSpec = None
    ctx: ArrayLikeSpec = None


@dataclass
class JaxprPpContext:
    """Context for Jaxpr pretty printing."""

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
class JaxprPpSettings:
    """Settings for Jaxpr pretty printing."""

    print_shapes: bool = True
    source_info: bool = False
    name_stack: bool = False
    custom_pp_eqn_rules: bool = False
    print_effects: bool = True


class JaxprTypeError(Exception):
    """Error raised when a Jaxpr is ill-typed."""

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
class Literal:
    """A literal value in a Jaxpr."""

    val: ArrayLikeSpec = None


@dataclass
class MainTrace:
    """The main trace context for JAX operations."""

    level: int = 0
    trace_type: ArrayLikeSpec = None


@dataclass
class MapPrimitive:
    """A primitive that maps over an axis."""

    name: str = ""


@dataclass
class NameGatheringSubst:
    """Substitution map gathering names."""

    names: dict[str, ArrayLikeSpec] = field(default_factory=dict)


@dataclass
class NamedShape:
    """A shape with named dimensions."""

    shape: tuple[ArrayLikeSpec, ...] = field(default_factory=tuple)


@dataclass
class OutDBIdx:
    """Output De Bruijn Index."""

    val: int = 0


@dataclass
class OutputType:
    """Built-in immutable sequence representing output types."""

    types: tuple[ArrayLikeSpec, ...] = field(default_factory=tuple)


@dataclass
class ParamDict:
    """Dictionary for primitive parameters."""

    params: dict[str, ArrayLikeSpec] = field(default_factory=dict)


@dataclass
class Sublevel:
    """A sub-level for trace evaluations."""

    level: int = 0


class TRACER_LEAK_DEBUGGER_WARNING:
    """Warning for tracer leaks."""


@dataclass
class ThreadLocalState:
    """Thread local state for JAX tracing context."""

    trace_state: ArrayLikeSpec = None


@dataclass
class TraceStack:
    """Stack of trace states."""

    stack: list[ArrayLikeSpec] = field(default_factory=list)


@dataclass
class TraceState:
    """State for JAX tracing."""

    trace_stack: ArrayLikeSpec = None


@dataclass
class Tracer:
    """Base class for Tracers."""

    trace: ArrayLikeSpec = None


@dataclass
class UnshapedArray:
    """An array with no known shape."""

    dtype: ArrayLikeSpec = None


def abstract_token(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for abstract_token.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def apply_todos(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Apply todos in the trace state.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.apply_todos(*args, **kwargs)


def as_named_shape(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert a sequence to a NamedShape.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.as_named_shape(*args, **kwargs)


def aval_mapping_handlers(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for aval_mapping_handlers.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def axis_frame(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Axis frame for tracing context.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.axis_frame(*args, **kwargs)


def call(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Call primitive.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.call(*args, **kwargs)


def call_bind_with_continuation(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Call bind with continuation for primitives.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.call_bind_with_continuation(*args, **kwargs)


def call_impl(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Implementation for call primitive.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.call_impl(*args, **kwargs)


def call_p(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for call_p.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def check_eqn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Check equation in a Jaxpr for validity.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.check_eqn(*args, **kwargs)


def check_jaxpr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Checks well-formedness of a jaxpr.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.check_jaxpr(*args, **kwargs)


def check_type(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Check the type of a variable.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.check_type(*args, **kwargs)


def check_valid_jaxtype(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Check if a given type is a valid JAX type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.check_valid_jaxtype(*args, **kwargs)


def closed_call_p(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for call_p.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def concrete_aval(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Get the concrete abstract value of an array.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.concrete_aval(*args, **kwargs)


def concrete_or_error(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Like force(val), but gives the context in the error message.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.concrete_or_error(*args, **kwargs)


def concretization_function_error(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Raise a concretization error with a detailed message.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.concretization_function_error(*args, **kwargs)


def cur_sublevel(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Get the current sublevel in the trace stack.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.cur_sublevel(*args, **kwargs)


def custom_typechecks(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for custom_typechecks.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def dedup_referents(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Deduplicate object referents in a trace.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.dedup_referents(*args, **kwargs)


def do_subst_axis_names_jaxpr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Substitute axis names in a Jaxpr.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.do_subst_axis_names_jaxpr(*args, **kwargs)


def ensure_compile_time_eval(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager to ensure evaluation at trace/compile time (or error).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.ensure_compile_time_eval(*args, **kwargs)


def escaped_tracer_error(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Error indicating a Tracer has escaped its dynamic scope.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.escaped_tracer_error(*args, **kwargs)


def eval_context(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Context manager to ensure evaluation at trace/compile time (or error).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.eval_context(*args, **kwargs)


def eval_jaxpr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Evaluate a Jaxpr.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.eval_jaxpr(*args, **kwargs)


def extend_axis_env(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Extend the current axis environment context.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.extend_axis_env(*args, **kwargs)


def extend_axis_env_nd(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Extend the current ND axis environment context.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.extend_axis_env_nd(*args, **kwargs)


def find_top_trace(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Find the top-level trace in the current context.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.find_top_trace(*args, **kwargs)


def full_lower(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Perform a full lowering of a JAX computation.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.full_lower(*args, **kwargs)


def gensym(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Produce distinct variables, printed with the optional suffix.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.gensym(*args, **kwargs)


def get_aval(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Get the abstract value of an object.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.get_aval(*args, **kwargs)


def get_referent(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Get the referent of a Tracer object.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.get_referent(*args, **kwargs)


def is_constant_dim(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Check if a dimension is constant.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.is_constant_dim(*args, **kwargs)


def is_constant_shape(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Check if all dimensions of a shape are constant.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.is_constant_shape(*args, **kwargs)


def jaxpr_as_fun(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert a Jaxpr to a callable function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jaxpr_as_fun(*args, **kwargs)


def jaxpr_uses_outfeed(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Finds if there are outfeed primitives anywhere inside a Jaxpr.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jaxpr_uses_outfeed(*args, **kwargs)


def jaxprs_in_params(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Extract all Jaxprs found inside primitive parameters.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.jaxprs_in_params(*args, **kwargs)


def join_effects(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Join multiple side-effects sets.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.join_effects(*args, **kwargs)


def join_named_shapes(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Join multiple NamedShapes.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.join_named_shapes(*args, **kwargs)


def lattice_join(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Lattice join two abstract values.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.lattice_join(*args, **kwargs)


def leaked_tracer_error(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Error indicating a Tracer has leaked out of its transform scope.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.leaked_tracer_error(*args, **kwargs)


def literalable_types(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for literalable_types.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def map_bind(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bind a MapPrimitive to arguments.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.map_bind(*args, **kwargs)


def map_bind_with_continuation(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bind a MapPrimitive with a continuation.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.map_bind_with_continuation(*args, **kwargs)


def mapped_aval(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Get the abstract value mapped across an axis.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.mapped_aval(*args, **kwargs)


def max_dim(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Like max(d1, d2) but for both constant and symbolic dimensions.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.max_dim(*args, **kwargs)


def maybe_find_leaked_tracers(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Find the leaked tracers holding a reference to the MainTrace or SubLevel.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.maybe_find_leaked_tracers(*args, **kwargs)


def min_dim(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Like min(d1, d2) but for both constant and symbolic dimensions.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.min_dim(*args, **kwargs)


def new_base_main(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Create a new base MainTrace.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.new_base_main(*args, **kwargs)


def new_jaxpr_eqn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Create a new Jaxpr equation.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.new_jaxpr_eqn(*args, **kwargs)


def new_main(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Create a new MainTrace context.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.new_main(*args, **kwargs)


def new_sublevel(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Create a new Sublevel for tracing.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.new_sublevel(*args, **kwargs)


def no_axis_name(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for no_axis_name.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def no_effects(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for no_effects.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def outfeed_primitives(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for outfeed_primitives.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def primal_dtype_to_tangent_dtype(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convert a primal dtype to its corresponding tangent dtype.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.primal_dtype_to_tangent_dtype(*args, **kwargs)


def primitive_uses_outfeed(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Determine if a primitive uses the outfeed.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.primitive_uses_outfeed(*args, **kwargs)


def process_env_traces_call(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.process_env_traces_call(*args, **kwargs)


def process_env_traces_map(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """partial(func, *args, **keywords) - new function with partial application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.process_env_traces_map(*args, **kwargs)


def pytype_aval_mappings(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for pytype_aval_mappings.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def raise_as_much_as_possible(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Raise as many shape exceptions as possible for missing dimensions.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.raise_as_much_as_possible(*args, **kwargs)


def raise_to_shaped(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Raise an abstract value to a ShapedArray.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.raise_to_shaped(*args, **kwargs)


def raise_to_shaped_mappings(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for raise_to_shaped_mappings.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def reset_trace_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Resets the global trace state and returns True if it was already clean.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.reset_trace_state(*args, **kwargs)


def stash_axis_env(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Promise that a function or with-suite does not depend implicitly on axis env.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.stash_axis_env(*args, **kwargs)


def str_eqn_compact(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compact equation to string conversion used in HLO metadata.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.str_eqn_compact(*args, **kwargs)


def subjaxprs(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Generator for all subjaxprs found in the params of jaxpr.eqns.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.subjaxprs(*args, **kwargs)


def subst_axis_names(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Substitute axis names in a computation context.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.subst_axis_names(*args, **kwargs)


def subst_axis_names_eqn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Substitute axis names in a Jaxpr equation.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.subst_axis_names_eqn(*args, **kwargs)


def subst_axis_names_jaxpr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Substitute axis names within a Jaxpr.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.subst_axis_names_jaxpr(*args, **kwargs)


def subst_axis_names_var(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Substitute axis names in a variable.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.subst_axis_names_var(*args, **kwargs)


def substitute_vars_in_output_ty(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Substitute variables in the output type signature.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.substitute_vars_in_output_ty(*args, **kwargs)


def thread_local_state(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stub for thread_local_state.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def trace_state_clean(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Check if the global trace state is clean.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.trace_state_clean(*args, **kwargs)


def traverse_jaxpr_params(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Applies f to each jaxpr parameter and returns a tuple of returned values.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.traverse_jaxpr_params(*args, **kwargs)


def typecheck(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Typecheck a primitive application.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.typecheck(*args, **kwargs)


def typecompat(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Determine whether `aval` conforms to `aval_ref`.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.typecompat(*args, **kwargs)


def typematch(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Determine whether `aval1` and `aval2` are equivalent.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.typematch(*args, **kwargs)


def unmapped_aval(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Get the abstract value unmapped from an axis.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.unmapped_aval(*args, **kwargs)


def used_axis_names(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Find all used axis names in a context.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.used_axis_names(*args, **kwargs)


def used_axis_names_jaxpr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Find all used axis names inside a Jaxpr.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.used_axis_names_jaxpr(*args, **kwargs)


def valid_jaxtype(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Check if a type is a valid JAX type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.valid_jaxtype(*args, **kwargs)


@dataclass
class Jaxpr:
    """A Jaxpr representation."""

    constvars: list[ArrayLikeSpec] = field(default_factory=list)
    invars: list[ArrayLikeSpec] = field(default_factory=list)
    outvars: list[ArrayLikeSpec] = field(default_factory=list)
    eqns: list[JaxprEqn] = field(default_factory=list)


@dataclass
class Primitive:
    """A JAX primitive operation."""

    name: str = ""


@dataclass
class ShapedArray:
    """A shaped array abstract value."""

    shape: tuple[ArrayLikeSpec, ...] = field(default_factory=tuple)
    dtype: ArrayLikeSpec = None


@dataclass
class Token:
    """A token used for ordering effects."""

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
class Trace:
    """Base trace class."""

    main: ArrayLikeSpec = None


@dataclass
class Var:
    """A variable in a Jaxpr."""

    aval: ArrayLikeSpec = None


Value = ArrayLikeSpec
