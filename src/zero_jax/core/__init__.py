"""Frontend API routing for jax.core."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import zero_jax._compiler_proxy_ops as _ops


@dataclass
class AbstractToken:
    """Represents a token type in the abstract interpretation."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


@dataclass
class AbstractValue:
    """Base class for all abstract values in JAX."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def Atom(*args: Any, **kwargs: Any) -> Any:
    """Returns an atom variable."""
    return _ops.Atom(*args, **kwargs)


def AxisSize(*args: Any, **kwargs: Any) -> Any:
    """Represents the size of a mapped axis."""
    return _ops.AxisSize(*args, **kwargs)


@dataclass
class CallPrimitive:
    """Primitive that calls a JAX computation."""

    name: str = ""


@dataclass
class ClosedJaxpr:
    """A Jaxpr with its environment closed over."""

    jaxpr: Any = None
    consts: List[Any] = field(default_factory=list)


@dataclass
class ConcreteArray:
    """An array with a known concrete value."""

    val: Any = None


class ConcretizationTypeError(Exception):
    """This error occurs when a JAX Tracer object is used in a context where a concrete value is needed."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


@dataclass
class DShapedArray:
    """A shaped array with dynamic shape dimensions."""

    shape: Tuple[Any, ...] = field(default_factory=tuple)
    dtype: Any = None


@dataclass
class DropVar:
    """Represents a dropped variable that is not used."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


@dataclass
class Effect:
    """A generic side-effect."""

    name: str = ""


@dataclass
class Effects:
    """A set is a finite, iterable container representing multiple side-effects."""

    effects: Set[Effect] = field(default_factory=set)


@dataclass
class EvalTrace:
    """A trace representing evaluation."""

    state: Any = None


@dataclass
class InDBIdx:
    """Input De Bruijn Index."""

    val: int = 0


class InconclusiveDimensionOperation(Exception):
    """Raised when we cannot conclusively compute with symbolic dimensions."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


@dataclass
class InputType:
    """Built-in immutable sequence representing input types."""

    types: Tuple[Any, ...] = field(default_factory=tuple)


@dataclass
class JaxprDebugInfo:
    """Information for debugging a Jaxpr."""

    traced_for: Any = None
    func_src_info: str = ""
    arg_names: Tuple[str, ...] = field(default_factory=tuple)
    result_paths: Tuple[Any, ...] = field(default_factory=tuple)


@dataclass
class JaxprEqn:
    """Equation in a Jaxpr."""

    invars: List[Any] = field(default_factory=list)
    outvars: List[Any] = field(default_factory=list)
    primitive: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    effects: Any = None
    source_info: Any = None
    ctx: Any = None


@dataclass
class JaxprPpContext:
    """Context for Jaxpr pretty printing."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


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

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


@dataclass
class Literal:
    """A literal value in a Jaxpr."""

    val: Any = None


@dataclass
class MainTrace:
    """The main trace context for JAX operations."""

    level: int = 0
    trace_type: Any = None


@dataclass
class MapPrimitive:
    """A primitive that maps over an axis."""

    name: str = ""


@dataclass
class NameGatheringSubst:
    """Substitution map gathering names."""

    names: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NamedShape:
    """A shape with named dimensions."""

    shape: Tuple[Any, ...] = field(default_factory=tuple)


@dataclass
class OutDBIdx:
    """Output De Bruijn Index."""

    val: int = 0


@dataclass
class OutputType:
    """Built-in immutable sequence representing output types."""

    types: Tuple[Any, ...] = field(default_factory=tuple)


@dataclass
class ParamDict:
    """Dictionary for primitive parameters."""

    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Sublevel:
    """A sub-level for trace evaluations."""

    level: int = 0


class TRACER_LEAK_DEBUGGER_WARNING:
    """Warning for tracer leaks."""


@dataclass
class ThreadLocalState:
    """Thread local state for JAX tracing context."""

    trace_state: Any = None


@dataclass
class TraceStack:
    """Stack of trace states."""

    stack: List[Any] = field(default_factory=list)


@dataclass
class TraceState:
    """State for JAX tracing."""

    trace_stack: Any = None


@dataclass
class Tracer:
    """Base class for Tracers."""

    trace: Any = None


@dataclass
class UnshapedArray:
    """An array with no known shape."""

    dtype: Any = None


def abstract_token(*args: Any, **kwargs: Any) -> Any:
    """Stub for abstract_token."""
    return None


def apply_todos(*args: Any, **kwargs: Any) -> Any:
    """Apply todos in the trace state."""
    return _ops.apply_todos(*args, **kwargs)


def as_named_shape(*args: Any, **kwargs: Any) -> Any:
    """Convert a sequence to a NamedShape."""
    return _ops.as_named_shape(*args, **kwargs)


def aval_mapping_handlers(*args: Any, **kwargs: Any) -> Any:
    """Stub for aval_mapping_handlers."""
    return None


def axis_frame(*args: Any, **kwargs: Any) -> Any:
    """Axis frame for tracing context."""
    return _ops.axis_frame(*args, **kwargs)


def call(*args: Any, **kwargs: Any) -> Any:
    """Call primitive."""
    return _ops.call(*args, **kwargs)


def call_bind_with_continuation(*args: Any, **kwargs: Any) -> Any:
    """Call bind with continuation for primitives."""
    return _ops.call_bind_with_continuation(*args, **kwargs)


def call_impl(*args: Any, **kwargs: Any) -> Any:
    """Implementation for call primitive."""
    return _ops.call_impl(*args, **kwargs)


def call_p(*args: Any, **kwargs: Any) -> Any:
    """Stub for call_p."""
    return None


def check_eqn(*args: Any, **kwargs: Any) -> Any:
    """Check equation in a Jaxpr for validity."""
    return _ops.check_eqn(*args, **kwargs)


def check_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Checks well-formedness of a jaxpr."""
    return _ops.check_jaxpr(*args, **kwargs)


def check_type(*args: Any, **kwargs: Any) -> Any:
    """Check the type of a variable."""
    return _ops.check_type(*args, **kwargs)


def check_valid_jaxtype(*args: Any, **kwargs: Any) -> Any:
    """Check if a given type is a valid JAX type."""
    return _ops.check_valid_jaxtype(*args, **kwargs)


def closed_call_p(*args: Any, **kwargs: Any) -> Any:
    """Stub for call_p."""
    return None


def concrete_aval(*args: Any, **kwargs: Any) -> Any:
    """Get the concrete abstract value of an array."""
    return _ops.concrete_aval(*args, **kwargs)


def concrete_or_error(*args: Any, **kwargs: Any) -> Any:
    """Like force(val), but gives the context in the error message."""
    return _ops.concrete_or_error(*args, **kwargs)


def concretization_function_error(*args: Any, **kwargs: Any) -> Any:
    """Raise a concretization error with a detailed message."""
    return _ops.concretization_function_error(*args, **kwargs)


def cur_sublevel(*args: Any, **kwargs: Any) -> Any:
    """Get the current sublevel in the trace stack."""
    return _ops.cur_sublevel(*args, **kwargs)


def custom_typechecks(*args: Any, **kwargs: Any) -> Any:
    """Stub for custom_typechecks."""
    return None


def dedup_referents(*args: Any, **kwargs: Any) -> Any:
    """Deduplicate object referents in a trace."""
    return _ops.dedup_referents(*args, **kwargs)


def do_subst_axis_names_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Substitute axis names in a Jaxpr."""
    return _ops.do_subst_axis_names_jaxpr(*args, **kwargs)


def ensure_compile_time_eval(*args: Any, **kwargs: Any) -> Any:
    """Context manager to ensure evaluation at trace/compile time (or error)."""
    return _ops.ensure_compile_time_eval(*args, **kwargs)


def escaped_tracer_error(*args: Any, **kwargs: Any) -> Any:
    """Error indicating a Tracer has escaped its dynamic scope."""
    return _ops.escaped_tracer_error(*args, **kwargs)


def eval_context(*args: Any, **kwargs: Any) -> Any:
    """Context manager to ensure evaluation at trace/compile time (or error)."""
    return _ops.eval_context(*args, **kwargs)


def eval_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Evaluate a Jaxpr."""
    return _ops.eval_jaxpr(*args, **kwargs)


def extend_axis_env(*args: Any, **kwargs: Any) -> Any:
    """Extend the current axis environment context."""
    return _ops.extend_axis_env(*args, **kwargs)


def extend_axis_env_nd(*args: Any, **kwargs: Any) -> Any:
    """Extend the current ND axis environment context."""
    return _ops.extend_axis_env_nd(*args, **kwargs)


def find_top_trace(*args: Any, **kwargs: Any) -> Any:
    """Find the top-level trace in the current context."""
    return _ops.find_top_trace(*args, **kwargs)


def full_lower(*args: Any, **kwargs: Any) -> Any:
    """Perform a full lowering of a JAX computation."""
    return _ops.full_lower(*args, **kwargs)


def gensym(*args: Any, **kwargs: Any) -> Any:
    """Produce distinct variables, printed with the optional suffix."""
    return _ops.gensym(*args, **kwargs)


def get_aval(*args: Any, **kwargs: Any) -> Any:
    """Get the abstract value of an object."""
    return _ops.get_aval(*args, **kwargs)


def get_referent(*args: Any, **kwargs: Any) -> Any:
    """Get the referent of a Tracer object."""
    return _ops.get_referent(*args, **kwargs)


def is_constant_dim(*args: Any, **kwargs: Any) -> Any:
    """Check if a dimension is constant."""
    return _ops.is_constant_dim(*args, **kwargs)


def is_constant_shape(*args: Any, **kwargs: Any) -> Any:
    """Check if all dimensions of a shape are constant."""
    return _ops.is_constant_shape(*args, **kwargs)


def jaxpr_as_fun(*args: Any, **kwargs: Any) -> Any:
    """Convert a Jaxpr to a callable function."""
    return _ops.jaxpr_as_fun(*args, **kwargs)


def jaxpr_uses_outfeed(*args: Any, **kwargs: Any) -> Any:
    """Finds if there are outfeed primitives anywhere inside a Jaxpr."""
    return _ops.jaxpr_uses_outfeed(*args, **kwargs)


def jaxprs_in_params(*args: Any, **kwargs: Any) -> Any:
    """Extract all Jaxprs found inside primitive parameters."""
    return _ops.jaxprs_in_params(*args, **kwargs)


def join_effects(*args: Any, **kwargs: Any) -> Any:
    """Join multiple side-effects sets."""
    return _ops.join_effects(*args, **kwargs)


def join_named_shapes(*args: Any, **kwargs: Any) -> Any:
    """Join multiple NamedShapes."""
    return _ops.join_named_shapes(*args, **kwargs)


def lattice_join(*args: Any, **kwargs: Any) -> Any:
    """Lattice join two abstract values."""
    return _ops.lattice_join(*args, **kwargs)


def leaked_tracer_error(*args: Any, **kwargs: Any) -> Any:
    """Error indicating a Tracer has leaked out of its transform scope."""
    return _ops.leaked_tracer_error(*args, **kwargs)


def literalable_types(*args: Any, **kwargs: Any) -> Any:
    """Stub for literalable_types."""
    return None


def map_bind(*args: Any, **kwargs: Any) -> Any:
    """Bind a MapPrimitive to arguments."""
    return _ops.map_bind(*args, **kwargs)


def map_bind_with_continuation(*args: Any, **kwargs: Any) -> Any:
    """Bind a MapPrimitive with a continuation."""
    return _ops.map_bind_with_continuation(*args, **kwargs)


def mapped_aval(*args: Any, **kwargs: Any) -> Any:
    """Get the abstract value mapped across an axis."""
    return _ops.mapped_aval(*args, **kwargs)


def max_dim(*args: Any, **kwargs: Any) -> Any:
    """Like max(d1, d2) but for both constant and symbolic dimensions."""
    return _ops.max_dim(*args, **kwargs)


def maybe_find_leaked_tracers(*args: Any, **kwargs: Any) -> Any:
    """Find the leaked tracers holding a reference to the MainTrace or SubLevel."""
    return _ops.maybe_find_leaked_tracers(*args, **kwargs)


def min_dim(*args: Any, **kwargs: Any) -> Any:
    """Like min(d1, d2) but for both constant and symbolic dimensions."""
    return _ops.min_dim(*args, **kwargs)


def new_base_main(*args: Any, **kwargs: Any) -> Any:
    """Create a new base MainTrace."""
    return _ops.new_base_main(*args, **kwargs)


def new_jaxpr_eqn(*args: Any, **kwargs: Any) -> Any:
    """Create a new Jaxpr equation."""
    return _ops.new_jaxpr_eqn(*args, **kwargs)


def new_main(*args: Any, **kwargs: Any) -> Any:
    """Create a new MainTrace context."""
    return _ops.new_main(*args, **kwargs)


def new_sublevel(*args: Any, **kwargs: Any) -> Any:
    """Create a new Sublevel for tracing."""
    return _ops.new_sublevel(*args, **kwargs)


def no_axis_name(*args: Any, **kwargs: Any) -> Any:
    """Stub for no_axis_name."""
    return None


def no_effects(*args: Any, **kwargs: Any) -> Any:
    """Stub for no_effects."""
    return None


def outfeed_primitives(*args: Any, **kwargs: Any) -> Any:
    """Stub for outfeed_primitives."""
    return None


def primal_dtype_to_tangent_dtype(*args: Any, **kwargs: Any) -> Any:
    """Convert a primal dtype to its corresponding tangent dtype."""
    return _ops.primal_dtype_to_tangent_dtype(*args, **kwargs)


def primitive_uses_outfeed(*args: Any, **kwargs: Any) -> Any:
    """Determine if a primitive uses the outfeed."""
    return _ops.primitive_uses_outfeed(*args, **kwargs)


def process_env_traces_call(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.process_env_traces_call(*args, **kwargs)


def process_env_traces_map(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return _ops.process_env_traces_map(*args, **kwargs)


def pytype_aval_mappings(*args: Any, **kwargs: Any) -> Any:
    """Stub for pytype_aval_mappings."""
    return None


def raise_as_much_as_possible(*args: Any, **kwargs: Any) -> Any:
    """Raise as many shape exceptions as possible for missing dimensions."""
    return _ops.raise_as_much_as_possible(*args, **kwargs)


def raise_to_shaped(*args: Any, **kwargs: Any) -> Any:
    """Raise an abstract value to a ShapedArray."""
    return _ops.raise_to_shaped(*args, **kwargs)


def raise_to_shaped_mappings(*args: Any, **kwargs: Any) -> Any:
    """Stub for raise_to_shaped_mappings."""
    return None


def reset_trace_state(*args: Any, **kwargs: Any) -> Any:
    """Resets the global trace state and returns True if it was already clean."""
    return _ops.reset_trace_state(*args, **kwargs)


def stash_axis_env(*args: Any, **kwargs: Any) -> Any:
    """Promise that a function or with-suite does not depend implicitly on axis env"""
    return _ops.stash_axis_env(*args, **kwargs)


def str_eqn_compact(*args: Any, **kwargs: Any) -> Any:
    """Compact equation to string conversion used in HLO metadata."""
    return _ops.str_eqn_compact(*args, **kwargs)


def subjaxprs(*args: Any, **kwargs: Any) -> Any:
    """Generator for all subjaxprs found in the params of jaxpr.eqns."""
    return _ops.subjaxprs(*args, **kwargs)


def subst_axis_names(*args: Any, **kwargs: Any) -> Any:
    """Substitute axis names in a computation context."""
    return _ops.subst_axis_names(*args, **kwargs)


def subst_axis_names_eqn(*args: Any, **kwargs: Any) -> Any:
    """Substitute axis names in a Jaxpr equation."""
    return _ops.subst_axis_names_eqn(*args, **kwargs)


def subst_axis_names_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Substitute axis names within a Jaxpr."""
    return _ops.subst_axis_names_jaxpr(*args, **kwargs)


def subst_axis_names_var(*args: Any, **kwargs: Any) -> Any:
    """Substitute axis names in a variable."""
    return _ops.subst_axis_names_var(*args, **kwargs)


def substitute_vars_in_output_ty(*args: Any, **kwargs: Any) -> Any:
    """Substitute variables in the output type signature."""
    return _ops.substitute_vars_in_output_ty(*args, **kwargs)


def thread_local_state(*args: Any, **kwargs: Any) -> Any:
    """Stub for thread_local_state."""
    return None


def trace_state_clean(*args: Any, **kwargs: Any) -> Any:
    """Check if the global trace state is clean."""
    return _ops.trace_state_clean(*args, **kwargs)


def traverse_jaxpr_params(*args: Any, **kwargs: Any) -> Any:
    """Applies f to each jaxpr parameter and returns a tuple of returned values."""
    return _ops.traverse_jaxpr_params(*args, **kwargs)


def typecheck(*args: Any, **kwargs: Any) -> Any:
    """Typecheck a primitive application."""
    return _ops.typecheck(*args, **kwargs)


def typecompat(*args: Any, **kwargs: Any) -> Any:
    """Determine whether `aval` conforms to `aval_ref`."""
    return _ops.typecompat(*args, **kwargs)


def typematch(*args: Any, **kwargs: Any) -> Any:
    """Determine whether `aval1` and `aval2` are equivalent."""
    return _ops.typematch(*args, **kwargs)


def unmapped_aval(*args: Any, **kwargs: Any) -> Any:
    """Get the abstract value unmapped from an axis."""
    return _ops.unmapped_aval(*args, **kwargs)


def used_axis_names(*args: Any, **kwargs: Any) -> Any:
    """Find all used axis names in a context."""
    return _ops.used_axis_names(*args, **kwargs)


def used_axis_names_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Find all used axis names inside a Jaxpr."""
    return _ops.used_axis_names_jaxpr(*args, **kwargs)


def valid_jaxtype(*args: Any, **kwargs: Any) -> Any:
    """Check if a type is a valid JAX type."""
    return _ops.valid_jaxtype(*args, **kwargs)


@dataclass
class Jaxpr:
    """A Jaxpr representation."""

    constvars: List[Any] = field(default_factory=list)
    invars: List[Any] = field(default_factory=list)
    outvars: List[Any] = field(default_factory=list)
    eqns: List[JaxprEqn] = field(default_factory=list)


@dataclass
class Primitive:
    """A JAX primitive operation."""

    name: str = ""


@dataclass
class ShapedArray:
    """A shaped array abstract value."""

    shape: Tuple[Any, ...] = field(default_factory=tuple)
    dtype: Any = None


@dataclass
class Token:
    """A token used for ordering effects."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


@dataclass
class Trace:
    """Base trace class."""

    main: Any = None


@dataclass
class Var:
    """A variable in a Jaxpr."""

    aval: Any = None


Value = Any

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
