"""Frontend API routing for jax.core."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class AbstractToken:
    """Mock implementation for AbstractToken."""

    pass


class AbstractValue:
    """Mock implementation for AbstractValue."""

    pass


def Atom(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for Atom."""
    return getattr(_ops, "Atom")(*args, **kwargs)


def AxisSize(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for AxisSize."""
    return getattr(_ops, "AxisSize")(*args, **kwargs)


class CallPrimitive:
    """Mock implementation for CallPrimitive."""

    pass


class ClosedJaxpr:
    """Mock implementation for ClosedJaxpr."""

    pass


class ConcreteArray:
    """Mock implementation for ConcreteArray."""

    pass


class ConcretizationTypeError:
    """This error occurs when a JAX Tracer object is used in a context where a"""

    pass


class DShapedArray:
    """Mock implementation for DShapedArray."""

    pass


class DropVar:
    """Mock implementation for DropVar."""

    pass


class Effect:
    """A generic side-effect."""

    pass


class Effects:
    """A set is a finite, iterable container."""

    pass


class EvalTrace:
    """Mock implementation for EvalTrace."""

    pass


class InDBIdx:
    """InDBIdx(val: 'int')"""

    pass


class InconclusiveDimensionOperation:
    """Raised when we cannot conclusively compute with symbolic dimensions."""

    pass


class InputType:
    """Built-in immutable sequence."""

    pass


class JaxprDebugInfo:
    """JaxprDebugInfo(traced_for, func_src_info, arg_names, result_paths)"""

    pass


class JaxprEqn:
    """JaxprEqn(invars, outvars, primitive, params, effects, source_info, ctx)"""

    pass


class JaxprPpContext:
    """Mock implementation for JaxprPpContext."""

    pass


class JaxprPpSettings:
    """JaxprPpSettings(print_shapes, source_info, name_stack, custom_pp_eqn_rules, print_effects)"""

    pass


class JaxprTypeError:
    """Mock implementation for JaxprTypeError."""

    pass


class Literal:
    """Mock implementation for Literal."""

    pass


class MainTrace:
    """Mock implementation for MainTrace."""

    pass


class MapPrimitive:
    """Mock implementation for MapPrimitive."""

    pass


class NameGatheringSubst:
    """Mock implementation for NameGatheringSubst."""

    pass


class NamedShape:
    """Mock implementation for NamedShape."""

    pass


class OutDBIdx:
    """OutDBIdx(val: 'int')"""

    pass


class OutputType:
    """Built-in immutable sequence."""

    pass


class ParamDict:
    """dict() -> new empty dictionary"""

    pass


class Sublevel:
    """Mock implementation for Sublevel."""

    pass


TRACER_LEAK_DEBUGGER_WARNING: Any = None


class ThreadLocalState:
    """Mock implementation for ThreadLocalState."""

    pass


class TraceStack:
    """Mock implementation for TraceStack."""

    pass


class TraceState:
    """Mock implementation for TraceState."""

    pass


class Tracer:
    """Mock implementation for Tracer."""

    pass


class UnshapedArray:
    """Mock implementation for UnshapedArray."""

    pass


abstract_token: Any = None


def apply_todos(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for apply_todos."""
    return getattr(_ops, "apply_todos")(*args, **kwargs)


def as_named_shape(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for as_named_shape."""
    return getattr(_ops, "as_named_shape")(*args, **kwargs)


aval_mapping_handlers: Any = None


def axis_frame(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for axis_frame."""
    return getattr(_ops, "axis_frame")(*args, **kwargs)


def call(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call."""
    return getattr(_ops, "call")(*args, **kwargs)


def call_bind_with_continuation(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_bind_with_continuation."""
    return getattr(_ops, "call_bind_with_continuation")(*args, **kwargs)


def call_impl(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_impl."""
    return getattr(_ops, "call_impl")(*args, **kwargs)


call_p: Any = None


def check_eqn(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_eqn."""
    return getattr(_ops, "check_eqn")(*args, **kwargs)


def check_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Checks well-formedness of a jaxpr."""
    return getattr(_ops, "check_jaxpr")(*args, **kwargs)


def check_type(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_type."""
    return getattr(_ops, "check_type")(*args, **kwargs)


def check_valid_jaxtype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_valid_jaxtype."""
    return getattr(_ops, "check_valid_jaxtype")(*args, **kwargs)


closed_call_p: Any = None


def concrete_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for concrete_aval."""
    return getattr(_ops, "concrete_aval")(*args, **kwargs)


def concrete_or_error(*args: Any, **kwargs: Any) -> Any:
    """Like force(val), but gives the context in the error message."""
    return getattr(_ops, "concrete_or_error")(*args, **kwargs)


def concretization_function_error(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for concretization_function_error."""
    return getattr(_ops, "concretization_function_error")(*args, **kwargs)


def cur_sublevel(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for cur_sublevel."""
    return getattr(_ops, "cur_sublevel")(*args, **kwargs)


custom_typechecks: Any = None


def dedup_referents(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dedup_referents."""
    return getattr(_ops, "dedup_referents")(*args, **kwargs)


def do_subst_axis_names_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for do_subst_axis_names_jaxpr."""
    return getattr(_ops, "do_subst_axis_names_jaxpr")(*args, **kwargs)


def ensure_compile_time_eval(*args: Any, **kwargs: Any) -> Any:
    """Context manager to ensure evaluation at trace/compile time (or error)."""
    return getattr(_ops, "ensure_compile_time_eval")(*args, **kwargs)


def escaped_tracer_error(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for escaped_tracer_error."""
    return getattr(_ops, "escaped_tracer_error")(*args, **kwargs)


def eval_context(*args: Any, **kwargs: Any) -> Any:
    """Context manager to ensure evaluation at trace/compile time (or error)."""
    return getattr(_ops, "eval_context")(*args, **kwargs)


def eval_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for eval_jaxpr."""
    return getattr(_ops, "eval_jaxpr")(*args, **kwargs)


def extend_axis_env(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for extend_axis_env."""
    return getattr(_ops, "extend_axis_env")(*args, **kwargs)


def extend_axis_env_nd(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for extend_axis_env_nd."""
    return getattr(_ops, "extend_axis_env_nd")(*args, **kwargs)


def find_top_trace(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for find_top_trace."""
    return getattr(_ops, "find_top_trace")(*args, **kwargs)


def full_lower(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for full_lower."""
    return getattr(_ops, "full_lower")(*args, **kwargs)


def gensym(*args: Any, **kwargs: Any) -> Any:
    """Produce distinct variables, printed with the optional suffix."""
    return getattr(_ops, "gensym")(*args, **kwargs)


def get_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_aval."""
    return getattr(_ops, "get_aval")(*args, **kwargs)


def get_referent(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_referent."""
    return getattr(_ops, "get_referent")(*args, **kwargs)


def is_constant_dim(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_constant_dim."""
    return getattr(_ops, "is_constant_dim")(*args, **kwargs)


def is_constant_shape(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_constant_shape."""
    return getattr(_ops, "is_constant_shape")(*args, **kwargs)


def jaxpr_as_fun(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for jaxpr_as_fun."""
    return getattr(_ops, "jaxpr_as_fun")(*args, **kwargs)


def jaxpr_uses_outfeed(*args: Any, **kwargs: Any) -> Any:
    """Finds if there are outfeed primitives anywhere inside a Jaxpr."""
    return getattr(_ops, "jaxpr_uses_outfeed")(*args, **kwargs)


def jaxprs_in_params(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for jaxprs_in_params."""
    return getattr(_ops, "jaxprs_in_params")(*args, **kwargs)


def join_effects(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for join_effects."""
    return getattr(_ops, "join_effects")(*args, **kwargs)


def join_named_shapes(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for join_named_shapes."""
    return getattr(_ops, "join_named_shapes")(*args, **kwargs)


def lattice_join(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for lattice_join."""
    return getattr(_ops, "lattice_join")(*args, **kwargs)


def leaked_tracer_error(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for leaked_tracer_error."""
    return getattr(_ops, "leaked_tracer_error")(*args, **kwargs)


literalable_types: Any = None


def map_bind(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for map_bind."""
    return getattr(_ops, "map_bind")(*args, **kwargs)


def map_bind_with_continuation(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for map_bind_with_continuation."""
    return getattr(_ops, "map_bind_with_continuation")(*args, **kwargs)


def mapped_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for mapped_aval."""
    return getattr(_ops, "mapped_aval")(*args, **kwargs)


def max_dim(*args: Any, **kwargs: Any) -> Any:
    """Like max(d1, d2) but for both constant and symbolic dimensions."""
    return getattr(_ops, "max_dim")(*args, **kwargs)


def maybe_find_leaked_tracers(*args: Any, **kwargs: Any) -> Any:
    """Find the leaked tracers holding a reference to the MainTrace or SubLevel."""
    return getattr(_ops, "maybe_find_leaked_tracers")(*args, **kwargs)


def min_dim(*args: Any, **kwargs: Any) -> Any:
    """Like min(d1, d2) but for both constant and symbolic dimensions."""
    return getattr(_ops, "min_dim")(*args, **kwargs)


def new_base_main(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_base_main."""
    return getattr(_ops, "new_base_main")(*args, **kwargs)


def new_jaxpr_eqn(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_jaxpr_eqn."""
    return getattr(_ops, "new_jaxpr_eqn")(*args, **kwargs)


def new_main(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_main."""
    return getattr(_ops, "new_main")(*args, **kwargs)


def new_sublevel(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_sublevel."""
    return getattr(_ops, "new_sublevel")(*args, **kwargs)


no_axis_name: Any = None

no_effects: Any = None

outfeed_primitives: Any = None


def primal_dtype_to_tangent_dtype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for primal_dtype_to_tangent_dtype."""
    return getattr(_ops, "primal_dtype_to_tangent_dtype")(*args, **kwargs)


def primitive_uses_outfeed(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for primitive_uses_outfeed."""
    return getattr(_ops, "primitive_uses_outfeed")(*args, **kwargs)


def process_env_traces_call(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "process_env_traces_call")(*args, **kwargs)


def process_env_traces_map(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    return getattr(_ops, "process_env_traces_map")(*args, **kwargs)


pytype_aval_mappings: Any = None


def raise_as_much_as_possible(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for raise_as_much_as_possible."""
    return getattr(_ops, "raise_as_much_as_possible")(*args, **kwargs)


def raise_to_shaped(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for raise_to_shaped."""
    return getattr(_ops, "raise_to_shaped")(*args, **kwargs)


raise_to_shaped_mappings: Any = None


def reset_trace_state(*args: Any, **kwargs: Any) -> Any:
    """Resets the global trace state and returns True if it was already clean."""
    return getattr(_ops, "reset_trace_state")(*args, **kwargs)


def stash_axis_env(*args: Any, **kwargs: Any) -> Any:
    """Promise that a function or with-suite does not depend implicitly on axis env"""
    return getattr(_ops, "stash_axis_env")(*args, **kwargs)


def str_eqn_compact(*args: Any, **kwargs: Any) -> Any:
    """Compact equation to string conversion used in HLO metadata."""
    return getattr(_ops, "str_eqn_compact")(*args, **kwargs)


def subjaxprs(*args: Any, **kwargs: Any) -> Any:
    """Generator for all subjaxprs found in the params of jaxpr.eqns."""
    return getattr(_ops, "subjaxprs")(*args, **kwargs)


def subst_axis_names(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subst_axis_names."""
    return getattr(_ops, "subst_axis_names")(*args, **kwargs)


def subst_axis_names_eqn(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subst_axis_names_eqn."""
    return getattr(_ops, "subst_axis_names_eqn")(*args, **kwargs)


def subst_axis_names_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subst_axis_names_jaxpr."""
    return getattr(_ops, "subst_axis_names_jaxpr")(*args, **kwargs)


def subst_axis_names_var(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subst_axis_names_var."""
    return getattr(_ops, "subst_axis_names_var")(*args, **kwargs)


def substitute_vars_in_output_ty(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for substitute_vars_in_output_ty."""
    return getattr(_ops, "substitute_vars_in_output_ty")(*args, **kwargs)


thread_local_state: Any = None


def trace_state_clean(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_state_clean."""
    return getattr(_ops, "trace_state_clean")(*args, **kwargs)


def traverse_jaxpr_params(*args: Any, **kwargs: Any) -> Any:
    """Applies f to each jaxpr parameter and returns a tuple of returned values."""
    return getattr(_ops, "traverse_jaxpr_params")(*args, **kwargs)


def typecheck(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for typecheck."""
    return getattr(_ops, "typecheck")(*args, **kwargs)


def typecompat(*args: Any, **kwargs: Any) -> Any:
    """Determine whether `aval` conforms to `aval_ref`."""
    return getattr(_ops, "typecompat")(*args, **kwargs)


def typematch(*args: Any, **kwargs: Any) -> Any:
    """Determine whether `aval1` and `aval2` are equivalent."""
    return getattr(_ops, "typematch")(*args, **kwargs)


def unmapped_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for unmapped_aval."""
    return getattr(_ops, "unmapped_aval")(*args, **kwargs)


def used_axis_names(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for used_axis_names."""
    return getattr(_ops, "used_axis_names")(*args, **kwargs)


def used_axis_names_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for used_axis_names_jaxpr."""
    return getattr(_ops, "used_axis_names_jaxpr")(*args, **kwargs)


def valid_jaxtype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for valid_jaxtype."""
    return getattr(_ops, "valid_jaxtype")(*args, **kwargs)


class Jaxpr:
    """Mock implementation for Jaxpr."""

    pass


class Primitive:
    """Mock implementation for Primitive."""

    pass


class ShapedArray:
    """Mock implementation for ShapedArray."""

    pass


class Token:
    """Mock implementation for Token."""

    pass


class Trace:
    """Mock implementation for Trace."""

    pass


class Var:
    """Mock implementation for Var."""

    pass


Value = Any
