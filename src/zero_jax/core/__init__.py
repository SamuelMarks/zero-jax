"""Frontend API routing for jax.core."""

from typing import Any


class AbstractToken:
    """Mock implementation for AbstractToken."""

    pass


class AbstractValue:
    """Mock implementation for AbstractValue."""

    pass


def Atom(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for Atom."""
    raise NotImplementedError("Atom not yet implemented in zero-jax")


def AxisSize(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for AxisSize."""
    raise NotImplementedError("AxisSize not yet implemented in zero-jax")


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
    raise NotImplementedError("apply_todos not yet implemented in zero-jax")


def as_named_shape(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for as_named_shape."""
    raise NotImplementedError("as_named_shape not yet implemented in zero-jax")


aval_mapping_handlers: Any = None


def axis_frame(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for axis_frame."""
    raise NotImplementedError("axis_frame not yet implemented in zero-jax")


def call(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call."""
    raise NotImplementedError("call not yet implemented in zero-jax")


def call_bind_with_continuation(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_bind_with_continuation."""
    raise NotImplementedError(
        "call_bind_with_continuation not yet implemented in zero-jax"
    )


def call_impl(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for call_impl."""
    raise NotImplementedError("call_impl not yet implemented in zero-jax")


call_p: Any = None


def check_eqn(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_eqn."""
    raise NotImplementedError("check_eqn not yet implemented in zero-jax")


def check_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Checks well-formedness of a jaxpr."""
    raise NotImplementedError("check_jaxpr not yet implemented in zero-jax")


def check_type(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_type."""
    raise NotImplementedError("check_type not yet implemented in zero-jax")


def check_valid_jaxtype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for check_valid_jaxtype."""
    raise NotImplementedError("check_valid_jaxtype not yet implemented in zero-jax")


closed_call_p: Any = None


def concrete_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for concrete_aval."""
    raise NotImplementedError("concrete_aval not yet implemented in zero-jax")


def concrete_or_error(*args: Any, **kwargs: Any) -> Any:
    """Like force(val), but gives the context in the error message."""
    raise NotImplementedError("concrete_or_error not yet implemented in zero-jax")


def concretization_function_error(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for concretization_function_error."""
    raise NotImplementedError(
        "concretization_function_error not yet implemented in zero-jax"
    )


def cur_sublevel(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for cur_sublevel."""
    raise NotImplementedError("cur_sublevel not yet implemented in zero-jax")


custom_typechecks: Any = None


def dedup_referents(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dedup_referents."""
    raise NotImplementedError("dedup_referents not yet implemented in zero-jax")


def do_subst_axis_names_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for do_subst_axis_names_jaxpr."""
    raise NotImplementedError(
        "do_subst_axis_names_jaxpr not yet implemented in zero-jax"
    )


def ensure_compile_time_eval(*args: Any, **kwargs: Any) -> Any:
    """Context manager to ensure evaluation at trace/compile time (or error)."""
    raise NotImplementedError(
        "ensure_compile_time_eval not yet implemented in zero-jax"
    )


def escaped_tracer_error(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for escaped_tracer_error."""
    raise NotImplementedError("escaped_tracer_error not yet implemented in zero-jax")


def eval_context(*args: Any, **kwargs: Any) -> Any:
    """Context manager to ensure evaluation at trace/compile time (or error)."""
    raise NotImplementedError("eval_context not yet implemented in zero-jax")


def eval_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for eval_jaxpr."""
    raise NotImplementedError("eval_jaxpr not yet implemented in zero-jax")


def extend_axis_env(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for extend_axis_env."""
    raise NotImplementedError("extend_axis_env not yet implemented in zero-jax")


def extend_axis_env_nd(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for extend_axis_env_nd."""
    raise NotImplementedError("extend_axis_env_nd not yet implemented in zero-jax")


def find_top_trace(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for find_top_trace."""
    raise NotImplementedError("find_top_trace not yet implemented in zero-jax")


def full_lower(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for full_lower."""
    raise NotImplementedError("full_lower not yet implemented in zero-jax")


def gensym(*args: Any, **kwargs: Any) -> Any:
    """Produce distinct variables, printed with the optional suffix."""
    raise NotImplementedError("gensym not yet implemented in zero-jax")


def get_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_aval."""
    raise NotImplementedError("get_aval not yet implemented in zero-jax")


def get_referent(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_referent."""
    raise NotImplementedError("get_referent not yet implemented in zero-jax")


def is_constant_dim(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_constant_dim."""
    raise NotImplementedError("is_constant_dim not yet implemented in zero-jax")


def is_constant_shape(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_constant_shape."""
    raise NotImplementedError("is_constant_shape not yet implemented in zero-jax")


def jaxpr_as_fun(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for jaxpr_as_fun."""
    raise NotImplementedError("jaxpr_as_fun not yet implemented in zero-jax")


def jaxpr_uses_outfeed(*args: Any, **kwargs: Any) -> Any:
    """Finds if there are outfeed primitives anywhere inside a Jaxpr."""
    raise NotImplementedError("jaxpr_uses_outfeed not yet implemented in zero-jax")


def jaxprs_in_params(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for jaxprs_in_params."""
    raise NotImplementedError("jaxprs_in_params not yet implemented in zero-jax")


def join_effects(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for join_effects."""
    raise NotImplementedError("join_effects not yet implemented in zero-jax")


def join_named_shapes(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for join_named_shapes."""
    raise NotImplementedError("join_named_shapes not yet implemented in zero-jax")


def lattice_join(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for lattice_join."""
    raise NotImplementedError("lattice_join not yet implemented in zero-jax")


def leaked_tracer_error(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for leaked_tracer_error."""
    raise NotImplementedError("leaked_tracer_error not yet implemented in zero-jax")


literalable_types: Any = None


def map_bind(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for map_bind."""
    raise NotImplementedError("map_bind not yet implemented in zero-jax")


def map_bind_with_continuation(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for map_bind_with_continuation."""
    raise NotImplementedError(
        "map_bind_with_continuation not yet implemented in zero-jax"
    )


def mapped_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for mapped_aval."""
    raise NotImplementedError("mapped_aval not yet implemented in zero-jax")


def max_dim(*args: Any, **kwargs: Any) -> Any:
    """Like max(d1, d2) but for both constant and symbolic dimensions."""
    raise NotImplementedError("max_dim not yet implemented in zero-jax")


def maybe_find_leaked_tracers(*args: Any, **kwargs: Any) -> Any:
    """Find the leaked tracers holding a reference to the MainTrace or SubLevel."""
    raise NotImplementedError(
        "maybe_find_leaked_tracers not yet implemented in zero-jax"
    )


def min_dim(*args: Any, **kwargs: Any) -> Any:
    """Like min(d1, d2) but for both constant and symbolic dimensions."""
    raise NotImplementedError("min_dim not yet implemented in zero-jax")


def new_base_main(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_base_main."""
    raise NotImplementedError("new_base_main not yet implemented in zero-jax")


def new_jaxpr_eqn(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_jaxpr_eqn."""
    raise NotImplementedError("new_jaxpr_eqn not yet implemented in zero-jax")


def new_main(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_main."""
    raise NotImplementedError("new_main not yet implemented in zero-jax")


def new_sublevel(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for new_sublevel."""
    raise NotImplementedError("new_sublevel not yet implemented in zero-jax")


no_axis_name: Any = None

no_effects: Any = None

outfeed_primitives: Any = None


def primal_dtype_to_tangent_dtype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for primal_dtype_to_tangent_dtype."""
    raise NotImplementedError(
        "primal_dtype_to_tangent_dtype not yet implemented in zero-jax"
    )


def primitive_uses_outfeed(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for primitive_uses_outfeed."""
    raise NotImplementedError("primitive_uses_outfeed not yet implemented in zero-jax")


def process_env_traces_call(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("process_env_traces_call not yet implemented in zero-jax")


def process_env_traces_map(*args: Any, **kwargs: Any) -> Any:
    """partial(func, *args, **keywords) - new function with partial application"""
    raise NotImplementedError("process_env_traces_map not yet implemented in zero-jax")


pytype_aval_mappings: Any = None


def raise_as_much_as_possible(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for raise_as_much_as_possible."""
    raise NotImplementedError(
        "raise_as_much_as_possible not yet implemented in zero-jax"
    )


def raise_to_shaped(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for raise_to_shaped."""
    raise NotImplementedError("raise_to_shaped not yet implemented in zero-jax")


raise_to_shaped_mappings: Any = None


def reset_trace_state(*args: Any, **kwargs: Any) -> Any:
    """Resets the global trace state and returns True if it was already clean."""
    raise NotImplementedError("reset_trace_state not yet implemented in zero-jax")


def stash_axis_env(*args: Any, **kwargs: Any) -> Any:
    """Promise that a function or with-suite does not depend implicitly on axis env"""
    raise NotImplementedError("stash_axis_env not yet implemented in zero-jax")


def str_eqn_compact(*args: Any, **kwargs: Any) -> Any:
    """Compact equation to string conversion used in HLO metadata."""
    raise NotImplementedError("str_eqn_compact not yet implemented in zero-jax")


def subjaxprs(*args: Any, **kwargs: Any) -> Any:
    """Generator for all subjaxprs found in the params of jaxpr.eqns."""
    raise NotImplementedError("subjaxprs not yet implemented in zero-jax")


def subst_axis_names(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subst_axis_names."""
    raise NotImplementedError("subst_axis_names not yet implemented in zero-jax")


def subst_axis_names_eqn(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subst_axis_names_eqn."""
    raise NotImplementedError("subst_axis_names_eqn not yet implemented in zero-jax")


def subst_axis_names_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subst_axis_names_jaxpr."""
    raise NotImplementedError("subst_axis_names_jaxpr not yet implemented in zero-jax")


def subst_axis_names_var(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for subst_axis_names_var."""
    raise NotImplementedError("subst_axis_names_var not yet implemented in zero-jax")


def substitute_vars_in_output_ty(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for substitute_vars_in_output_ty."""
    raise NotImplementedError(
        "substitute_vars_in_output_ty not yet implemented in zero-jax"
    )


thread_local_state: Any = None


def trace_state_clean(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for trace_state_clean."""
    raise NotImplementedError("trace_state_clean not yet implemented in zero-jax")


def traverse_jaxpr_params(*args: Any, **kwargs: Any) -> Any:
    """Applies f to each jaxpr parameter and returns a tuple of returned values."""
    raise NotImplementedError("traverse_jaxpr_params not yet implemented in zero-jax")


def typecheck(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for typecheck."""
    raise NotImplementedError("typecheck not yet implemented in zero-jax")


def typecompat(*args: Any, **kwargs: Any) -> Any:
    """Determine whether `aval` conforms to `aval_ref`."""
    raise NotImplementedError("typecompat not yet implemented in zero-jax")


def typematch(*args: Any, **kwargs: Any) -> Any:
    """Determine whether `aval1` and `aval2` are equivalent."""
    raise NotImplementedError("typematch not yet implemented in zero-jax")


def unmapped_aval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for unmapped_aval."""
    raise NotImplementedError("unmapped_aval not yet implemented in zero-jax")


def used_axis_names(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for used_axis_names."""
    raise NotImplementedError("used_axis_names not yet implemented in zero-jax")


def used_axis_names_jaxpr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for used_axis_names_jaxpr."""
    raise NotImplementedError("used_axis_names_jaxpr not yet implemented in zero-jax")


def valid_jaxtype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for valid_jaxtype."""
    raise NotImplementedError("valid_jaxtype not yet implemented in zero-jax")
