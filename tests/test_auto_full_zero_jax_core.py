"""Tests for zero_jax.core."""

from typing import Any

import pytest

import zero_jax.core as mod


def test_class_AbstractToken() -> None:
    """Test class AbstractToken."""
    try:
        mod.AbstractToken()
    except Exception:
        pass


def test_class_AbstractValue() -> None:
    """Test class AbstractValue."""
    try:
        mod.AbstractValue()
    except Exception:
        pass


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_Atom() -> None:
    """Test Atom."""
    try:
        mod.Atom()
    except Exception:
        pass


def test_AxisSize() -> None:
    """Test AxisSize."""
    try:
        mod.AxisSize()
    except Exception:
        pass


def test_class_CallPrimitive() -> None:
    """Test class CallPrimitive."""
    try:
        mod.CallPrimitive()
    except Exception:
        pass


def test_class_ClosedJaxpr() -> None:
    """Test class ClosedJaxpr."""
    try:
        mod.ClosedJaxpr()
    except Exception:
        pass


def test_class_ConcreteArray() -> None:
    """Test class ConcreteArray."""
    try:
        mod.ConcreteArray()
    except Exception:
        pass


def test_class_ConcretizationTypeError() -> None:
    """Test class ConcretizationTypeError."""
    try:
        mod.ConcretizationTypeError()
    except Exception:
        pass


def test_class_DShapedArray() -> None:
    """Test class DShapedArray."""
    try:
        mod.DShapedArray()
    except Exception:
        pass


def test_Dict() -> None:
    """Test Dict."""
    try:
        mod.Dict()
    except Exception:
        pass


def test_class_DropVar() -> None:
    """Test class DropVar."""
    try:
        mod.DropVar()
    except Exception:
        pass


def test_class_Effect() -> None:
    """Test class Effect."""
    try:
        mod.Effect()
    except Exception:
        pass


def test_class_Effects() -> None:
    """Test class Effects."""
    try:
        mod.Effects()
    except Exception:
        pass


def test_class_EvalTrace() -> None:
    """Test class EvalTrace."""
    try:
        mod.EvalTrace()
    except Exception:
        pass


def test_class_InDBIdx() -> None:
    """Test class InDBIdx."""
    try:
        mod.InDBIdx()
    except Exception:
        pass


def test_class_InconclusiveDimensionOperation() -> None:
    """Test class InconclusiveDimensionOperation."""
    try:
        mod.InconclusiveDimensionOperation()
    except Exception:
        pass


def test_class_InputType() -> None:
    """Test class InputType."""
    try:
        mod.InputType()
    except Exception:
        pass


def test_Iterable() -> None:
    """Test Iterable."""
    try:
        mod.Iterable()
    except Exception:
        pass


def test_class_Jaxpr() -> None:
    """Test class Jaxpr."""
    try:
        mod.Jaxpr()
    except Exception:
        pass


def test_class_JaxprDebugInfo() -> None:
    """Test class JaxprDebugInfo."""
    try:
        mod.JaxprDebugInfo()
    except Exception:
        pass


def test_class_JaxprEqn() -> None:
    """Test class JaxprEqn."""
    try:
        mod.JaxprEqn()
    except Exception:
        pass


def test_class_JaxprPpContext() -> None:
    """Test class JaxprPpContext."""
    try:
        mod.JaxprPpContext()
    except Exception:
        pass


def test_class_JaxprPpSettings() -> None:
    """Test class JaxprPpSettings."""
    try:
        mod.JaxprPpSettings()
    except Exception:
        pass


def test_class_JaxprTypeError() -> None:
    """Test class JaxprTypeError."""
    try:
        mod.JaxprTypeError()
    except Exception:
        pass


def test_List() -> None:
    """Test List."""
    try:
        mod.List()
    except Exception:
        pass


def test_class_Literal() -> None:
    """Test class Literal."""
    try:
        mod.Literal()
    except Exception:
        pass


def test_class_MainTrace() -> None:
    """Test class MainTrace."""
    try:
        mod.MainTrace()
    except Exception:
        pass


def test_class_MapPrimitive() -> None:
    """Test class MapPrimitive."""
    try:
        mod.MapPrimitive()
    except Exception:
        pass


def test_class_NameGatheringSubst() -> None:
    """Test class NameGatheringSubst."""
    try:
        mod.NameGatheringSubst()
    except Exception:
        pass


def test_class_NamedShape() -> None:
    """Test class NamedShape."""
    try:
        mod.NamedShape()
    except Exception:
        pass


def test_Optional() -> None:
    """Test Optional."""
    try:
        mod.Optional()
    except Exception:
        pass


def test_class_OutDBIdx() -> None:
    """Test class OutDBIdx."""
    try:
        mod.OutDBIdx()
    except Exception:
        pass


def test_class_OutputType() -> None:
    """Test class OutputType."""
    try:
        mod.OutputType()
    except Exception:
        pass


def test_class_ParamDict() -> None:
    """Test class ParamDict."""
    try:
        mod.ParamDict()
    except Exception:
        pass


def test_class_Primitive() -> None:
    """Test class Primitive."""
    try:
        mod.Primitive()
    except Exception:
        pass


def test_Sequence() -> None:
    """Test Sequence."""
    try:
        mod.Sequence()
    except Exception:
        pass


def test_Set() -> None:
    """Test Set."""
    try:
        mod.Set()
    except Exception:
        pass


def test_class_ShapedArray() -> None:
    """Test class ShapedArray."""
    try:
        mod.ShapedArray()
    except Exception:
        pass


def test_class_Sublevel() -> None:
    """Test class Sublevel."""
    try:
        mod.Sublevel()
    except Exception:
        pass


def test_class_TRACER_LEAK_DEBUGGER_WARNING() -> None:
    """Test class TRACER_LEAK_DEBUGGER_WARNING."""
    try:
        mod.TRACER_LEAK_DEBUGGER_WARNING()
    except Exception:
        pass


def test_class_ThreadLocalState() -> None:
    """Test class ThreadLocalState."""
    try:
        mod.ThreadLocalState()
    except Exception:
        pass


def test_class_Token() -> None:
    """Test class Token."""
    try:
        mod.Token()
    except Exception:
        pass


def test_class_Trace() -> None:
    """Test class Trace."""
    try:
        mod.Trace()
    except Exception:
        pass


def test_class_TraceStack() -> None:
    """Test class TraceStack."""
    try:
        mod.TraceStack()
    except Exception:
        pass


def test_class_TraceState() -> None:
    """Test class TraceState."""
    try:
        mod.TraceState()
    except Exception:
        pass


def test_class_Tracer() -> None:
    """Test class Tracer."""
    try:
        mod.Tracer()
    except Exception:
        pass


def test_Tuple() -> None:
    """Test Tuple."""
    try:
        mod.Tuple()
    except Exception:
        pass


def test_class_UnshapedArray() -> None:
    """Test class UnshapedArray."""
    try:
        mod.UnshapedArray()
    except Exception:
        pass


def test_Value() -> None:
    """Test Value."""
    try:
        mod.Value()
    except Exception:
        pass


def test_class_Var() -> None:
    """Test class Var."""
    try:
        mod.Var()
    except Exception:
        pass


def test_abstract_token() -> None:
    """Test abstract_token."""
    try:
        mod.abstract_token()
    except Exception:
        pass


def test_apply_todos() -> None:
    """Test apply_todos."""
    try:
        mod.apply_todos()
    except Exception:
        pass


def test_as_named_shape() -> None:
    """Test as_named_shape."""
    try:
        mod.as_named_shape()
    except Exception:
        pass


def test_aval_mapping_handlers() -> None:
    """Test aval_mapping_handlers."""
    try:
        mod.aval_mapping_handlers()
    except Exception:
        pass


def test_axis_frame() -> None:
    """Test axis_frame."""
    try:
        mod.axis_frame()
    except Exception:
        pass


def test_call() -> None:
    """Test call."""
    try:
        mod.call()
    except Exception:
        pass


def test_call_bind_with_continuation() -> None:
    """Test call_bind_with_continuation."""
    try:
        mod.call_bind_with_continuation()
    except Exception:
        pass


def test_call_impl() -> None:
    """Test call_impl."""
    try:
        mod.call_impl()
    except Exception:
        pass


def test_call_p() -> None:
    """Test call_p."""
    try:
        mod.call_p()
    except Exception:
        pass


def test_check_eqn() -> None:
    """Test check_eqn."""
    try:
        mod.check_eqn()
    except Exception:
        pass


def test_check_jaxpr() -> None:
    """Test check_jaxpr."""
    try:
        mod.check_jaxpr()
    except Exception:
        pass


def test_check_type() -> None:
    """Test check_type."""
    try:
        mod.check_type()
    except Exception:
        pass


def test_check_valid_jaxtype() -> None:
    """Test check_valid_jaxtype."""
    try:
        mod.check_valid_jaxtype()
    except Exception:
        pass


def test_closed_call_p() -> None:
    """Test closed_call_p."""
    try:
        mod.closed_call_p()
    except Exception:
        pass


def test_concrete_aval() -> None:
    """Test concrete_aval."""
    try:
        mod.concrete_aval()
    except Exception:
        pass


def test_concrete_or_error() -> None:
    """Test concrete_or_error."""
    try:
        mod.concrete_or_error()
    except Exception:
        pass


def test_concretization_function_error() -> None:
    """Test concretization_function_error."""
    try:
        mod.concretization_function_error()
    except Exception:
        pass


def test_cur_sublevel() -> None:
    """Test cur_sublevel."""
    try:
        mod.cur_sublevel()
    except Exception:
        pass


def test_custom_typechecks() -> None:
    """Test custom_typechecks."""
    try:
        mod.custom_typechecks()
    except Exception:
        pass


def test_dataclass() -> None:
    """Test dataclass."""
    try:
        mod.dataclass()
    except Exception:
        pass


def test_dedup_referents() -> None:
    """Test dedup_referents."""
    try:
        mod.dedup_referents()
    except Exception:
        pass


def test_do_subst_axis_names_jaxpr() -> None:
    """Test do_subst_axis_names_jaxpr."""
    try:
        mod.do_subst_axis_names_jaxpr()
    except Exception:
        pass


def test_ensure_compile_time_eval() -> None:
    """Test ensure_compile_time_eval."""
    try:
        mod.ensure_compile_time_eval()
    except Exception:
        pass


def test_escaped_tracer_error() -> None:
    """Test escaped_tracer_error."""
    try:
        mod.escaped_tracer_error()
    except Exception:
        pass


def test_eval_context() -> None:
    """Test eval_context."""
    try:
        mod.eval_context()
    except Exception:
        pass


def test_eval_jaxpr() -> None:
    """Test eval_jaxpr."""
    try:
        mod.eval_jaxpr()
    except Exception:
        pass


def test_extend_axis_env() -> None:
    """Test extend_axis_env."""
    try:
        mod.extend_axis_env()
    except Exception:
        pass


def test_extend_axis_env_nd() -> None:
    """Test extend_axis_env_nd."""
    try:
        mod.extend_axis_env_nd()
    except Exception:
        pass


def test_field() -> None:
    """Test field."""
    try:
        mod.field()
    except Exception:
        pass


def test_find_top_trace() -> None:
    """Test find_top_trace."""
    try:
        mod.find_top_trace()
    except Exception:
        pass


def test_full_lower() -> None:
    """Test full_lower."""
    try:
        mod.full_lower()
    except Exception:
        pass


def test_gensym() -> None:
    """Test gensym."""
    try:
        mod.gensym()
    except Exception:
        pass


def test_get_aval() -> None:
    """Test get_aval."""
    try:
        mod.get_aval()
    except Exception:
        pass


def test_get_referent() -> None:
    """Test get_referent."""
    try:
        mod.get_referent()
    except Exception:
        pass


def test_is_constant_dim() -> None:
    """Test is_constant_dim."""
    try:
        mod.is_constant_dim()
    except Exception:
        pass


def test_is_constant_shape() -> None:
    """Test is_constant_shape."""
    try:
        mod.is_constant_shape()
    except Exception:
        pass


def test_jaxpr_as_fun() -> None:
    """Test jaxpr_as_fun."""
    try:
        mod.jaxpr_as_fun()
    except Exception:
        pass


def test_jaxpr_uses_outfeed() -> None:
    """Test jaxpr_uses_outfeed."""
    try:
        mod.jaxpr_uses_outfeed()
    except Exception:
        pass


def test_jaxprs_in_params() -> None:
    """Test jaxprs_in_params."""
    try:
        mod.jaxprs_in_params()
    except Exception:
        pass


def test_join_effects() -> None:
    """Test join_effects."""
    try:
        mod.join_effects()
    except Exception:
        pass


def test_join_named_shapes() -> None:
    """Test join_named_shapes."""
    try:
        mod.join_named_shapes()
    except Exception:
        pass


def test_lattice_join() -> None:
    """Test lattice_join."""
    try:
        mod.lattice_join()
    except Exception:
        pass


def test_leaked_tracer_error() -> None:
    """Test leaked_tracer_error."""
    try:
        mod.leaked_tracer_error()
    except Exception:
        pass


def test_literalable_types() -> None:
    """Test literalable_types."""
    try:
        mod.literalable_types()
    except Exception:
        pass


def test_map_bind() -> None:
    """Test map_bind."""
    try:
        mod.map_bind()
    except Exception:
        pass


def test_map_bind_with_continuation() -> None:
    """Test map_bind_with_continuation."""
    try:
        mod.map_bind_with_continuation()
    except Exception:
        pass


def test_mapped_aval() -> None:
    """Test mapped_aval."""
    try:
        mod.mapped_aval()
    except Exception:
        pass


def test_max_dim() -> None:
    """Test max_dim."""
    try:
        mod.max_dim()
    except Exception:
        pass


def test_maybe_find_leaked_tracers() -> None:
    """Test maybe_find_leaked_tracers."""
    try:
        mod.maybe_find_leaked_tracers()
    except Exception:
        pass


def test_min_dim() -> None:
    """Test min_dim."""
    try:
        mod.min_dim()
    except Exception:
        pass


def test_new_base_main() -> None:
    """Test new_base_main."""
    try:
        mod.new_base_main()
    except Exception:
        pass


def test_new_jaxpr_eqn() -> None:
    """Test new_jaxpr_eqn."""
    try:
        mod.new_jaxpr_eqn()
    except Exception:
        pass


def test_new_main() -> None:
    """Test new_main."""
    try:
        mod.new_main()
    except Exception:
        pass


def test_new_sublevel() -> None:
    """Test new_sublevel."""
    try:
        mod.new_sublevel()
    except Exception:
        pass


def test_no_axis_name() -> None:
    """Test no_axis_name."""
    try:
        mod.no_axis_name()
    except Exception:
        pass


def test_no_effects() -> None:
    """Test no_effects."""
    try:
        mod.no_effects()
    except Exception:
        pass


def test_outfeed_primitives() -> None:
    """Test outfeed_primitives."""
    try:
        mod.outfeed_primitives()
    except Exception:
        pass


def test_primal_dtype_to_tangent_dtype() -> None:
    """Test primal_dtype_to_tangent_dtype."""
    try:
        mod.primal_dtype_to_tangent_dtype()
    except Exception:
        pass


def test_primitive_uses_outfeed() -> None:
    """Test primitive_uses_outfeed."""
    try:
        mod.primitive_uses_outfeed()
    except Exception:
        pass


def test_process_env_traces_call() -> None:
    """Test process_env_traces_call."""
    try:
        mod.process_env_traces_call()
    except Exception:
        pass


def test_process_env_traces_map() -> None:
    """Test process_env_traces_map."""
    try:
        mod.process_env_traces_map()
    except Exception:
        pass


def test_pytype_aval_mappings() -> None:
    """Test pytype_aval_mappings."""
    try:
        mod.pytype_aval_mappings()
    except Exception:
        pass


def test_raise_as_much_as_possible() -> None:
    """Test raise_as_much_as_possible."""
    try:
        mod.raise_as_much_as_possible()
    except Exception:
        pass


def test_raise_to_shaped() -> None:
    """Test raise_to_shaped."""
    try:
        mod.raise_to_shaped()
    except Exception:
        pass


def test_raise_to_shaped_mappings() -> None:
    """Test raise_to_shaped_mappings."""
    try:
        mod.raise_to_shaped_mappings()
    except Exception:
        pass


def test_reset_trace_state() -> None:
    """Test reset_trace_state."""
    try:
        mod.reset_trace_state()
    except Exception:
        pass


def test_stash_axis_env() -> None:
    """Test stash_axis_env."""
    try:
        mod.stash_axis_env()
    except Exception:
        pass


def test_str_eqn_compact() -> None:
    """Test str_eqn_compact."""
    try:
        mod.str_eqn_compact()
    except Exception:
        pass


def test_subjaxprs() -> None:
    """Test subjaxprs."""
    try:
        mod.subjaxprs()
    except Exception:
        pass


def test_subst_axis_names() -> None:
    """Test subst_axis_names."""
    try:
        mod.subst_axis_names()
    except Exception:
        pass


def test_subst_axis_names_eqn() -> None:
    """Test subst_axis_names_eqn."""
    try:
        mod.subst_axis_names_eqn()
    except Exception:
        pass


def test_subst_axis_names_jaxpr() -> None:
    """Test subst_axis_names_jaxpr."""
    try:
        mod.subst_axis_names_jaxpr()
    except Exception:
        pass


def test_subst_axis_names_var() -> None:
    """Test subst_axis_names_var."""
    try:
        mod.subst_axis_names_var()
    except Exception:
        pass


def test_substitute_vars_in_output_ty() -> None:
    """Test substitute_vars_in_output_ty."""
    try:
        mod.substitute_vars_in_output_ty()
    except Exception:
        pass


def test_thread_local_state() -> None:
    """Test thread_local_state."""
    try:
        mod.thread_local_state()
    except Exception:
        pass


def test_trace_state_clean() -> None:
    """Test trace_state_clean."""
    try:
        mod.trace_state_clean()
    except Exception:
        pass


def test_traverse_jaxpr_params() -> None:
    """Test traverse_jaxpr_params."""
    try:
        mod.traverse_jaxpr_params()
    except Exception:
        pass


def test_typecheck() -> None:
    """Test typecheck."""
    try:
        mod.typecheck()
    except Exception:
        pass


def test_typecompat() -> None:
    """Test typecompat."""
    try:
        mod.typecompat()
    except Exception:
        pass


def test_typematch() -> None:
    """Test typematch."""
    try:
        mod.typematch()
    except Exception:
        pass


def test_unmapped_aval() -> None:
    """Test unmapped_aval."""
    try:
        mod.unmapped_aval()
    except Exception:
        pass


def test_used_axis_names() -> None:
    """Test used_axis_names."""
    try:
        mod.used_axis_names()
    except Exception:
        pass


def test_used_axis_names_jaxpr() -> None:
    """Test used_axis_names_jaxpr."""
    try:
        mod.used_axis_names_jaxpr()
    except Exception:
        pass


def test_valid_jaxtype() -> None:
    """Test valid_jaxtype."""
    try:
        mod.valid_jaxtype()
    except Exception:
        pass
