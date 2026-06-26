"""Tests for zero_jax module."""

import pytest
import zero_jax.core as mod


def test_AbstractToken() -> None:
    """Test AbstractToken."""
    obj = mod.AbstractToken()
    assert obj is not None


def test_AbstractValue() -> None:
    """Test AbstractValue."""
    obj = mod.AbstractValue()
    assert obj is not None


def test_Atom() -> None:
    """Test Atom."""
    with pytest.raises(NotImplementedError):
        mod.Atom()


def test_AxisSize() -> None:
    """Test AxisSize."""
    with pytest.raises(NotImplementedError):
        mod.AxisSize()


def test_CallPrimitive() -> None:
    """Test CallPrimitive."""
    obj = mod.CallPrimitive()
    assert obj is not None


def test_ClosedJaxpr() -> None:
    """Test ClosedJaxpr."""
    obj = mod.ClosedJaxpr()
    assert obj is not None


def test_ConcreteArray() -> None:
    """Test ConcreteArray."""
    obj = mod.ConcreteArray()
    assert obj is not None


def test_ConcretizationTypeError() -> None:
    """Test ConcretizationTypeError."""
    obj = mod.ConcretizationTypeError()
    assert obj is not None


def test_DShapedArray() -> None:
    """Test DShapedArray."""
    obj = mod.DShapedArray()
    assert obj is not None


def test_DropVar() -> None:
    """Test DropVar."""
    obj = mod.DropVar()
    assert obj is not None


def test_Effect() -> None:
    """Test Effect."""
    obj = mod.Effect()
    assert obj is not None


def test_Effects() -> None:
    """Test Effects."""
    obj = mod.Effects()
    assert obj is not None


def test_EvalTrace() -> None:
    """Test EvalTrace."""
    obj = mod.EvalTrace()
    assert obj is not None


def test_InDBIdx() -> None:
    """Test InDBIdx."""
    obj = mod.InDBIdx()
    assert obj is not None


def test_InconclusiveDimensionOperation() -> None:
    """Test InconclusiveDimensionOperation."""
    obj = mod.InconclusiveDimensionOperation()
    assert obj is not None


def test_InputType() -> None:
    """Test InputType."""
    obj = mod.InputType()
    assert obj is not None


def test_JaxprDebugInfo() -> None:
    """Test JaxprDebugInfo."""
    obj = mod.JaxprDebugInfo()
    assert obj is not None


def test_JaxprEqn() -> None:
    """Test JaxprEqn."""
    obj = mod.JaxprEqn()
    assert obj is not None


def test_JaxprPpContext() -> None:
    """Test JaxprPpContext."""
    obj = mod.JaxprPpContext()
    assert obj is not None


def test_JaxprPpSettings() -> None:
    """Test JaxprPpSettings."""
    obj = mod.JaxprPpSettings()
    assert obj is not None


def test_JaxprTypeError() -> None:
    """Test JaxprTypeError."""
    obj = mod.JaxprTypeError()
    assert obj is not None


def test_Literal() -> None:
    """Test Literal."""
    obj = mod.Literal()
    assert obj is not None


def test_MainTrace() -> None:
    """Test MainTrace."""
    obj = mod.MainTrace()
    assert obj is not None


def test_MapPrimitive() -> None:
    """Test MapPrimitive."""
    obj = mod.MapPrimitive()
    assert obj is not None


def test_NameGatheringSubst() -> None:
    """Test NameGatheringSubst."""
    obj = mod.NameGatheringSubst()
    assert obj is not None


def test_NamedShape() -> None:
    """Test NamedShape."""
    obj = mod.NamedShape()
    assert obj is not None


def test_OutDBIdx() -> None:
    """Test OutDBIdx."""
    obj = mod.OutDBIdx()
    assert obj is not None


def test_OutputType() -> None:
    """Test OutputType."""
    obj = mod.OutputType()
    assert obj is not None


def test_ParamDict() -> None:
    """Test ParamDict."""
    obj = mod.ParamDict()
    assert obj is not None


def test_Sublevel() -> None:
    """Test Sublevel."""
    obj = mod.Sublevel()
    assert obj is not None


def test_ThreadLocalState() -> None:
    """Test ThreadLocalState."""
    obj = mod.ThreadLocalState()
    assert obj is not None


def test_TraceStack() -> None:
    """Test TraceStack."""
    obj = mod.TraceStack()
    assert obj is not None


def test_TraceState() -> None:
    """Test TraceState."""
    obj = mod.TraceState()
    assert obj is not None


def test_Tracer() -> None:
    """Test Tracer."""
    obj = mod.Tracer()
    assert obj is not None


def test_UnshapedArray() -> None:
    """Test UnshapedArray."""
    obj = mod.UnshapedArray()
    assert obj is not None


def test_apply_todos() -> None:
    """Test apply_todos."""
    with pytest.raises(NotImplementedError):
        mod.apply_todos()


def test_as_named_shape() -> None:
    """Test as_named_shape."""
    with pytest.raises(NotImplementedError):
        mod.as_named_shape()


def test_axis_frame() -> None:
    """Test axis_frame."""
    with pytest.raises(NotImplementedError):
        mod.axis_frame()


def test_call() -> None:
    """Test call."""
    with pytest.raises(NotImplementedError):
        mod.call()


def test_call_bind_with_continuation() -> None:
    """Test call_bind_with_continuation."""
    with pytest.raises(NotImplementedError):
        mod.call_bind_with_continuation()


def test_call_impl() -> None:
    """Test call_impl."""
    with pytest.raises(NotImplementedError):
        mod.call_impl()


def test_check_eqn() -> None:
    """Test check_eqn."""
    with pytest.raises(NotImplementedError):
        mod.check_eqn()


def test_check_jaxpr() -> None:
    """Test check_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.check_jaxpr()


def test_check_type() -> None:
    """Test check_type."""
    with pytest.raises(NotImplementedError):
        mod.check_type()


def test_check_valid_jaxtype() -> None:
    """Test check_valid_jaxtype."""
    with pytest.raises(NotImplementedError):
        mod.check_valid_jaxtype()


def test_concrete_aval() -> None:
    """Test concrete_aval."""
    with pytest.raises(NotImplementedError):
        mod.concrete_aval()


def test_concrete_or_error() -> None:
    """Test concrete_or_error."""
    with pytest.raises(NotImplementedError):
        mod.concrete_or_error()


def test_concretization_function_error() -> None:
    """Test concretization_function_error."""
    with pytest.raises(NotImplementedError):
        mod.concretization_function_error()


def test_cur_sublevel() -> None:
    """Test cur_sublevel."""
    with pytest.raises(NotImplementedError):
        mod.cur_sublevel()


def test_dedup_referents() -> None:
    """Test dedup_referents."""
    with pytest.raises(NotImplementedError):
        mod.dedup_referents()


def test_do_subst_axis_names_jaxpr() -> None:
    """Test do_subst_axis_names_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.do_subst_axis_names_jaxpr()


def test_ensure_compile_time_eval() -> None:
    """Test ensure_compile_time_eval."""
    with pytest.raises(NotImplementedError):
        mod.ensure_compile_time_eval()


def test_escaped_tracer_error() -> None:
    """Test escaped_tracer_error."""
    with pytest.raises(NotImplementedError):
        mod.escaped_tracer_error()


def test_eval_context() -> None:
    """Test eval_context."""
    with pytest.raises(NotImplementedError):
        mod.eval_context()


def test_eval_jaxpr() -> None:
    """Test eval_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.eval_jaxpr()


def test_extend_axis_env() -> None:
    """Test extend_axis_env."""
    with pytest.raises(NotImplementedError):
        mod.extend_axis_env()


def test_extend_axis_env_nd() -> None:
    """Test extend_axis_env_nd."""
    with pytest.raises(NotImplementedError):
        mod.extend_axis_env_nd()


def test_find_top_trace() -> None:
    """Test find_top_trace."""
    with pytest.raises(NotImplementedError):
        mod.find_top_trace()


def test_full_lower() -> None:
    """Test full_lower."""
    with pytest.raises(NotImplementedError):
        mod.full_lower()


def test_gensym() -> None:
    """Test gensym."""
    with pytest.raises(NotImplementedError):
        mod.gensym()


def test_get_aval() -> None:
    """Test get_aval."""
    with pytest.raises(NotImplementedError):
        mod.get_aval()


def test_get_referent() -> None:
    """Test get_referent."""
    with pytest.raises(NotImplementedError):
        mod.get_referent()


def test_is_constant_dim() -> None:
    """Test is_constant_dim."""
    with pytest.raises(NotImplementedError):
        mod.is_constant_dim()


def test_is_constant_shape() -> None:
    """Test is_constant_shape."""
    with pytest.raises(NotImplementedError):
        mod.is_constant_shape()


def test_jaxpr_as_fun() -> None:
    """Test jaxpr_as_fun."""
    with pytest.raises(NotImplementedError):
        mod.jaxpr_as_fun()


def test_jaxpr_uses_outfeed() -> None:
    """Test jaxpr_uses_outfeed."""
    with pytest.raises(NotImplementedError):
        mod.jaxpr_uses_outfeed()


def test_jaxprs_in_params() -> None:
    """Test jaxprs_in_params."""
    with pytest.raises(NotImplementedError):
        mod.jaxprs_in_params()


def test_join_effects() -> None:
    """Test join_effects."""
    with pytest.raises(NotImplementedError):
        mod.join_effects()


def test_join_named_shapes() -> None:
    """Test join_named_shapes."""
    with pytest.raises(NotImplementedError):
        mod.join_named_shapes()


def test_lattice_join() -> None:
    """Test lattice_join."""
    with pytest.raises(NotImplementedError):
        mod.lattice_join()


def test_leaked_tracer_error() -> None:
    """Test leaked_tracer_error."""
    with pytest.raises(NotImplementedError):
        mod.leaked_tracer_error()


def test_map_bind() -> None:
    """Test map_bind."""
    with pytest.raises(NotImplementedError):
        mod.map_bind()


def test_map_bind_with_continuation() -> None:
    """Test map_bind_with_continuation."""
    with pytest.raises(NotImplementedError):
        mod.map_bind_with_continuation()


def test_mapped_aval() -> None:
    """Test mapped_aval."""
    with pytest.raises(NotImplementedError):
        mod.mapped_aval()


def test_max_dim() -> None:
    """Test max_dim."""
    with pytest.raises(NotImplementedError):
        mod.max_dim()


def test_maybe_find_leaked_tracers() -> None:
    """Test maybe_find_leaked_tracers."""
    with pytest.raises(NotImplementedError):
        mod.maybe_find_leaked_tracers()


def test_min_dim() -> None:
    """Test min_dim."""
    with pytest.raises(NotImplementedError):
        mod.min_dim()


def test_new_base_main() -> None:
    """Test new_base_main."""
    with pytest.raises(NotImplementedError):
        mod.new_base_main()


def test_new_jaxpr_eqn() -> None:
    """Test new_jaxpr_eqn."""
    with pytest.raises(NotImplementedError):
        mod.new_jaxpr_eqn()


def test_new_main() -> None:
    """Test new_main."""
    with pytest.raises(NotImplementedError):
        mod.new_main()


def test_new_sublevel() -> None:
    """Test new_sublevel."""
    with pytest.raises(NotImplementedError):
        mod.new_sublevel()


def test_primal_dtype_to_tangent_dtype() -> None:
    """Test primal_dtype_to_tangent_dtype."""
    with pytest.raises(NotImplementedError):
        mod.primal_dtype_to_tangent_dtype()


def test_primitive_uses_outfeed() -> None:
    """Test primitive_uses_outfeed."""
    with pytest.raises(NotImplementedError):
        mod.primitive_uses_outfeed()


def test_process_env_traces_call() -> None:
    """Test process_env_traces_call."""
    with pytest.raises(NotImplementedError):
        mod.process_env_traces_call()


def test_process_env_traces_map() -> None:
    """Test process_env_traces_map."""
    with pytest.raises(NotImplementedError):
        mod.process_env_traces_map()


def test_raise_as_much_as_possible() -> None:
    """Test raise_as_much_as_possible."""
    with pytest.raises(NotImplementedError):
        mod.raise_as_much_as_possible()


def test_raise_to_shaped() -> None:
    """Test raise_to_shaped."""
    with pytest.raises(NotImplementedError):
        mod.raise_to_shaped()


def test_reset_trace_state() -> None:
    """Test reset_trace_state."""
    with pytest.raises(NotImplementedError):
        mod.reset_trace_state()


def test_stash_axis_env() -> None:
    """Test stash_axis_env."""
    with pytest.raises(NotImplementedError):
        mod.stash_axis_env()


def test_str_eqn_compact() -> None:
    """Test str_eqn_compact."""
    with pytest.raises(NotImplementedError):
        mod.str_eqn_compact()


def test_subjaxprs() -> None:
    """Test subjaxprs."""
    with pytest.raises(NotImplementedError):
        mod.subjaxprs()


def test_subst_axis_names() -> None:
    """Test subst_axis_names."""
    with pytest.raises(NotImplementedError):
        mod.subst_axis_names()


def test_subst_axis_names_eqn() -> None:
    """Test subst_axis_names_eqn."""
    with pytest.raises(NotImplementedError):
        mod.subst_axis_names_eqn()


def test_subst_axis_names_jaxpr() -> None:
    """Test subst_axis_names_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.subst_axis_names_jaxpr()


def test_subst_axis_names_var() -> None:
    """Test subst_axis_names_var."""
    with pytest.raises(NotImplementedError):
        mod.subst_axis_names_var()


def test_substitute_vars_in_output_ty() -> None:
    """Test substitute_vars_in_output_ty."""
    with pytest.raises(NotImplementedError):
        mod.substitute_vars_in_output_ty()


def test_trace_state_clean() -> None:
    """Test trace_state_clean."""
    with pytest.raises(NotImplementedError):
        mod.trace_state_clean()


def test_traverse_jaxpr_params() -> None:
    """Test traverse_jaxpr_params."""
    with pytest.raises(NotImplementedError):
        mod.traverse_jaxpr_params()


def test_typecheck() -> None:
    """Test typecheck."""
    with pytest.raises(NotImplementedError):
        mod.typecheck()


def test_typecompat() -> None:
    """Test typecompat."""
    with pytest.raises(NotImplementedError):
        mod.typecompat()


def test_typematch() -> None:
    """Test typematch."""
    with pytest.raises(NotImplementedError):
        mod.typematch()


def test_unmapped_aval() -> None:
    """Test unmapped_aval."""
    with pytest.raises(NotImplementedError):
        mod.unmapped_aval()


def test_used_axis_names() -> None:
    """Test used_axis_names."""
    with pytest.raises(NotImplementedError):
        mod.used_axis_names()


def test_used_axis_names_jaxpr() -> None:
    """Test used_axis_names_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.used_axis_names_jaxpr()


def test_valid_jaxtype() -> None:
    """Test valid_jaxtype."""
    with pytest.raises(NotImplementedError):
        mod.valid_jaxtype()
