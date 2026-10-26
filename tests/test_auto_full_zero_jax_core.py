"""Tests for zero_jax.core."""

from __future__ import annotations

import pytest

import zero_jax.core as mod


def test_class_AbstractToken() -> None:
    """Test AbstractToken."""
    obj = mod.AbstractToken()
    assert isinstance(obj, mod.AbstractToken)


def test_class_AbstractValue() -> None:
    """Test AbstractValue."""
    obj = mod.AbstractValue()
    assert isinstance(obj, mod.AbstractValue)


def test_Atom() -> None:
    """Test Atom."""
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError):
        mod.Atom()


def test_AxisSize() -> None:
    """Test AxisSize."""
    with pytest.raises(AttributeError):
        mod.AxisSize()


def test_class_CallPrimitive() -> None:
    """Test CallPrimitive."""
    obj = mod.CallPrimitive()
    assert isinstance(obj, mod.CallPrimitive)


def test_class_ClosedJaxpr() -> None:
    """Test ClosedJaxpr."""
    obj = mod.ClosedJaxpr()
    assert obj.consts == []


def test_class_ConcreteArray() -> None:
    """Test ConcreteArray."""
    obj = mod.ConcreteArray(val=42)
    assert obj.val == 42


def test_class_ConcretizationTypeError() -> None:
    """Test ConcretizationTypeError."""
    obj = mod.ConcretizationTypeError()
    assert isinstance(obj, mod.ConcretizationTypeError)


def test_class_DShapedArray() -> None:
    """Test DShapedArray."""
    obj = mod.DShapedArray()
    assert isinstance(obj, mod.DShapedArray)


def test_Dict() -> None:
    """Test Dict."""
    assert "Dict" not in dir(mod)


def test_class_DropVar() -> None:
    """Test DropVar."""
    obj = mod.DropVar()
    assert isinstance(obj, mod.DropVar)


def test_class_Effect() -> None:
    """Test Effect."""
    obj = mod.Effect()
    assert isinstance(obj, mod.Effect)


def test_class_Effects() -> None:
    """Test Effects."""
    obj = mod.Effects()
    assert isinstance(obj, mod.Effects)


def test_class_EvalTrace() -> None:
    """Test EvalTrace."""
    obj = mod.EvalTrace()
    assert isinstance(obj, mod.EvalTrace)


def test_class_InDBIdx() -> None:
    """Test InDBIdx."""
    obj = mod.InDBIdx()
    assert isinstance(obj, mod.InDBIdx)


def test_class_InconclusiveDimensionOperation() -> None:
    """Test InconclusiveDimensionOperation."""
    obj = mod.InconclusiveDimensionOperation()
    assert isinstance(obj, mod.InconclusiveDimensionOperation)


def test_class_InputType() -> None:
    """Test InputType."""
    obj = mod.InputType()
    assert isinstance(obj, mod.InputType)


def test_Iterable() -> None:
    """Test Iterable."""
    assert mod.Iterable is not None


def test_class_Jaxpr() -> None:
    """Test Jaxpr."""
    obj = mod.Jaxpr()
    assert obj.eqns == []


def test_class_JaxprDebugInfo() -> None:
    """Test JaxprDebugInfo."""
    obj = mod.JaxprDebugInfo()
    assert isinstance(obj, mod.JaxprDebugInfo)


def test_class_JaxprEqn() -> None:
    """Test JaxprEqn."""
    obj = mod.JaxprEqn()
    assert isinstance(obj, mod.JaxprEqn)


def test_class_JaxprPpContext() -> None:
    """Test JaxprPpContext."""
    obj = mod.JaxprPpContext()
    assert isinstance(obj, mod.JaxprPpContext)


def test_class_JaxprPpSettings() -> None:
    """Test JaxprPpSettings."""
    obj = mod.JaxprPpSettings()
    assert isinstance(obj, mod.JaxprPpSettings)


def test_class_JaxprTypeError() -> None:
    """Test JaxprTypeError."""
    obj = mod.JaxprTypeError()
    assert isinstance(obj, mod.JaxprTypeError)


def test_List() -> None:
    """Test List."""
    assert "List" not in dir(mod)


def test_class_Literal() -> None:
    """Test Literal."""
    obj = mod.Literal(val=3.5)
    assert obj.val == 3.5


def test_class_MainTrace() -> None:
    """Test MainTrace."""
    obj = mod.MainTrace()
    assert isinstance(obj, mod.MainTrace)


def test_class_MapPrimitive() -> None:
    """Test MapPrimitive."""
    obj = mod.MapPrimitive()
    assert isinstance(obj, mod.MapPrimitive)


def test_class_NameGatheringSubst() -> None:
    """Test NameGatheringSubst."""
    obj = mod.NameGatheringSubst()
    assert isinstance(obj, mod.NameGatheringSubst)


def test_class_NamedShape() -> None:
    """Test NamedShape."""
    obj = mod.NamedShape()
    assert isinstance(obj, mod.NamedShape)


def test_Optional() -> None:
    """Test Optional."""
    assert "Optional" not in dir(mod)


def test_class_OutDBIdx() -> None:
    """Test OutDBIdx."""
    obj = mod.OutDBIdx()
    assert isinstance(obj, mod.OutDBIdx)


def test_class_OutputType() -> None:
    """Test OutputType."""
    obj = mod.OutputType()
    assert isinstance(obj, mod.OutputType)


def test_class_ParamDict() -> None:
    """Test ParamDict."""
    obj = mod.ParamDict()
    assert isinstance(obj, mod.ParamDict)


def test_class_Primitive() -> None:
    """Test Primitive."""
    obj = mod.Primitive(name="sin")
    assert obj.name == "sin"


def test_Sequence() -> None:
    """Test Sequence."""
    assert mod.Sequence is not None


def test_Set() -> None:
    """Test Set."""
    assert "Set" not in dir(mod)


def test_class_ShapedArray() -> None:
    """Test ShapedArray."""
    obj = mod.ShapedArray(shape=(2, 3), dtype="float32")
    assert obj.shape == (2, 3)
    assert obj.dtype == "float32"


def test_class_Sublevel() -> None:
    """Test Sublevel."""
    obj = mod.Sublevel()
    assert isinstance(obj, mod.Sublevel)


def test_class_TRACER_LEAK_DEBUGGER_WARNING() -> None:
    """Test TRACER_LEAK_DEBUGGER_WARNING."""
    obj = mod.TRACER_LEAK_DEBUGGER_WARNING()
    assert isinstance(obj, mod.TRACER_LEAK_DEBUGGER_WARNING)


def test_class_ThreadLocalState() -> None:
    """Test ThreadLocalState."""
    obj = mod.ThreadLocalState()
    assert isinstance(obj, mod.ThreadLocalState)


def test_class_Token() -> None:
    """Test Token."""
    obj = mod.Token(name="tok")
    assert obj.name == "tok"


def test_class_Trace() -> None:
    """Test Trace."""
    obj = mod.Trace()
    assert isinstance(obj, mod.Trace)


def test_class_TraceStack() -> None:
    """Test TraceStack."""
    obj = mod.TraceStack()
    assert isinstance(obj, mod.TraceStack)


def test_class_TraceState() -> None:
    """Test TraceState."""
    obj = mod.TraceState()
    assert isinstance(obj, mod.TraceState)


def test_class_Tracer() -> None:
    """Test Tracer."""
    obj = mod.Tracer()
    assert isinstance(obj, mod.Tracer)


def test_Tuple() -> None:
    """Test Tuple."""
    assert "Tuple" not in dir(mod)


def test_class_UnshapedArray() -> None:
    """Test UnshapedArray."""
    obj = mod.UnshapedArray()
    assert isinstance(obj, mod.UnshapedArray)


def test_Value() -> None:
    """Test Value."""
    assert mod.Value is not None


def test_class_Var() -> None:
    """Test Var."""
    obj = mod.Var(aval="float32")
    assert obj.aval == "float32"


def test_abstract_token() -> None:
    """Test abstract_token."""
    assert mod.abstract_token() is None


def test_apply_todos() -> None:
    """Test apply_todos."""
    with pytest.raises(AttributeError):
        mod.apply_todos()


def test_as_named_shape() -> None:
    """Test as_named_shape."""
    with pytest.raises(AttributeError):
        mod.as_named_shape()


def test_aval_mapping_handlers() -> None:
    """Test aval_mapping_handlers."""
    assert mod.aval_mapping_handlers() is None


def test_axis_frame() -> None:
    """Test axis_frame."""
    with pytest.raises(AttributeError):
        mod.axis_frame()


def test_call() -> None:
    """Test call."""
    with pytest.raises(AttributeError):
        mod.call()


def test_call_bind_with_continuation() -> None:
    """Test call_bind_with_continuation."""
    with pytest.raises(AttributeError):
        mod.call_bind_with_continuation()


def test_call_impl() -> None:
    """Test call_impl."""
    with pytest.raises(AttributeError):
        mod.call_impl()


def test_call_p() -> None:
    """Test call_p."""
    assert mod.call_p() is None


def test_check_eqn() -> None:
    """Test check_eqn."""
    with pytest.raises(AttributeError):
        mod.check_eqn()


def test_check_jaxpr() -> None:
    """Test check_jaxpr."""
    with pytest.raises(AttributeError):
        mod.check_jaxpr()


def test_check_type() -> None:
    """Test check_type."""
    with pytest.raises(AttributeError):
        mod.check_type()


def test_check_valid_jaxtype() -> None:
    """Test check_valid_jaxtype."""
    with pytest.raises(AttributeError):
        mod.check_valid_jaxtype()


def test_closed_call_p() -> None:
    """Test closed_call_p."""
    assert mod.closed_call_p() is None


def test_concrete_aval() -> None:
    """Test concrete_aval."""
    with pytest.raises(AttributeError):
        mod.concrete_aval()


def test_concrete_or_error() -> None:
    """Test concrete_or_error."""
    with pytest.raises(AttributeError):
        mod.concrete_or_error()


def test_concretization_function_error() -> None:
    """Test concretization_function_error."""
    with pytest.raises(AttributeError):
        mod.concretization_function_error()


def test_cur_sublevel() -> None:
    """Test cur_sublevel."""
    with pytest.raises(AttributeError):
        mod.cur_sublevel()


def test_custom_typechecks() -> None:
    """Test custom_typechecks."""
    assert mod.custom_typechecks() is None


def test_dataclass() -> None:
    """Test dataclass."""
    assert mod.dataclass is not None


def test_dedup_referents() -> None:
    """Test dedup_referents."""
    with pytest.raises(AttributeError):
        mod.dedup_referents()


def test_do_subst_axis_names_jaxpr() -> None:
    """Test do_subst_axis_names_jaxpr."""
    with pytest.raises(AttributeError):
        mod.do_subst_axis_names_jaxpr()


def test_ensure_compile_time_eval() -> None:
    """Test ensure_compile_time_eval."""
    with pytest.raises(AttributeError):
        mod.ensure_compile_time_eval()


def test_escaped_tracer_error() -> None:
    """Test escaped_tracer_error."""
    with pytest.raises(AttributeError):
        mod.escaped_tracer_error()


def test_eval_context() -> None:
    """Test eval_context."""
    with pytest.raises(AttributeError):
        mod.eval_context()


def test_eval_jaxpr() -> None:
    """Test eval_jaxpr."""
    with pytest.raises(AttributeError):
        mod.eval_jaxpr()


def test_extend_axis_env() -> None:
    """Test extend_axis_env."""
    with pytest.raises(AttributeError):
        mod.extend_axis_env()


def test_extend_axis_env_nd() -> None:
    """Test extend_axis_env_nd."""
    with pytest.raises(AttributeError):
        mod.extend_axis_env_nd()


def test_field() -> None:
    """Test field."""
    assert mod.field is not None


def test_find_top_trace() -> None:
    """Test find_top_trace."""
    with pytest.raises(AttributeError):
        mod.find_top_trace()


def test_full_lower() -> None:
    """Test full_lower."""
    with pytest.raises(AttributeError):
        mod.full_lower()


def test_gensym() -> None:
    """Test gensym."""
    with pytest.raises(AttributeError):
        mod.gensym()


def test_get_aval() -> None:
    """Test get_aval."""
    with pytest.raises(AttributeError):
        mod.get_aval()


def test_get_referent() -> None:
    """Test get_referent."""
    with pytest.raises(AttributeError):
        mod.get_referent()


def test_is_constant_dim() -> None:
    """Test is_constant_dim."""
    with pytest.raises(AttributeError):
        mod.is_constant_dim()


def test_is_constant_shape() -> None:
    """Test is_constant_shape."""
    with pytest.raises(AttributeError):
        mod.is_constant_shape()


def test_jaxpr_as_fun() -> None:
    """Test jaxpr_as_fun."""
    with pytest.raises(AttributeError):
        mod.jaxpr_as_fun()


def test_jaxpr_uses_outfeed() -> None:
    """Test jaxpr_uses_outfeed."""
    with pytest.raises(AttributeError):
        mod.jaxpr_uses_outfeed()


def test_jaxprs_in_params() -> None:
    """Test jaxprs_in_params."""
    with pytest.raises(AttributeError):
        mod.jaxprs_in_params()


def test_join_effects() -> None:
    """Test join_effects."""
    with pytest.raises(AttributeError):
        mod.join_effects()


def test_join_named_shapes() -> None:
    """Test join_named_shapes."""
    with pytest.raises(AttributeError):
        mod.join_named_shapes()


def test_lattice_join() -> None:
    """Test lattice_join."""
    with pytest.raises(AttributeError):
        mod.lattice_join()


def test_leaked_tracer_error() -> None:
    """Test leaked_tracer_error."""
    with pytest.raises(AttributeError):
        mod.leaked_tracer_error()


def test_literalable_types() -> None:
    """Test literalable_types."""
    assert mod.literalable_types() is None


def test_map_bind() -> None:
    """Test map_bind."""
    with pytest.raises(AttributeError):
        mod.map_bind()


def test_map_bind_with_continuation() -> None:
    """Test map_bind_with_continuation."""
    with pytest.raises(AttributeError):
        mod.map_bind_with_continuation()


def test_mapped_aval() -> None:
    """Test mapped_aval."""
    with pytest.raises(AttributeError):
        mod.mapped_aval()


def test_max_dim() -> None:
    """Test max_dim."""
    with pytest.raises(AttributeError):
        mod.max_dim()


def test_maybe_find_leaked_tracers() -> None:
    """Test maybe_find_leaked_tracers."""
    with pytest.raises(AttributeError):
        mod.maybe_find_leaked_tracers()


def test_min_dim() -> None:
    """Test min_dim."""
    with pytest.raises(AttributeError):
        mod.min_dim()


def test_new_base_main() -> None:
    """Test new_base_main."""
    with pytest.raises(AttributeError):
        mod.new_base_main()


def test_new_jaxpr_eqn() -> None:
    """Test new_jaxpr_eqn."""
    with pytest.raises(AttributeError):
        mod.new_jaxpr_eqn()


def test_new_main() -> None:
    """Test new_main."""
    with pytest.raises(AttributeError):
        mod.new_main()


def test_new_sublevel() -> None:
    """Test new_sublevel."""
    with pytest.raises(AttributeError):
        mod.new_sublevel()


def test_no_axis_name() -> None:
    """Test no_axis_name."""
    assert mod.no_axis_name() is None


def test_no_effects() -> None:
    """Test no_effects."""
    assert mod.no_effects() is None


def test_outfeed_primitives() -> None:
    """Test outfeed_primitives."""
    assert mod.outfeed_primitives() is None


def test_primal_dtype_to_tangent_dtype() -> None:
    """Test primal_dtype_to_tangent_dtype."""
    with pytest.raises(AttributeError):
        mod.primal_dtype_to_tangent_dtype()


def test_primitive_uses_outfeed() -> None:
    """Test primitive_uses_outfeed."""
    with pytest.raises(AttributeError):
        mod.primitive_uses_outfeed()


def test_process_env_traces_call() -> None:
    """Test process_env_traces_call."""
    with pytest.raises(AttributeError):
        mod.process_env_traces_call()


def test_process_env_traces_map() -> None:
    """Test process_env_traces_map."""
    with pytest.raises(AttributeError):
        mod.process_env_traces_map()


def test_pytype_aval_mappings() -> None:
    """Test pytype_aval_mappings."""
    assert mod.pytype_aval_mappings() is None


def test_raise_as_much_as_possible() -> None:
    """Test raise_as_much_as_possible."""
    with pytest.raises(AttributeError):
        mod.raise_as_much_as_possible()


def test_raise_to_shaped() -> None:
    """Test raise_to_shaped."""
    with pytest.raises(AttributeError):
        mod.raise_to_shaped()


def test_raise_to_shaped_mappings() -> None:
    """Test raise_to_shaped_mappings."""
    assert mod.raise_to_shaped_mappings() is None


def test_reset_trace_state() -> None:
    """Test reset_trace_state."""
    with pytest.raises(AttributeError):
        mod.reset_trace_state()


def test_stash_axis_env() -> None:
    """Test stash_axis_env."""
    with pytest.raises(AttributeError):
        mod.stash_axis_env()


def test_str_eqn_compact() -> None:
    """Test str_eqn_compact."""
    with pytest.raises(AttributeError):
        mod.str_eqn_compact()


def test_subjaxprs() -> None:
    """Test subjaxprs."""
    with pytest.raises(AttributeError):
        mod.subjaxprs()


def test_subst_axis_names() -> None:
    """Test subst_axis_names."""
    with pytest.raises(AttributeError):
        mod.subst_axis_names()


def test_subst_axis_names_eqn() -> None:
    """Test subst_axis_names_eqn."""
    with pytest.raises(AttributeError):
        mod.subst_axis_names_eqn()


def test_subst_axis_names_jaxpr() -> None:
    """Test subst_axis_names_jaxpr."""
    with pytest.raises(AttributeError):
        mod.subst_axis_names_jaxpr()


def test_subst_axis_names_var() -> None:
    """Test subst_axis_names_var."""
    with pytest.raises(AttributeError):
        mod.subst_axis_names_var()


def test_substitute_vars_in_output_ty() -> None:
    """Test substitute_vars_in_output_ty."""
    with pytest.raises(AttributeError):
        mod.substitute_vars_in_output_ty()


def test_thread_local_state() -> None:
    """Test thread_local_state."""
    assert mod.thread_local_state() is None


def test_trace_state_clean() -> None:
    """Test trace_state_clean."""
    with pytest.raises(AttributeError):
        mod.trace_state_clean()


def test_traverse_jaxpr_params() -> None:
    """Test traverse_jaxpr_params."""
    with pytest.raises(AttributeError):
        mod.traverse_jaxpr_params()


def test_typecheck() -> None:
    """Test typecheck."""
    with pytest.raises(AttributeError):
        mod.typecheck()


def test_typecompat() -> None:
    """Test typecompat."""
    with pytest.raises(AttributeError):
        mod.typecompat()


def test_typematch() -> None:
    """Test typematch."""
    with pytest.raises(AttributeError):
        mod.typematch()


def test_unmapped_aval() -> None:
    """Test unmapped_aval."""
    with pytest.raises(AttributeError):
        mod.unmapped_aval()


def test_used_axis_names() -> None:
    """Test used_axis_names."""
    with pytest.raises(AttributeError):
        mod.used_axis_names()


def test_used_axis_names_jaxpr() -> None:
    """Test used_axis_names_jaxpr."""
    with pytest.raises(AttributeError):
        mod.used_axis_names_jaxpr()


def test_valid_jaxtype() -> None:
    """Test valid_jaxtype."""
    with pytest.raises(AttributeError):
        mod.valid_jaxtype()
