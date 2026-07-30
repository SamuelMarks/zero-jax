"""Tests for zero_jax module."""

from unittest.mock import patch

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
    with patch("zero_jax._compiler_proxy_ops.Atom", create=True) as mock_op:
        mod.Atom()
        mock_op.assert_called_once_with()


def test_AxisSize() -> None:
    """Test AxisSize."""
    with patch("zero_jax._compiler_proxy_ops.AxisSize", create=True) as mock_op:
        mod.AxisSize()
        mock_op.assert_called_once_with()


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
    with patch("zero_jax._compiler_proxy_ops.apply_todos", create=True) as mock_op:
        mod.apply_todos()
        mock_op.assert_called_once_with()


def test_as_named_shape() -> None:
    """Test as_named_shape."""
    with patch("zero_jax._compiler_proxy_ops.as_named_shape", create=True) as mock_op:
        mod.as_named_shape()
        mock_op.assert_called_once_with()


def test_axis_frame() -> None:
    """Test axis_frame."""
    with patch("zero_jax._compiler_proxy_ops.axis_frame", create=True) as mock_op:
        mod.axis_frame()
        mock_op.assert_called_once_with()


def test_call() -> None:
    """Test call."""
    with patch("zero_jax._compiler_proxy_ops.call", create=True) as mock_op:
        mod.call()
        mock_op.assert_called_once_with()


def test_call_bind_with_continuation() -> None:
    """Test call_bind_with_continuation."""
    with patch(
        "zero_jax._compiler_proxy_ops.call_bind_with_continuation", create=True
    ) as mock_op:
        mod.call_bind_with_continuation()
        mock_op.assert_called_once_with()


def test_call_impl() -> None:
    """Test call_impl."""
    with patch("zero_jax._compiler_proxy_ops.call_impl", create=True) as mock_op:
        mod.call_impl()
        mock_op.assert_called_once_with()


def test_check_eqn() -> None:
    """Test check_eqn."""
    with patch("zero_jax._compiler_proxy_ops.check_eqn", create=True) as mock_op:
        mod.check_eqn()
        mock_op.assert_called_once_with()


def test_check_jaxpr() -> None:
    """Test check_jaxpr."""
    with patch("zero_jax._compiler_proxy_ops.check_jaxpr", create=True) as mock_op:
        mod.check_jaxpr()
        mock_op.assert_called_once_with()


def test_check_type() -> None:
    """Test check_type."""
    with patch("zero_jax._compiler_proxy_ops.check_type", create=True) as mock_op:
        mod.check_type()
        mock_op.assert_called_once_with()


def test_check_valid_jaxtype() -> None:
    """Test check_valid_jaxtype."""
    with patch(
        "zero_jax._compiler_proxy_ops.check_valid_jaxtype", create=True
    ) as mock_op:
        mod.check_valid_jaxtype()
        mock_op.assert_called_once_with()


def test_concrete_aval() -> None:
    """Test concrete_aval."""
    with patch("zero_jax._compiler_proxy_ops.concrete_aval", create=True) as mock_op:
        mod.concrete_aval()
        mock_op.assert_called_once_with()


def test_concrete_or_error() -> None:
    """Test concrete_or_error."""
    with patch(
        "zero_jax._compiler_proxy_ops.concrete_or_error", create=True
    ) as mock_op:
        mod.concrete_or_error()
        mock_op.assert_called_once_with()


def test_concretization_function_error() -> None:
    """Test concretization_function_error."""
    with patch(
        "zero_jax._compiler_proxy_ops.concretization_function_error", create=True
    ) as mock_op:
        mod.concretization_function_error()
        mock_op.assert_called_once_with()


def test_cur_sublevel() -> None:
    """Test cur_sublevel."""
    with patch("zero_jax._compiler_proxy_ops.cur_sublevel", create=True) as mock_op:
        mod.cur_sublevel()
        mock_op.assert_called_once_with()


def test_dedup_referents() -> None:
    """Test dedup_referents."""
    with patch("zero_jax._compiler_proxy_ops.dedup_referents", create=True) as mock_op:
        mod.dedup_referents()
        mock_op.assert_called_once_with()


def test_do_subst_axis_names_jaxpr() -> None:
    """Test do_subst_axis_names_jaxpr."""
    with patch(
        "zero_jax._compiler_proxy_ops.do_subst_axis_names_jaxpr", create=True
    ) as mock_op:
        mod.do_subst_axis_names_jaxpr()
        mock_op.assert_called_once_with()


def test_ensure_compile_time_eval() -> None:
    """Test ensure_compile_time_eval."""
    with patch(
        "zero_jax._compiler_proxy_ops.ensure_compile_time_eval", create=True
    ) as mock_op:
        mod.ensure_compile_time_eval()
        mock_op.assert_called_once_with()


def test_escaped_tracer_error() -> None:
    """Test escaped_tracer_error."""
    with patch(
        "zero_jax._compiler_proxy_ops.escaped_tracer_error", create=True
    ) as mock_op:
        mod.escaped_tracer_error()
        mock_op.assert_called_once_with()


def test_eval_context() -> None:
    """Test eval_context."""
    with patch("zero_jax._compiler_proxy_ops.eval_context", create=True) as mock_op:
        mod.eval_context()
        mock_op.assert_called_once_with()


def test_eval_jaxpr() -> None:
    """Test eval_jaxpr."""
    with patch("zero_jax._compiler_proxy_ops.eval_jaxpr", create=True) as mock_op:
        mod.eval_jaxpr()
        mock_op.assert_called_once_with()


def test_extend_axis_env() -> None:
    """Test extend_axis_env."""
    with patch("zero_jax._compiler_proxy_ops.extend_axis_env", create=True) as mock_op:
        mod.extend_axis_env()
        mock_op.assert_called_once_with()


def test_extend_axis_env_nd() -> None:
    """Test extend_axis_env_nd."""
    with patch(
        "zero_jax._compiler_proxy_ops.extend_axis_env_nd", create=True
    ) as mock_op:
        mod.extend_axis_env_nd()
        mock_op.assert_called_once_with()


def test_find_top_trace() -> None:
    """Test find_top_trace."""
    with patch("zero_jax._compiler_proxy_ops.find_top_trace", create=True) as mock_op:
        mod.find_top_trace()
        mock_op.assert_called_once_with()


def test_full_lower() -> None:
    """Test full_lower."""
    with patch("zero_jax._compiler_proxy_ops.full_lower", create=True) as mock_op:
        mod.full_lower()
        mock_op.assert_called_once_with()


def test_gensym() -> None:
    """Test gensym."""
    with patch("zero_jax._compiler_proxy_ops.gensym", create=True) as mock_op:
        mod.gensym()
        mock_op.assert_called_once_with()


def test_get_aval() -> None:
    """Test get_aval."""
    with patch("zero_jax._compiler_proxy_ops.get_aval", create=True) as mock_op:
        mod.get_aval()
        mock_op.assert_called_once_with()


def test_get_referent() -> None:
    """Test get_referent."""
    with patch("zero_jax._compiler_proxy_ops.get_referent", create=True) as mock_op:
        mod.get_referent()
        mock_op.assert_called_once_with()


def test_is_constant_dim() -> None:
    """Test is_constant_dim."""
    with patch("zero_jax._compiler_proxy_ops.is_constant_dim", create=True) as mock_op:
        mod.is_constant_dim()
        mock_op.assert_called_once_with()


def test_is_constant_shape() -> None:
    """Test is_constant_shape."""
    with patch(
        "zero_jax._compiler_proxy_ops.is_constant_shape", create=True
    ) as mock_op:
        mod.is_constant_shape()
        mock_op.assert_called_once_with()


def test_jaxpr_as_fun() -> None:
    """Test jaxpr_as_fun."""
    with patch("zero_jax._compiler_proxy_ops.jaxpr_as_fun", create=True) as mock_op:
        mod.jaxpr_as_fun()
        mock_op.assert_called_once_with()


def test_jaxpr_uses_outfeed() -> None:
    """Test jaxpr_uses_outfeed."""
    with patch(
        "zero_jax._compiler_proxy_ops.jaxpr_uses_outfeed", create=True
    ) as mock_op:
        mod.jaxpr_uses_outfeed()
        mock_op.assert_called_once_with()


def test_jaxprs_in_params() -> None:
    """Test jaxprs_in_params."""
    with patch("zero_jax._compiler_proxy_ops.jaxprs_in_params", create=True) as mock_op:
        mod.jaxprs_in_params()
        mock_op.assert_called_once_with()


def test_join_effects() -> None:
    """Test join_effects."""
    with patch("zero_jax._compiler_proxy_ops.join_effects", create=True) as mock_op:
        mod.join_effects()
        mock_op.assert_called_once_with()


def test_join_named_shapes() -> None:
    """Test join_named_shapes."""
    with patch(
        "zero_jax._compiler_proxy_ops.join_named_shapes", create=True
    ) as mock_op:
        mod.join_named_shapes()
        mock_op.assert_called_once_with()


def test_lattice_join() -> None:
    """Test lattice_join."""
    with patch("zero_jax._compiler_proxy_ops.lattice_join", create=True) as mock_op:
        mod.lattice_join()
        mock_op.assert_called_once_with()


def test_leaked_tracer_error() -> None:
    """Test leaked_tracer_error."""
    with patch(
        "zero_jax._compiler_proxy_ops.leaked_tracer_error", create=True
    ) as mock_op:
        mod.leaked_tracer_error()
        mock_op.assert_called_once_with()


def test_map_bind() -> None:
    """Test map_bind."""
    with patch("zero_jax._compiler_proxy_ops.map_bind", create=True) as mock_op:
        mod.map_bind()
        mock_op.assert_called_once_with()


def test_map_bind_with_continuation() -> None:
    """Test map_bind_with_continuation."""
    with patch(
        "zero_jax._compiler_proxy_ops.map_bind_with_continuation", create=True
    ) as mock_op:
        mod.map_bind_with_continuation()
        mock_op.assert_called_once_with()


def test_mapped_aval() -> None:
    """Test mapped_aval."""
    with patch("zero_jax._compiler_proxy_ops.mapped_aval", create=True) as mock_op:
        mod.mapped_aval()
        mock_op.assert_called_once_with()


def test_max_dim() -> None:
    """Test max_dim."""
    with patch("zero_jax._compiler_proxy_ops.max_dim", create=True) as mock_op:
        mod.max_dim()
        mock_op.assert_called_once_with()


def test_maybe_find_leaked_tracers() -> None:
    """Test maybe_find_leaked_tracers."""
    with patch(
        "zero_jax._compiler_proxy_ops.maybe_find_leaked_tracers", create=True
    ) as mock_op:
        mod.maybe_find_leaked_tracers()
        mock_op.assert_called_once_with()


def test_min_dim() -> None:
    """Test min_dim."""
    with patch("zero_jax._compiler_proxy_ops.min_dim", create=True) as mock_op:
        mod.min_dim()
        mock_op.assert_called_once_with()


def test_new_base_main() -> None:
    """Test new_base_main."""
    with patch("zero_jax._compiler_proxy_ops.new_base_main", create=True) as mock_op:
        mod.new_base_main()
        mock_op.assert_called_once_with()


def test_new_jaxpr_eqn() -> None:
    """Test new_jaxpr_eqn."""
    with patch("zero_jax._compiler_proxy_ops.new_jaxpr_eqn", create=True) as mock_op:
        mod.new_jaxpr_eqn()
        mock_op.assert_called_once_with()


def test_new_main() -> None:
    """Test new_main."""
    with patch("zero_jax._compiler_proxy_ops.new_main", create=True) as mock_op:
        mod.new_main()
        mock_op.assert_called_once_with()


def test_new_sublevel() -> None:
    """Test new_sublevel."""
    with patch("zero_jax._compiler_proxy_ops.new_sublevel", create=True) as mock_op:
        mod.new_sublevel()
        mock_op.assert_called_once_with()


def test_primal_dtype_to_tangent_dtype() -> None:
    """Test primal_dtype_to_tangent_dtype."""
    with patch(
        "zero_jax._compiler_proxy_ops.primal_dtype_to_tangent_dtype", create=True
    ) as mock_op:
        mod.primal_dtype_to_tangent_dtype()
        mock_op.assert_called_once_with()


def test_primitive_uses_outfeed() -> None:
    """Test primitive_uses_outfeed."""
    with patch(
        "zero_jax._compiler_proxy_ops.primitive_uses_outfeed", create=True
    ) as mock_op:
        mod.primitive_uses_outfeed()
        mock_op.assert_called_once_with()


def test_process_env_traces_call() -> None:
    """Test process_env_traces_call."""
    with patch(
        "zero_jax._compiler_proxy_ops.process_env_traces_call", create=True
    ) as mock_op:
        mod.process_env_traces_call()
        mock_op.assert_called_once_with()


def test_process_env_traces_map() -> None:
    """Test process_env_traces_map."""
    with patch(
        "zero_jax._compiler_proxy_ops.process_env_traces_map", create=True
    ) as mock_op:
        mod.process_env_traces_map()
        mock_op.assert_called_once_with()


def test_raise_as_much_as_possible() -> None:
    """Test raise_as_much_as_possible."""
    with patch(
        "zero_jax._compiler_proxy_ops.raise_as_much_as_possible", create=True
    ) as mock_op:
        mod.raise_as_much_as_possible()
        mock_op.assert_called_once_with()


def test_raise_to_shaped() -> None:
    """Test raise_to_shaped."""
    with patch("zero_jax._compiler_proxy_ops.raise_to_shaped", create=True) as mock_op:
        mod.raise_to_shaped()
        mock_op.assert_called_once_with()


def test_reset_trace_state() -> None:
    """Test reset_trace_state."""
    with patch(
        "zero_jax._compiler_proxy_ops.reset_trace_state", create=True
    ) as mock_op:
        mod.reset_trace_state()
        mock_op.assert_called_once_with()


def test_stash_axis_env() -> None:
    """Test stash_axis_env."""
    with patch("zero_jax._compiler_proxy_ops.stash_axis_env", create=True) as mock_op:
        mod.stash_axis_env()
        mock_op.assert_called_once_with()


def test_str_eqn_compact() -> None:
    """Test str_eqn_compact."""
    with patch("zero_jax._compiler_proxy_ops.str_eqn_compact", create=True) as mock_op:
        mod.str_eqn_compact()
        mock_op.assert_called_once_with()


def test_subjaxprs() -> None:
    """Test subjaxprs."""
    with patch("zero_jax._compiler_proxy_ops.subjaxprs", create=True) as mock_op:
        mod.subjaxprs()
        mock_op.assert_called_once_with()


def test_subst_axis_names() -> None:
    """Test subst_axis_names."""
    with patch("zero_jax._compiler_proxy_ops.subst_axis_names", create=True) as mock_op:
        mod.subst_axis_names()
        mock_op.assert_called_once_with()


def test_subst_axis_names_eqn() -> None:
    """Test subst_axis_names_eqn."""
    with patch(
        "zero_jax._compiler_proxy_ops.subst_axis_names_eqn", create=True
    ) as mock_op:
        mod.subst_axis_names_eqn()
        mock_op.assert_called_once_with()


def test_subst_axis_names_jaxpr() -> None:
    """Test subst_axis_names_jaxpr."""
    with patch(
        "zero_jax._compiler_proxy_ops.subst_axis_names_jaxpr", create=True
    ) as mock_op:
        mod.subst_axis_names_jaxpr()
        mock_op.assert_called_once_with()


def test_subst_axis_names_var() -> None:
    """Test subst_axis_names_var."""
    with patch(
        "zero_jax._compiler_proxy_ops.subst_axis_names_var", create=True
    ) as mock_op:
        mod.subst_axis_names_var()
        mock_op.assert_called_once_with()


def test_substitute_vars_in_output_ty() -> None:
    """Test substitute_vars_in_output_ty."""
    with patch(
        "zero_jax._compiler_proxy_ops.substitute_vars_in_output_ty", create=True
    ) as mock_op:
        mod.substitute_vars_in_output_ty()
        mock_op.assert_called_once_with()


def test_trace_state_clean() -> None:
    """Test trace_state_clean."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_state_clean", create=True
    ) as mock_op:
        mod.trace_state_clean()
        mock_op.assert_called_once_with()


def test_traverse_jaxpr_params() -> None:
    """Test traverse_jaxpr_params."""
    with patch(
        "zero_jax._compiler_proxy_ops.traverse_jaxpr_params", create=True
    ) as mock_op:
        mod.traverse_jaxpr_params()
        mock_op.assert_called_once_with()


def test_typecheck() -> None:
    """Test typecheck."""
    with patch("zero_jax._compiler_proxy_ops.typecheck", create=True) as mock_op:
        mod.typecheck()
        mock_op.assert_called_once_with()


def test_typecompat() -> None:
    """Test typecompat."""
    with patch("zero_jax._compiler_proxy_ops.typecompat", create=True) as mock_op:
        mod.typecompat()
        mock_op.assert_called_once_with()


def test_typematch() -> None:
    """Test typematch."""
    with patch("zero_jax._compiler_proxy_ops.typematch", create=True) as mock_op:
        mod.typematch()
        mock_op.assert_called_once_with()


def test_unmapped_aval() -> None:
    """Test unmapped_aval."""
    with patch("zero_jax._compiler_proxy_ops.unmapped_aval", create=True) as mock_op:
        mod.unmapped_aval()
        mock_op.assert_called_once_with()


def test_used_axis_names() -> None:
    """Test used_axis_names."""
    with patch("zero_jax._compiler_proxy_ops.used_axis_names", create=True) as mock_op:
        mod.used_axis_names()
        mock_op.assert_called_once_with()


def test_used_axis_names_jaxpr() -> None:
    """Test used_axis_names_jaxpr."""
    with patch(
        "zero_jax._compiler_proxy_ops.used_axis_names_jaxpr", create=True
    ) as mock_op:
        mod.used_axis_names_jaxpr()
        mock_op.assert_called_once_with()


def test_valid_jaxtype() -> None:
    """Test valid_jaxtype."""
    with patch("zero_jax._compiler_proxy_ops.valid_jaxtype", create=True) as mock_op:
        mod.valid_jaxtype()
        mock_op.assert_called_once_with()


def test_Jaxpr() -> None:
    """Test Jaxpr."""
    obj = mod.Jaxpr()
    assert obj is not None


def test_Primitive() -> None:
    """Test Primitive."""
    obj = mod.Primitive()
    assert obj is not None


def test_ShapedArray() -> None:
    """Test ShapedArray."""
    obj = mod.ShapedArray()
    assert obj is not None


def test_Token() -> None:
    """Test Token."""
    obj = mod.Token()
    assert obj is not None


def test_Trace() -> None:
    """Test Trace."""
    obj = mod.Trace()
    assert obj is not None


def test_Var() -> None:
    """Test Var."""
    obj = mod.Var()
    assert obj is not None
