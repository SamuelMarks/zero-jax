"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.partial_eval as mod


def test_AbstractedAxesSpec() -> None:
    """Test AbstractedAxesSpec."""
    with pytest.raises(NotImplementedError):
        mod.AbstractedAxesSpec()


def test_AbstractedAxisName() -> None:
    """Test AbstractedAxisName."""
    obj = mod.AbstractedAxisName()
    assert obj is not None


def test_BoundedAxisSize() -> None:
    """Test BoundedAxisSize."""
    obj = mod.BoundedAxisSize()
    assert obj is not None


def test_Const() -> None:
    """Test Const."""
    with pytest.raises(NotImplementedError):
        mod.Const()


def test_ConstFoldRule() -> None:
    """Test ConstFoldRule."""
    with pytest.raises(NotImplementedError):
        mod.ConstFoldRule()


def test_ConstVar() -> None:
    """Test ConstVar."""
    obj = mod.ConstVar()
    assert obj is not None


def test_DCERule() -> None:
    """Test DCERule."""
    with pytest.raises(NotImplementedError):
        mod.DCERule()


def test_DebugInfo() -> None:
    """Test DebugInfo."""
    obj = mod.DebugInfo()
    assert obj is not None


def test_DynamicJaxprTrace() -> None:
    """Test DynamicJaxprTrace."""
    obj = mod.DynamicJaxprTrace()
    assert obj is not None


def test_DynamicJaxprTracer() -> None:
    """Test DynamicJaxprTracer."""
    obj = mod.DynamicJaxprTracer()
    assert obj is not None


def test_ForwardingRule() -> None:
    """Test ForwardingRule."""
    with pytest.raises(NotImplementedError):
        mod.ForwardingRule()


def test_FreeVar() -> None:
    """Test FreeVar."""
    obj = mod.FreeVar()
    assert obj is not None


def test_Jaxpr() -> None:
    """Test Jaxpr."""
    obj = mod.Jaxpr()
    assert obj is not None


def test_JaxprEqnRecipe() -> None:
    """Test JaxprEqnRecipe."""
    obj = mod.JaxprEqnRecipe()
    assert obj is not None


def test_JaxprStackFrame() -> None:
    """Test JaxprStackFrame."""
    obj = mod.JaxprStackFrame()
    assert obj is not None


def test_JaxprTracerRecipe() -> None:
    """Test JaxprTracerRecipe."""
    with pytest.raises(NotImplementedError):
        mod.JaxprTracerRecipe()


def test_LambdaBinding() -> None:
    """Test LambdaBinding."""
    obj = mod.LambdaBinding()
    assert obj is not None


def test_ParamsUpdater() -> None:
    """Test ParamsUpdater."""
    with pytest.raises(NotImplementedError):
        mod.ParamsUpdater()


def test_PartialEvalCustomResult() -> None:
    """Test PartialEvalCustomResult."""
    obj = mod.PartialEvalCustomResult()
    assert obj is not None


def test_PartialEvalCustomRule() -> None:
    """Test PartialEvalCustomRule."""
    with pytest.raises(NotImplementedError):
        mod.PartialEvalCustomRule()


def test_PartialVal() -> None:
    """Test PartialVal."""
    obj = mod.PartialVal()
    assert obj is not None


def test_ResAvalUpdater() -> None:
    """Test ResAvalUpdater."""
    with pytest.raises(NotImplementedError):
        mod.ResAvalUpdater()


def test_TracerAsName() -> None:
    """Test TracerAsName."""
    obj = mod.TracerAsName()
    assert obj is not None


def test_TracerId() -> None:
    """Test TracerId."""
    obj = mod.TracerId()
    assert obj is not None


def test_abstract_eval_fun() -> None:
    """Test abstract_eval_fun."""
    with pytest.raises(NotImplementedError):
        mod.abstract_eval_fun()


def test_arg_info_all() -> None:
    """Test arg_info_all."""
    with pytest.raises(NotImplementedError):
        mod.arg_info_all()


def test_call_padding_rule() -> None:
    """Test call_padding_rule."""
    with pytest.raises(NotImplementedError):
        mod.call_padding_rule()


def test_call_partial_eval_custom_rule() -> None:
    """Test call_partial_eval_custom_rule."""
    with pytest.raises(NotImplementedError):
        mod.call_partial_eval_custom_rule()


def test_close_jaxpr() -> None:
    """Test close_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.close_jaxpr()


def test_closed_call_partial_eval_custom_rule() -> None:
    """Test closed_call_partial_eval_custom_rule."""
    with pytest.raises(NotImplementedError):
        mod.closed_call_partial_eval_custom_rule()


def test_convert_constvars_jaxpr() -> None:
    """Test convert_constvars_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.convert_constvars_jaxpr()


def test_convert_envvars_to_constvars() -> None:
    """Test convert_envvars_to_constvars."""
    with pytest.raises(NotImplementedError):
        mod.convert_envvars_to_constvars()


def test_convert_invars_to_constvars() -> None:
    """Test convert_invars_to_constvars."""
    with pytest.raises(NotImplementedError):
        mod.convert_invars_to_constvars()


def test_dce_jaxpr() -> None:
    """Test dce_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.dce_jaxpr()


def test_dce_jaxpr_call_rule() -> None:
    """Test dce_jaxpr_call_rule."""
    with pytest.raises(NotImplementedError):
        mod.dce_jaxpr_call_rule()


def test_dce_jaxpr_closed_call_rule() -> None:
    """Test dce_jaxpr_closed_call_rule."""
    with pytest.raises(NotImplementedError):
        mod.dce_jaxpr_closed_call_rule()


def test_dce_jaxpr_consts() -> None:
    """Test dce_jaxpr_consts."""
    with pytest.raises(NotImplementedError):
        mod.dce_jaxpr_consts()


def test_debug_info() -> None:
    """Test debug_info."""
    with pytest.raises(NotImplementedError):
        mod.debug_info()


def test_debug_info_final() -> None:
    """Test debug_info_final."""
    with pytest.raises(NotImplementedError):
        mod.debug_info_final()


def test_def_trivial_padding() -> None:
    """Test def_trivial_padding."""
    with pytest.raises(NotImplementedError):
        mod.def_trivial_padding()


def test_extend_jaxpr_stack() -> None:
    """Test extend_jaxpr_stack."""
    with pytest.raises(NotImplementedError):
        mod.extend_jaxpr_stack()


def test_infer_lambda_input_type() -> None:
    """Test infer_lambda_input_type."""
    with pytest.raises(NotImplementedError):
        mod.infer_lambda_input_type()


def test_instantiate_const_at() -> None:
    """Test instantiate_const_at."""
    with pytest.raises(NotImplementedError):
        mod.instantiate_const_at()


def test_make_jaxpr_effects() -> None:
    """Test make_jaxpr_effects."""
    with pytest.raises(NotImplementedError):
        mod.make_jaxpr_effects()


def test_move_binders_to_back() -> None:
    """Test move_binders_to_back."""
    with pytest.raises(NotImplementedError):
        mod.move_binders_to_back()


def test_move_binders_to_front() -> None:
    """Test move_binders_to_front."""
    with pytest.raises(NotImplementedError):
        mod.move_binders_to_front()


def test_new_eqn_recipe() -> None:
    """Test new_eqn_recipe."""
    with pytest.raises(NotImplementedError):
        mod.new_eqn_recipe()


def test_pad_jaxpr() -> None:
    """Test pad_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.pad_jaxpr()


def test_partial_eval_jaxpr_custom() -> None:
    """Test partial_eval_jaxpr_custom."""
    with pytest.raises(NotImplementedError):
        mod.partial_eval_jaxpr_custom()


def test_partial_eval_jaxpr_custom_rule_not_implemented() -> None:
    """Test partial_eval_jaxpr_custom_rule_not_implemented."""
    with pytest.raises(NotImplementedError):
        mod.partial_eval_jaxpr_custom_rule_not_implemented()


def test_partial_eval_jaxpr_nounits() -> None:
    """Test partial_eval_jaxpr_nounits."""
    with pytest.raises(NotImplementedError):
        mod.partial_eval_jaxpr_nounits()


def test_partial_eval_wrapper_nounits() -> None:
    """Test partial_eval_wrapper_nounits."""
    with pytest.raises(NotImplementedError):
        mod.partial_eval_wrapper_nounits()


def test_partition_pvals() -> None:
    """Test partition_pvals."""
    with pytest.raises(NotImplementedError):
        mod.partition_pvals()


def test_recipe_to_eqn() -> None:
    """Test recipe_to_eqn."""
    with pytest.raises(NotImplementedError):
        mod.recipe_to_eqn()


def test_result_info() -> None:
    """Test result_info."""
    with pytest.raises(NotImplementedError):
        mod.result_info()


def test_sig_info() -> None:
    """Test sig_info."""
    with pytest.raises(NotImplementedError):
        mod.sig_info()


def test_trace_to_jaxpr() -> None:
    """Test trace_to_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_jaxpr()


def test_trace_to_jaxpr_dynamic() -> None:
    """Test trace_to_jaxpr_dynamic."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_jaxpr_dynamic()


def test_trace_to_jaxpr_dynamic2() -> None:
    """Test trace_to_jaxpr_dynamic2."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_jaxpr_dynamic2()


def test_trace_to_jaxpr_final() -> None:
    """Test trace_to_jaxpr_final."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_jaxpr_final()


def test_trace_to_jaxpr_final2() -> None:
    """Test trace_to_jaxpr_final2."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_jaxpr_final2()


def test_trace_to_jaxpr_nounits() -> None:
    """Test trace_to_jaxpr_nounits."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_jaxpr_nounits()


def test_trace_to_subjaxpr() -> None:
    """Test trace_to_subjaxpr."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_subjaxpr()


def test_trace_to_subjaxpr_dynamic() -> None:
    """Test trace_to_subjaxpr_dynamic."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_subjaxpr_dynamic()


def test_trace_to_subjaxpr_dynamic2() -> None:
    """Test trace_to_subjaxpr_dynamic2."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_subjaxpr_dynamic2()


def test_trace_to_subjaxpr_nounits() -> None:
    """Test trace_to_subjaxpr_nounits."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_subjaxpr_nounits()


def test_trace_to_subjaxpr_nounits_dyn() -> None:
    """Test trace_to_subjaxpr_nounits_dyn."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_subjaxpr_nounits_dyn()


def test_trace_to_subjaxpr_nounits_fwd() -> None:
    """Test trace_to_subjaxpr_nounits_fwd."""
    with pytest.raises(NotImplementedError):
        mod.trace_to_subjaxpr_nounits_fwd()


def test_tracers_to_jaxpr() -> None:
    """Test tracers_to_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.tracers_to_jaxpr()


def test_trivial_ctx() -> None:
    """Test trivial_ctx."""
    with pytest.raises(NotImplementedError):
        mod.trivial_ctx()
