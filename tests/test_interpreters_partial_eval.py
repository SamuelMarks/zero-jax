"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.interpreters.partial_eval as mod


def test_AbstractedAxesSpec() -> None:
    """Test AbstractedAxesSpec."""
    with patch("ml_switcheroo_compiler.ops.AbstractedAxesSpec") as mock_op:
        mod.AbstractedAxesSpec()
        mock_op.assert_called_once_with()


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
    with patch("ml_switcheroo_compiler.ops.Const") as mock_op:
        mod.Const()
        mock_op.assert_called_once_with()


def test_ConstFoldRule() -> None:
    """Test ConstFoldRule."""
    with patch("ml_switcheroo_compiler.ops.ConstFoldRule") as mock_op:
        mod.ConstFoldRule()
        mock_op.assert_called_once_with()


def test_ConstVar() -> None:
    """Test ConstVar."""
    obj = mod.ConstVar()
    assert obj is not None


def test_DCERule() -> None:
    """Test DCERule."""
    with patch("ml_switcheroo_compiler.ops.DCERule") as mock_op:
        mod.DCERule()
        mock_op.assert_called_once_with()


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
    with patch("ml_switcheroo_compiler.ops.ForwardingRule") as mock_op:
        mod.ForwardingRule()
        mock_op.assert_called_once_with()


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
    with patch("ml_switcheroo_compiler.ops.JaxprTracerRecipe") as mock_op:
        mod.JaxprTracerRecipe()
        mock_op.assert_called_once_with()


def test_LambdaBinding() -> None:
    """Test LambdaBinding."""
    obj = mod.LambdaBinding()
    assert obj is not None


def test_ParamsUpdater() -> None:
    """Test ParamsUpdater."""
    with patch("ml_switcheroo_compiler.ops.ParamsUpdater") as mock_op:
        mod.ParamsUpdater()
        mock_op.assert_called_once_with()


def test_PartialEvalCustomResult() -> None:
    """Test PartialEvalCustomResult."""
    obj = mod.PartialEvalCustomResult()
    assert obj is not None


def test_PartialEvalCustomRule() -> None:
    """Test PartialEvalCustomRule."""
    with patch("ml_switcheroo_compiler.ops.PartialEvalCustomRule") as mock_op:
        mod.PartialEvalCustomRule()
        mock_op.assert_called_once_with()


def test_PartialVal() -> None:
    """Test PartialVal."""
    obj = mod.PartialVal()
    assert obj is not None


def test_ResAvalUpdater() -> None:
    """Test ResAvalUpdater."""
    with patch("ml_switcheroo_compiler.ops.ResAvalUpdater") as mock_op:
        mod.ResAvalUpdater()
        mock_op.assert_called_once_with()


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
    with patch("ml_switcheroo_compiler.ops.abstract_eval_fun") as mock_op:
        mod.abstract_eval_fun()
        mock_op.assert_called_once_with()


def test_arg_info_all() -> None:
    """Test arg_info_all."""
    with patch("ml_switcheroo_compiler.ops.arg_info_all") as mock_op:
        mod.arg_info_all()
        mock_op.assert_called_once_with()


def test_call_padding_rule() -> None:
    """Test call_padding_rule."""
    with patch("ml_switcheroo_compiler.ops.call_padding_rule") as mock_op:
        mod.call_padding_rule()
        mock_op.assert_called_once_with()


def test_call_partial_eval_custom_rule() -> None:
    """Test call_partial_eval_custom_rule."""
    with patch("ml_switcheroo_compiler.ops.call_partial_eval_custom_rule") as mock_op:
        mod.call_partial_eval_custom_rule()
        mock_op.assert_called_once_with()


def test_close_jaxpr() -> None:
    """Test close_jaxpr."""
    with patch("ml_switcheroo_compiler.ops.close_jaxpr") as mock_op:
        mod.close_jaxpr()
        mock_op.assert_called_once_with()


def test_closed_call_partial_eval_custom_rule() -> None:
    """Test closed_call_partial_eval_custom_rule."""
    with patch(
        "ml_switcheroo_compiler.ops.closed_call_partial_eval_custom_rule"
    ) as mock_op:
        mod.closed_call_partial_eval_custom_rule()
        mock_op.assert_called_once_with()


def test_convert_constvars_jaxpr() -> None:
    """Test convert_constvars_jaxpr."""
    with patch("ml_switcheroo_compiler.ops.convert_constvars_jaxpr") as mock_op:
        mod.convert_constvars_jaxpr()
        mock_op.assert_called_once_with()


def test_convert_envvars_to_constvars() -> None:
    """Test convert_envvars_to_constvars."""
    with patch("ml_switcheroo_compiler.ops.convert_envvars_to_constvars") as mock_op:
        mod.convert_envvars_to_constvars()
        mock_op.assert_called_once_with()


def test_convert_invars_to_constvars() -> None:
    """Test convert_invars_to_constvars."""
    with patch("ml_switcheroo_compiler.ops.convert_invars_to_constvars") as mock_op:
        mod.convert_invars_to_constvars()
        mock_op.assert_called_once_with()


def test_dce_jaxpr() -> None:
    """Test dce_jaxpr."""
    with patch("ml_switcheroo_compiler.ops.dce_jaxpr") as mock_op:
        mod.dce_jaxpr()
        mock_op.assert_called_once_with()


def test_dce_jaxpr_call_rule() -> None:
    """Test dce_jaxpr_call_rule."""
    with patch("ml_switcheroo_compiler.ops.dce_jaxpr_call_rule") as mock_op:
        mod.dce_jaxpr_call_rule()
        mock_op.assert_called_once_with()


def test_dce_jaxpr_closed_call_rule() -> None:
    """Test dce_jaxpr_closed_call_rule."""
    with patch("ml_switcheroo_compiler.ops.dce_jaxpr_closed_call_rule") as mock_op:
        mod.dce_jaxpr_closed_call_rule()
        mock_op.assert_called_once_with()


def test_dce_jaxpr_consts() -> None:
    """Test dce_jaxpr_consts."""
    with patch("ml_switcheroo_compiler.ops.dce_jaxpr_consts") as mock_op:
        mod.dce_jaxpr_consts()
        mock_op.assert_called_once_with()


def test_debug_info() -> None:
    """Test debug_info."""
    with patch("ml_switcheroo_compiler.ops.debug_info") as mock_op:
        mod.debug_info()
        mock_op.assert_called_once_with()


def test_debug_info_final() -> None:
    """Test debug_info_final."""
    with patch("ml_switcheroo_compiler.ops.debug_info_final") as mock_op:
        mod.debug_info_final()
        mock_op.assert_called_once_with()


def test_def_trivial_padding() -> None:
    """Test def_trivial_padding."""
    with patch("ml_switcheroo_compiler.ops.def_trivial_padding") as mock_op:
        mod.def_trivial_padding()
        mock_op.assert_called_once_with()


def test_extend_jaxpr_stack() -> None:
    """Test extend_jaxpr_stack."""
    with patch("ml_switcheroo_compiler.ops.extend_jaxpr_stack") as mock_op:
        mod.extend_jaxpr_stack()
        mock_op.assert_called_once_with()


def test_infer_lambda_input_type() -> None:
    """Test infer_lambda_input_type."""
    with patch("ml_switcheroo_compiler.ops.infer_lambda_input_type") as mock_op:
        mod.infer_lambda_input_type()
        mock_op.assert_called_once_with()


def test_instantiate_const_at() -> None:
    """Test instantiate_const_at."""
    with patch("ml_switcheroo_compiler.ops.instantiate_const_at") as mock_op:
        mod.instantiate_const_at()
        mock_op.assert_called_once_with()


def test_make_jaxpr_effects() -> None:
    """Test make_jaxpr_effects."""
    with patch("ml_switcheroo_compiler.ops.make_jaxpr_effects") as mock_op:
        mod.make_jaxpr_effects()
        mock_op.assert_called_once_with()


def test_move_binders_to_back() -> None:
    """Test move_binders_to_back."""
    with patch("ml_switcheroo_compiler.ops.move_binders_to_back") as mock_op:
        mod.move_binders_to_back()
        mock_op.assert_called_once_with()


def test_move_binders_to_front() -> None:
    """Test move_binders_to_front."""
    with patch("ml_switcheroo_compiler.ops.move_binders_to_front") as mock_op:
        mod.move_binders_to_front()
        mock_op.assert_called_once_with()


def test_new_eqn_recipe() -> None:
    """Test new_eqn_recipe."""
    with patch("ml_switcheroo_compiler.ops.new_eqn_recipe") as mock_op:
        mod.new_eqn_recipe()
        mock_op.assert_called_once_with()


def test_pad_jaxpr() -> None:
    """Test pad_jaxpr."""
    with patch("ml_switcheroo_compiler.ops.pad_jaxpr") as mock_op:
        mod.pad_jaxpr()
        mock_op.assert_called_once_with()


def test_partial_eval_jaxpr_custom() -> None:
    """Test partial_eval_jaxpr_custom."""
    with patch("ml_switcheroo_compiler.ops.partial_eval_jaxpr_custom") as mock_op:
        mod.partial_eval_jaxpr_custom()
        mock_op.assert_called_once_with()


def test_partial_eval_jaxpr_custom_rule_not_implemented() -> None:
    """Test partial_eval_jaxpr_custom_rule_not_implemented."""
    with patch(
        "ml_switcheroo_compiler.ops.partial_eval_jaxpr_custom_rule_not_implemented"
    ) as mock_op:
        mod.partial_eval_jaxpr_custom_rule_not_implemented()
        mock_op.assert_called_once_with()


def test_partial_eval_jaxpr_nounits() -> None:
    """Test partial_eval_jaxpr_nounits."""
    with patch("ml_switcheroo_compiler.ops.partial_eval_jaxpr_nounits") as mock_op:
        mod.partial_eval_jaxpr_nounits()
        mock_op.assert_called_once_with()


def test_partial_eval_wrapper_nounits() -> None:
    """Test partial_eval_wrapper_nounits."""
    with patch("ml_switcheroo_compiler.ops.partial_eval_wrapper_nounits") as mock_op:
        mod.partial_eval_wrapper_nounits()
        mock_op.assert_called_once_with()


def test_partition_pvals() -> None:
    """Test partition_pvals."""
    with patch("ml_switcheroo_compiler.ops.partition_pvals") as mock_op:
        mod.partition_pvals()
        mock_op.assert_called_once_with()


def test_recipe_to_eqn() -> None:
    """Test recipe_to_eqn."""
    with patch("ml_switcheroo_compiler.ops.recipe_to_eqn") as mock_op:
        mod.recipe_to_eqn()
        mock_op.assert_called_once_with()


def test_result_info() -> None:
    """Test result_info."""
    with patch("ml_switcheroo_compiler.ops.result_info") as mock_op:
        mod.result_info()
        mock_op.assert_called_once_with()


def test_sig_info() -> None:
    """Test sig_info."""
    with patch("ml_switcheroo_compiler.ops.sig_info") as mock_op:
        mod.sig_info()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr() -> None:
    """Test trace_to_jaxpr."""
    with patch("ml_switcheroo_compiler.ops.trace_to_jaxpr") as mock_op:
        mod.trace_to_jaxpr()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_dynamic() -> None:
    """Test trace_to_jaxpr_dynamic."""
    with patch("ml_switcheroo_compiler.ops.trace_to_jaxpr_dynamic") as mock_op:
        mod.trace_to_jaxpr_dynamic()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_dynamic2() -> None:
    """Test trace_to_jaxpr_dynamic2."""
    with patch("ml_switcheroo_compiler.ops.trace_to_jaxpr_dynamic2") as mock_op:
        mod.trace_to_jaxpr_dynamic2()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_final() -> None:
    """Test trace_to_jaxpr_final."""
    with patch("ml_switcheroo_compiler.ops.trace_to_jaxpr_final") as mock_op:
        mod.trace_to_jaxpr_final()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_final2() -> None:
    """Test trace_to_jaxpr_final2."""
    with patch("ml_switcheroo_compiler.ops.trace_to_jaxpr_final2") as mock_op:
        mod.trace_to_jaxpr_final2()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_nounits() -> None:
    """Test trace_to_jaxpr_nounits."""
    with patch("ml_switcheroo_compiler.ops.trace_to_jaxpr_nounits") as mock_op:
        mod.trace_to_jaxpr_nounits()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr() -> None:
    """Test trace_to_subjaxpr."""
    with patch("ml_switcheroo_compiler.ops.trace_to_subjaxpr") as mock_op:
        mod.trace_to_subjaxpr()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_dynamic() -> None:
    """Test trace_to_subjaxpr_dynamic."""
    with patch("ml_switcheroo_compiler.ops.trace_to_subjaxpr_dynamic") as mock_op:
        mod.trace_to_subjaxpr_dynamic()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_dynamic2() -> None:
    """Test trace_to_subjaxpr_dynamic2."""
    with patch("ml_switcheroo_compiler.ops.trace_to_subjaxpr_dynamic2") as mock_op:
        mod.trace_to_subjaxpr_dynamic2()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_nounits() -> None:
    """Test trace_to_subjaxpr_nounits."""
    with patch("ml_switcheroo_compiler.ops.trace_to_subjaxpr_nounits") as mock_op:
        mod.trace_to_subjaxpr_nounits()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_nounits_dyn() -> None:
    """Test trace_to_subjaxpr_nounits_dyn."""
    with patch("ml_switcheroo_compiler.ops.trace_to_subjaxpr_nounits_dyn") as mock_op:
        mod.trace_to_subjaxpr_nounits_dyn()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_nounits_fwd() -> None:
    """Test trace_to_subjaxpr_nounits_fwd."""
    with patch("ml_switcheroo_compiler.ops.trace_to_subjaxpr_nounits_fwd") as mock_op:
        mod.trace_to_subjaxpr_nounits_fwd()
        mock_op.assert_called_once_with()


def test_tracers_to_jaxpr() -> None:
    """Test tracers_to_jaxpr."""
    with patch("ml_switcheroo_compiler.ops.tracers_to_jaxpr") as mock_op:
        mod.tracers_to_jaxpr()
        mock_op.assert_called_once_with()


def test_trivial_ctx() -> None:
    """Test trivial_ctx."""
    with patch("ml_switcheroo_compiler.ops.trivial_ctx") as mock_op:
        mod.trivial_ctx()
        mock_op.assert_called_once_with()
