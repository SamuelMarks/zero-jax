"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.interpreters.partial_eval as mod


def test_AbstractedAxesSpec() -> None:
    """Test AbstractedAxesSpec."""
    with patch(
        "zero_jax._compiler_proxy_ops.AbstractedAxesSpec", create=True
    ) as mock_op:
        mod.AbstractedAxesSpec()
        mock_op.assert_called_once_with()


def test_AbstractedAxisName() -> None:
    """Test AbstractedAxisName."""
    obj = mod.AbstractedAxisName(
        id=1, name="test_AbstractedAxisName", value="test_value"
    )
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_AbstractedAxisName"
    assert obj.value == "test_value"


def test_BoundedAxisSize() -> None:
    """Test BoundedAxisSize."""
    obj = mod.BoundedAxisSize(id=1, name="test_BoundedAxisSize", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_BoundedAxisSize"
    assert obj.value == "test_value"


def test_Const() -> None:
    """Test Const."""
    with patch("zero_jax._compiler_proxy_ops.Const", create=True) as mock_op:
        mod.Const()
        mock_op.assert_called_once_with()


def test_ConstFoldRule() -> None:
    """Test ConstFoldRule."""
    with patch("zero_jax._compiler_proxy_ops.ConstFoldRule", create=True) as mock_op:
        mod.ConstFoldRule()
        mock_op.assert_called_once_with()


def test_ConstVar() -> None:
    """Test ConstVar."""
    obj = mod.ConstVar(id=1, name="test_ConstVar", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_ConstVar"
    assert obj.value == "test_value"


def test_DCERule() -> None:
    """Test DCERule."""
    with patch("zero_jax._compiler_proxy_ops.DCERule", create=True) as mock_op:
        mod.DCERule()
        mock_op.assert_called_once_with()


def test_DebugInfo() -> None:
    """Test DebugInfo."""
    obj = mod.DebugInfo(id=1, name="test_DebugInfo", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_DebugInfo"
    assert obj.value == "test_value"


def test_DynamicJaxprTrace() -> None:
    """Test DynamicJaxprTrace."""
    obj = mod.DynamicJaxprTrace(id=1, name="test_DynamicJaxprTrace", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_DynamicJaxprTrace"
    assert obj.value == "test_value"


def test_DynamicJaxprTracer() -> None:
    """Test DynamicJaxprTracer."""
    obj = mod.DynamicJaxprTracer(
        id=1, name="test_DynamicJaxprTracer", value="test_value"
    )
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_DynamicJaxprTracer"
    assert obj.value == "test_value"


def test_ForwardingRule() -> None:
    """Test ForwardingRule."""
    with patch("zero_jax._compiler_proxy_ops.ForwardingRule", create=True) as mock_op:
        mod.ForwardingRule()
        mock_op.assert_called_once_with()


def test_FreeVar() -> None:
    """Test FreeVar."""
    obj = mod.FreeVar(id=1, name="test_FreeVar", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_FreeVar"
    assert obj.value == "test_value"


def test_Jaxpr() -> None:
    """Test Jaxpr."""
    obj = mod.Jaxpr(id=1, name="test_Jaxpr", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Jaxpr"
    assert obj.value == "test_value"


def test_JaxprEqnRecipe() -> None:
    """Test JaxprEqnRecipe."""
    obj = mod.JaxprEqnRecipe(id=1, name="test_JaxprEqnRecipe", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_JaxprEqnRecipe"
    assert obj.value == "test_value"


def test_JaxprStackFrame() -> None:
    """Test JaxprStackFrame."""
    obj = mod.JaxprStackFrame(id=1, name="test_JaxprStackFrame", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_JaxprStackFrame"
    assert obj.value == "test_value"


def test_JaxprTrace() -> None:
    """Test JaxprTrace."""
    obj = mod.JaxprTrace(id=1, name="test_JaxprTrace", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_JaxprTrace"
    assert obj.value == "test_value"


def test_JaxprTracer() -> None:
    """Test JaxprTracer."""
    obj = mod.JaxprTracer(id=1, name="test_JaxprTracer", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_JaxprTracer"
    assert obj.value == "test_value"


def test_Val() -> None:
    """Test Val."""
    obj = mod.Val(id=1, name="test_Val", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Val"
    assert obj.value == "test_value"


def test_JaxprTracerRecipe() -> None:
    """Test JaxprTracerRecipe."""
    with patch(
        "zero_jax._compiler_proxy_ops.JaxprTracerRecipe", create=True
    ) as mock_op:
        mod.JaxprTracerRecipe()
        mock_op.assert_called_once_with()


def test_LambdaBinding() -> None:
    """Test LambdaBinding."""
    obj = mod.LambdaBinding(id=1, name="test_LambdaBinding", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_LambdaBinding"
    assert obj.value == "test_value"


def test_ParamsUpdater() -> None:
    """Test ParamsUpdater."""
    with patch("zero_jax._compiler_proxy_ops.ParamsUpdater", create=True) as mock_op:
        mod.ParamsUpdater()
        mock_op.assert_called_once_with()


def test_PartialEvalCustomResult() -> None:
    """Test PartialEvalCustomResult."""
    obj = mod.PartialEvalCustomResult(
        id=1, name="test_PartialEvalCustomResult", value="test_value"
    )
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_PartialEvalCustomResult"
    assert obj.value == "test_value"


def test_PartialEvalCustomRule() -> None:
    """Test PartialEvalCustomRule."""
    with patch(
        "zero_jax._compiler_proxy_ops.PartialEvalCustomRule", create=True
    ) as mock_op:
        mod.PartialEvalCustomRule()
        mock_op.assert_called_once_with()


def test_PartialVal() -> None:
    """Test PartialVal."""
    obj = mod.PartialVal(id=1, name="test_PartialVal", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_PartialVal"
    assert obj.value == "test_value"


def test_ResAvalUpdater() -> None:
    """Test ResAvalUpdater."""
    with patch("zero_jax._compiler_proxy_ops.ResAvalUpdater", create=True) as mock_op:
        mod.ResAvalUpdater()
        mock_op.assert_called_once_with()


def test_TracerAsName() -> None:
    """Test TracerAsName."""
    obj = mod.TracerAsName(id=1, name="test_TracerAsName", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_TracerAsName"
    assert obj.value == "test_value"


def test_TracerId() -> None:
    """Test TracerId."""
    obj = mod.TracerId(id=1, name="test_TracerId", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_TracerId"
    assert obj.value == "test_value"


def test_abstract_eval_fun() -> None:
    """Test abstract_eval_fun."""
    with patch(
        "zero_jax._compiler_proxy_ops.abstract_eval_fun", create=True
    ) as mock_op:
        mod.abstract_eval_fun()
        mock_op.assert_called_once_with()


def test_arg_info_all() -> None:
    """Test arg_info_all."""
    with patch("zero_jax._compiler_proxy_ops.arg_info_all", create=True) as mock_op:
        mod.arg_info_all()
        mock_op.assert_called_once_with()


def test_call_padding_rule() -> None:
    """Test call_padding_rule."""
    with patch(
        "zero_jax._compiler_proxy_ops.call_padding_rule", create=True
    ) as mock_op:
        mod.call_padding_rule()
        mock_op.assert_called_once_with()


def test_call_partial_eval_custom_rule() -> None:
    """Test call_partial_eval_custom_rule."""
    with patch(
        "zero_jax._compiler_proxy_ops.call_partial_eval_custom_rule", create=True
    ) as mock_op:
        mod.call_partial_eval_custom_rule()
        mock_op.assert_called_once_with()


def test_close_jaxpr() -> None:
    """Test close_jaxpr."""
    with patch("zero_jax._compiler_proxy_ops.close_jaxpr", create=True) as mock_op:
        mod.close_jaxpr()
        mock_op.assert_called_once_with()


def test_closed_call_partial_eval_custom_rule() -> None:
    """Test closed_call_partial_eval_custom_rule."""
    with patch(
        "zero_jax._compiler_proxy_ops.closed_call_partial_eval_custom_rule", create=True
    ) as mock_op:
        mod.closed_call_partial_eval_custom_rule()
        mock_op.assert_called_once_with()


def test_convert_constvars_jaxpr() -> None:
    """Test convert_constvars_jaxpr."""
    with patch(
        "zero_jax._compiler_proxy_ops.convert_constvars_jaxpr", create=True
    ) as mock_op:
        mod.convert_constvars_jaxpr()
        mock_op.assert_called_once_with()


def test_convert_envvars_to_constvars() -> None:
    """Test convert_envvars_to_constvars."""
    with patch(
        "zero_jax._compiler_proxy_ops.convert_envvars_to_constvars", create=True
    ) as mock_op:
        mod.convert_envvars_to_constvars()
        mock_op.assert_called_once_with()


def test_convert_invars_to_constvars() -> None:
    """Test convert_invars_to_constvars."""
    with patch(
        "zero_jax._compiler_proxy_ops.convert_invars_to_constvars", create=True
    ) as mock_op:
        mod.convert_invars_to_constvars()
        mock_op.assert_called_once_with()


def test_dce_jaxpr() -> None:
    """Test dce_jaxpr."""
    with patch("zero_jax._compiler_proxy_ops.dce_jaxpr", create=True) as mock_op:
        mod.dce_jaxpr()
        mock_op.assert_called_once_with()


def test_dce_jaxpr_call_rule() -> None:
    """Test dce_jaxpr_call_rule."""
    with patch(
        "zero_jax._compiler_proxy_ops.dce_jaxpr_call_rule", create=True
    ) as mock_op:
        mod.dce_jaxpr_call_rule()
        mock_op.assert_called_once_with()


def test_dce_jaxpr_closed_call_rule() -> None:
    """Test dce_jaxpr_closed_call_rule."""
    with patch(
        "zero_jax._compiler_proxy_ops.dce_jaxpr_closed_call_rule", create=True
    ) as mock_op:
        mod.dce_jaxpr_closed_call_rule()
        mock_op.assert_called_once_with()


def test_dce_jaxpr_consts() -> None:
    """Test dce_jaxpr_consts."""
    with patch("zero_jax._compiler_proxy_ops.dce_jaxpr_consts", create=True) as mock_op:
        mod.dce_jaxpr_consts()
        mock_op.assert_called_once_with()


def test_debug_info() -> None:
    """Test debug_info."""
    with patch("zero_jax._compiler_proxy_ops.debug_info", create=True) as mock_op:
        mod.debug_info()
        mock_op.assert_called_once_with()


def test_debug_info_final() -> None:
    """Test debug_info_final."""
    with patch("zero_jax._compiler_proxy_ops.debug_info_final", create=True) as mock_op:
        mod.debug_info_final()
        mock_op.assert_called_once_with()


def test_def_trivial_padding() -> None:
    """Test def_trivial_padding."""
    with patch(
        "zero_jax._compiler_proxy_ops.def_trivial_padding", create=True
    ) as mock_op:
        mod.def_trivial_padding()
        mock_op.assert_called_once_with()


def test_extend_jaxpr_stack() -> None:
    """Test extend_jaxpr_stack."""
    with patch(
        "zero_jax._compiler_proxy_ops.extend_jaxpr_stack", create=True
    ) as mock_op:
        mod.extend_jaxpr_stack()
        mock_op.assert_called_once_with()


def test_infer_lambda_input_type() -> None:
    """Test infer_lambda_input_type."""
    with patch(
        "zero_jax._compiler_proxy_ops.infer_lambda_input_type", create=True
    ) as mock_op:
        mod.infer_lambda_input_type()
        mock_op.assert_called_once_with()


def test_instantiate_const_at() -> None:
    """Test instantiate_const_at."""
    with patch(
        "zero_jax._compiler_proxy_ops.instantiate_const_at", create=True
    ) as mock_op:
        mod.instantiate_const_at()
        mock_op.assert_called_once_with()


def test_make_jaxpr_effects() -> None:
    """Test make_jaxpr_effects."""
    with patch(
        "zero_jax._compiler_proxy_ops.make_jaxpr_effects", create=True
    ) as mock_op:
        mod.make_jaxpr_effects()
        mock_op.assert_called_once_with()


def test_move_binders_to_back() -> None:
    """Test move_binders_to_back."""
    with patch(
        "zero_jax._compiler_proxy_ops.move_binders_to_back", create=True
    ) as mock_op:
        mod.move_binders_to_back()
        mock_op.assert_called_once_with()


def test_move_binders_to_front() -> None:
    """Test move_binders_to_front."""
    with patch(
        "zero_jax._compiler_proxy_ops.move_binders_to_front", create=True
    ) as mock_op:
        mod.move_binders_to_front()
        mock_op.assert_called_once_with()


def test_new_eqn_recipe() -> None:
    """Test new_eqn_recipe."""
    with patch("zero_jax._compiler_proxy_ops.new_eqn_recipe", create=True) as mock_op:
        mod.new_eqn_recipe()
        mock_op.assert_called_once_with()


def test_pad_jaxpr() -> None:
    """Test pad_jaxpr."""
    with patch("zero_jax._compiler_proxy_ops.pad_jaxpr", create=True) as mock_op:
        mod.pad_jaxpr()
        mock_op.assert_called_once_with()


def test_partial_eval_jaxpr_custom() -> None:
    """Test partial_eval_jaxpr_custom."""
    with patch(
        "zero_jax._compiler_proxy_ops.partial_eval_jaxpr_custom", create=True
    ) as mock_op:
        mod.partial_eval_jaxpr_custom()
        mock_op.assert_called_once_with()


def test_partial_eval_jaxpr_custom_rule_not_implemented() -> None:
    """Test partial_eval_jaxpr_custom_rule_not_implemented."""
    with patch(
        "zero_jax._compiler_proxy_ops.partial_eval_jaxpr_custom_rule_not_implemented",
        create=True,
    ) as mock_op:
        mod.partial_eval_jaxpr_custom_rule_not_implemented()
        mock_op.assert_called_once_with()


def test_partial_eval_jaxpr_nounits() -> None:
    """Test partial_eval_jaxpr_nounits."""
    with patch(
        "zero_jax._compiler_proxy_ops.partial_eval_jaxpr_nounits", create=True
    ) as mock_op:
        mod.partial_eval_jaxpr_nounits()
        mock_op.assert_called_once_with()


def test_partial_eval_wrapper_nounits() -> None:
    """Test partial_eval_wrapper_nounits."""
    with patch(
        "zero_jax._compiler_proxy_ops.partial_eval_wrapper_nounits", create=True
    ) as mock_op:
        mod.partial_eval_wrapper_nounits()
        mock_op.assert_called_once_with()


def test_partition_pvals() -> None:
    """Test partition_pvals."""
    with patch("zero_jax._compiler_proxy_ops.partition_pvals", create=True) as mock_op:
        mod.partition_pvals()
        mock_op.assert_called_once_with()


def test_recipe_to_eqn() -> None:
    """Test recipe_to_eqn."""
    with patch("zero_jax._compiler_proxy_ops.recipe_to_eqn", create=True) as mock_op:
        mod.recipe_to_eqn()
        mock_op.assert_called_once_with()


def test_result_info() -> None:
    """Test result_info."""
    with patch("zero_jax._compiler_proxy_ops.result_info", create=True) as mock_op:
        mod.result_info()
        mock_op.assert_called_once_with()


def test_sig_info() -> None:
    """Test sig_info."""
    with patch("zero_jax._compiler_proxy_ops.sig_info", create=True) as mock_op:
        mod.sig_info()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr() -> None:
    """Test trace_to_jaxpr."""
    with patch("zero_jax._compiler_proxy_ops.trace_to_jaxpr", create=True) as mock_op:
        mod.trace_to_jaxpr()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_dynamic() -> None:
    """Test trace_to_jaxpr_dynamic."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_jaxpr_dynamic", create=True
    ) as mock_op:
        mod.trace_to_jaxpr_dynamic()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_dynamic2() -> None:
    """Test trace_to_jaxpr_dynamic2."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_jaxpr_dynamic2", create=True
    ) as mock_op:
        mod.trace_to_jaxpr_dynamic2()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_final() -> None:
    """Test trace_to_jaxpr_final."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_jaxpr_final", create=True
    ) as mock_op:
        mod.trace_to_jaxpr_final()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_final2() -> None:
    """Test trace_to_jaxpr_final2."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_jaxpr_final2", create=True
    ) as mock_op:
        mod.trace_to_jaxpr_final2()
        mock_op.assert_called_once_with()


def test_trace_to_jaxpr_nounits() -> None:
    """Test trace_to_jaxpr_nounits."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_jaxpr_nounits", create=True
    ) as mock_op:
        mod.trace_to_jaxpr_nounits()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr() -> None:
    """Test trace_to_subjaxpr."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_subjaxpr", create=True
    ) as mock_op:
        mod.trace_to_subjaxpr()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_dynamic() -> None:
    """Test trace_to_subjaxpr_dynamic."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_subjaxpr_dynamic", create=True
    ) as mock_op:
        mod.trace_to_subjaxpr_dynamic()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_dynamic2() -> None:
    """Test trace_to_subjaxpr_dynamic2."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_subjaxpr_dynamic2", create=True
    ) as mock_op:
        mod.trace_to_subjaxpr_dynamic2()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_nounits() -> None:
    """Test trace_to_subjaxpr_nounits."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_subjaxpr_nounits", create=True
    ) as mock_op:
        mod.trace_to_subjaxpr_nounits()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_nounits_dyn() -> None:
    """Test trace_to_subjaxpr_nounits_dyn."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_subjaxpr_nounits_dyn", create=True
    ) as mock_op:
        mod.trace_to_subjaxpr_nounits_dyn()
        mock_op.assert_called_once_with()


def test_trace_to_subjaxpr_nounits_fwd() -> None:
    """Test trace_to_subjaxpr_nounits_fwd."""
    with patch(
        "zero_jax._compiler_proxy_ops.trace_to_subjaxpr_nounits_fwd", create=True
    ) as mock_op:
        mod.trace_to_subjaxpr_nounits_fwd()
        mock_op.assert_called_once_with()


def test_tracers_to_jaxpr() -> None:
    """Test tracers_to_jaxpr."""
    with patch("zero_jax._compiler_proxy_ops.tracers_to_jaxpr", create=True) as mock_op:
        mod.tracers_to_jaxpr()
        mock_op.assert_called_once_with()


def test_trivial_ctx() -> None:
    """Test trivial_ctx."""
    with patch("zero_jax._compiler_proxy_ops.trivial_ctx", create=True) as mock_op:
        mod.trivial_ctx()
        mock_op.assert_called_once_with()
