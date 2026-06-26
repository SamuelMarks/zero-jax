"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.interpreters.mlir as mod


def test_AxisContext() -> None:
    """Test AxisContext."""
    with patch("ml_switcheroo_compiler.ops.AxisContext") as mock_op:
        mod.AxisContext()
        mock_op.assert_called_once_with()


def test_ConstantHandler() -> None:
    """Test ConstantHandler."""
    obj = mod.ConstantHandler()
    assert obj is not None


def test_LoweringParameters() -> None:
    """Test LoweringParameters."""
    obj = mod.LoweringParameters()
    assert obj is not None


def test_LoweringResult() -> None:
    """Test LoweringResult."""
    obj = mod.LoweringResult()
    assert obj is not None


def test_LoweringRule() -> None:
    """Test LoweringRule."""
    obj = mod.LoweringRule()
    assert obj is not None


def test_LoweringRuleContext() -> None:
    """Test LoweringRuleContext."""
    obj = mod.LoweringRuleContext()
    assert obj is not None


def test_Mesh() -> None:
    """Test Mesh."""
    obj = mod.Mesh()
    assert obj is not None


def test_MeshAxisName() -> None:
    """Test MeshAxisName."""
    with patch("ml_switcheroo_compiler.ops.MeshAxisName") as mock_op:
        mod.MeshAxisName()
        mock_op.assert_called_once_with()


def test_ModuleContext() -> None:
    """Test ModuleContext."""
    obj = mod.ModuleContext()
    assert obj is not None


def test_ReplicaAxisContext() -> None:
    """Test ReplicaAxisContext."""
    obj = mod.ReplicaAxisContext()
    assert obj is not None


def test_SPMDAxisContext() -> None:
    """Test SPMDAxisContext."""
    obj = mod.SPMDAxisContext()
    assert obj is not None


def test_ShapePolyLoweringState() -> None:
    """Test ShapePolyLoweringState."""
    obj = mod.ShapePolyLoweringState()
    assert obj is not None


def test_ShardingContext() -> None:
    """Test ShardingContext."""
    obj = mod.ShardingContext()
    assert obj is not None


def test_Token() -> None:
    """Test Token."""
    obj = mod.Token()
    assert obj is not None


def test_TokenSet() -> None:
    """Test TokenSet."""
    obj = mod.TokenSet()
    assert obj is not None


def test_Value() -> None:
    """Test Value."""
    with patch("ml_switcheroo_compiler.ops.Value") as mock_op:
        mod.Value()
        mock_op.assert_called_once_with()


def test_aval_to_ir_type() -> None:
    """Test aval_to_ir_type."""
    with patch("ml_switcheroo_compiler.ops.aval_to_ir_type") as mock_op:
        mod.aval_to_ir_type()
        mock_op.assert_called_once_with()


def test_aval_to_ir_types() -> None:
    """Test aval_to_ir_types."""
    with patch("ml_switcheroo_compiler.ops.aval_to_ir_types") as mock_op:
        mod.aval_to_ir_types()
        mock_op.assert_called_once_with()


def test_core_call_lowering() -> None:
    """Test core_call_lowering."""
    with patch("ml_switcheroo_compiler.ops.core_call_lowering") as mock_op:
        mod.core_call_lowering()
        mock_op.assert_called_once_with()


def test_custom_call() -> None:
    """Test custom_call."""
    with patch("ml_switcheroo_compiler.ops.custom_call") as mock_op:
        mod.custom_call()
        mock_op.assert_called_once_with()


def test_dense_bool_array() -> None:
    """Test dense_bool_array."""
    with patch("ml_switcheroo_compiler.ops.dense_bool_array") as mock_op:
        mod.dense_bool_array()
        mock_op.assert_called_once_with()


def test_dense_bool_elements() -> None:
    """Test dense_bool_elements."""
    with patch("ml_switcheroo_compiler.ops.dense_bool_elements") as mock_op:
        mod.dense_bool_elements()
        mock_op.assert_called_once_with()


def test_dense_int_array() -> None:
    """Test dense_int_array."""
    with patch("ml_switcheroo_compiler.ops.dense_int_array") as mock_op:
        mod.dense_int_array()
        mock_op.assert_called_once_with()


def test_dense_int_elements() -> None:
    """Test dense_int_elements."""
    with patch("ml_switcheroo_compiler.ops.dense_int_elements") as mock_op:
        mod.dense_int_elements()
        mock_op.assert_called_once_with()


def test_dtype_to_ir_type() -> None:
    """Test dtype_to_ir_type."""
    with patch("ml_switcheroo_compiler.ops.dtype_to_ir_type") as mock_op:
        mod.dtype_to_ir_type()
        mock_op.assert_called_once_with()


def test_emit_python_callback() -> None:
    """Test emit_python_callback."""
    with patch("ml_switcheroo_compiler.ops.emit_python_callback") as mock_op:
        mod.emit_python_callback()
        mock_op.assert_called_once_with()


def test_flatten_lowering_ir_args() -> None:
    """Test flatten_lowering_ir_args."""
    with patch("ml_switcheroo_compiler.ops.flatten_lowering_ir_args") as mock_op:
        mod.flatten_lowering_ir_args()
        mock_op.assert_called_once_with()


def test_i32_attr() -> None:
    """Test i32_attr."""
    with patch("ml_switcheroo_compiler.ops.i32_attr") as mock_op:
        mod.i32_attr()
        mock_op.assert_called_once_with()


def test_i64_attr() -> None:
    """Test i64_attr."""
    with patch("ml_switcheroo_compiler.ops.i64_attr") as mock_op:
        mod.i64_attr()
        mock_op.assert_called_once_with()


def test_ir_constant() -> None:
    """Test ir_constant."""
    with patch("ml_switcheroo_compiler.ops.ir_constant") as mock_op:
        mod.ir_constant()
        mock_op.assert_called_once_with()


def test_ir_constants() -> None:
    """Test ir_constants."""
    with patch("ml_switcheroo_compiler.ops.ir_constants") as mock_op:
        mod.ir_constants()
        mock_op.assert_called_once_with()


def test_jaxpr_subcomp() -> None:
    """Test jaxpr_subcomp."""
    with patch("ml_switcheroo_compiler.ops.jaxpr_subcomp") as mock_op:
        mod.jaxpr_subcomp()
        mock_op.assert_called_once_with()


def test_lower_fun() -> None:
    """Test lower_fun."""
    with patch("ml_switcheroo_compiler.ops.lower_fun") as mock_op:
        mod.lower_fun()
        mock_op.assert_called_once_with()


def test_lower_jaxpr_to_fun() -> None:
    """Test lower_jaxpr_to_fun."""
    with patch("ml_switcheroo_compiler.ops.lower_jaxpr_to_fun") as mock_op:
        mod.lower_jaxpr_to_fun()
        mock_op.assert_called_once_with()


def test_lower_jaxpr_to_module() -> None:
    """Test lower_jaxpr_to_module."""
    with patch("ml_switcheroo_compiler.ops.lower_jaxpr_to_module") as mock_op:
        mod.lower_jaxpr_to_module()
        mock_op.assert_called_once_with()


def test_make_ir_context() -> None:
    """Test make_ir_context."""
    with patch("ml_switcheroo_compiler.ops.make_ir_context") as mock_op:
        mod.make_ir_context()
        mock_op.assert_called_once_with()


def test_merge_mlir_modules() -> None:
    """Test merge_mlir_modules."""
    with patch("ml_switcheroo_compiler.ops.merge_mlir_modules") as mock_op:
        mod.merge_mlir_modules()
        mock_op.assert_called_once_with()


def test_module_to_bytecode() -> None:
    """Test module_to_bytecode."""
    with patch("ml_switcheroo_compiler.ops.module_to_bytecode") as mock_op:
        mod.module_to_bytecode()
        mock_op.assert_called_once_with()


def test_module_to_string() -> None:
    """Test module_to_string."""
    with patch("ml_switcheroo_compiler.ops.module_to_string") as mock_op:
        mod.module_to_string()
        mock_op.assert_called_once_with()


def test_register_constant_handler() -> None:
    """Test register_constant_handler."""
    with patch("ml_switcheroo_compiler.ops.register_constant_handler") as mock_op:
        mod.register_constant_handler()
        mock_op.assert_called_once_with()


def test_register_lowering() -> None:
    """Test register_lowering."""
    with patch("ml_switcheroo_compiler.ops.register_lowering") as mock_op:
        mod.register_lowering()
        mock_op.assert_called_once_with()


def test_shape_tensor() -> None:
    """Test shape_tensor."""
    with patch("ml_switcheroo_compiler.ops.shape_tensor") as mock_op:
        mod.shape_tensor()
        mock_op.assert_called_once_with()


def test_token_type() -> None:
    """Test token_type."""
    with patch("ml_switcheroo_compiler.ops.token_type") as mock_op:
        mod.token_type()
        mock_op.assert_called_once_with()


def test_xla_computation_to_mlir_module() -> None:
    """Test xla_computation_to_mlir_module."""
    with patch("ml_switcheroo_compiler.ops.xla_computation_to_mlir_module") as mock_op:
        mod.xla_computation_to_mlir_module()
        mock_op.assert_called_once_with()
