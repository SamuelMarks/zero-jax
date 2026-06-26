"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.mlir as mod


def test_AxisContext() -> None:
    """Test AxisContext."""
    with pytest.raises(NotImplementedError):
        mod.AxisContext()


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
    with pytest.raises(NotImplementedError):
        mod.MeshAxisName()


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
    with pytest.raises(NotImplementedError):
        mod.Value()


def test_aval_to_ir_type() -> None:
    """Test aval_to_ir_type."""
    with pytest.raises(NotImplementedError):
        mod.aval_to_ir_type()


def test_aval_to_ir_types() -> None:
    """Test aval_to_ir_types."""
    with pytest.raises(NotImplementedError):
        mod.aval_to_ir_types()


def test_core_call_lowering() -> None:
    """Test core_call_lowering."""
    with pytest.raises(NotImplementedError):
        mod.core_call_lowering()


def test_custom_call() -> None:
    """Test custom_call."""
    with pytest.raises(NotImplementedError):
        mod.custom_call()


def test_dense_bool_array() -> None:
    """Test dense_bool_array."""
    with pytest.raises(NotImplementedError):
        mod.dense_bool_array()


def test_dense_bool_elements() -> None:
    """Test dense_bool_elements."""
    with pytest.raises(NotImplementedError):
        mod.dense_bool_elements()


def test_dense_int_array() -> None:
    """Test dense_int_array."""
    with pytest.raises(NotImplementedError):
        mod.dense_int_array()


def test_dense_int_elements() -> None:
    """Test dense_int_elements."""
    with pytest.raises(NotImplementedError):
        mod.dense_int_elements()


def test_dtype_to_ir_type() -> None:
    """Test dtype_to_ir_type."""
    with pytest.raises(NotImplementedError):
        mod.dtype_to_ir_type()


def test_emit_python_callback() -> None:
    """Test emit_python_callback."""
    with pytest.raises(NotImplementedError):
        mod.emit_python_callback()


def test_flatten_lowering_ir_args() -> None:
    """Test flatten_lowering_ir_args."""
    with pytest.raises(NotImplementedError):
        mod.flatten_lowering_ir_args()


def test_i32_attr() -> None:
    """Test i32_attr."""
    with pytest.raises(NotImplementedError):
        mod.i32_attr()


def test_i64_attr() -> None:
    """Test i64_attr."""
    with pytest.raises(NotImplementedError):
        mod.i64_attr()


def test_ir_constant() -> None:
    """Test ir_constant."""
    with pytest.raises(NotImplementedError):
        mod.ir_constant()


def test_ir_constants() -> None:
    """Test ir_constants."""
    with pytest.raises(NotImplementedError):
        mod.ir_constants()


def test_jaxpr_subcomp() -> None:
    """Test jaxpr_subcomp."""
    with pytest.raises(NotImplementedError):
        mod.jaxpr_subcomp()


def test_lower_fun() -> None:
    """Test lower_fun."""
    with pytest.raises(NotImplementedError):
        mod.lower_fun()


def test_lower_jaxpr_to_fun() -> None:
    """Test lower_jaxpr_to_fun."""
    with pytest.raises(NotImplementedError):
        mod.lower_jaxpr_to_fun()


def test_lower_jaxpr_to_module() -> None:
    """Test lower_jaxpr_to_module."""
    with pytest.raises(NotImplementedError):
        mod.lower_jaxpr_to_module()


def test_make_ir_context() -> None:
    """Test make_ir_context."""
    with pytest.raises(NotImplementedError):
        mod.make_ir_context()


def test_merge_mlir_modules() -> None:
    """Test merge_mlir_modules."""
    with pytest.raises(NotImplementedError):
        mod.merge_mlir_modules()


def test_module_to_bytecode() -> None:
    """Test module_to_bytecode."""
    with pytest.raises(NotImplementedError):
        mod.module_to_bytecode()


def test_module_to_string() -> None:
    """Test module_to_string."""
    with pytest.raises(NotImplementedError):
        mod.module_to_string()


def test_register_constant_handler() -> None:
    """Test register_constant_handler."""
    with pytest.raises(NotImplementedError):
        mod.register_constant_handler()


def test_register_lowering() -> None:
    """Test register_lowering."""
    with pytest.raises(NotImplementedError):
        mod.register_lowering()


def test_shape_tensor() -> None:
    """Test shape_tensor."""
    with pytest.raises(NotImplementedError):
        mod.shape_tensor()


def test_token_type() -> None:
    """Test token_type."""
    with pytest.raises(NotImplementedError):
        mod.token_type()


def test_xla_computation_to_mlir_module() -> None:
    """Test xla_computation_to_mlir_module."""
    with pytest.raises(NotImplementedError):
        mod.xla_computation_to_mlir_module()
