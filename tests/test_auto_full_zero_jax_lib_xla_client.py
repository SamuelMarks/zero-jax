"""Tests for zero_jax.lib.xla_client."""

from __future__ import annotations

import pytest

import zero_jax.lib.xla_client as mod


def test_class_ArrayImpl() -> None:
    """Test ArrayImpl."""
    obj = mod.ArrayImpl(name="ArrayImpl", value=1)
    assert isinstance(obj, mod.ArrayImpl)
    assert obj.name == "ArrayImpl"
    assert obj.value == 1


def test_class_Client() -> None:
    """Test Client."""
    obj = mod.Client(name="Client", value=1)
    assert isinstance(obj, mod.Client)
    assert obj.name == "Client"
    assert obj.value == 1


def test_class_CompileOptions() -> None:
    """Test CompileOptions."""
    obj = mod.CompileOptions(name="CompileOptions", value=1)
    assert isinstance(obj, mod.CompileOptions)
    assert obj.name == "CompileOptions"
    assert obj.value == 1


def test_class_ConvolutionDimensionNumbers() -> None:
    """Test ConvolutionDimensionNumbers."""
    obj = mod.ConvolutionDimensionNumbers(name="ConvolutionDimensionNumbers", value=1)
    assert isinstance(obj, mod.ConvolutionDimensionNumbers)
    assert obj.name == "ConvolutionDimensionNumbers"
    assert obj.value == 1


def test_CurrentSourceInfoMetadata() -> None:
    """Test CurrentSourceInfoMetadata."""
    with pytest.raises(AttributeError):
        mod.CurrentSourceInfoMetadata()


def test_class_CustomCallHandler() -> None:
    """Test CustomCallHandler."""
    obj = mod.CustomCallHandler(name="CustomCallHandler", value=1)
    assert isinstance(obj, mod.CustomCallHandler)
    assert obj.name == "CustomCallHandler"
    assert obj.value == 1


def test_class_CustomCallTargetTraits() -> None:
    """Test CustomCallTargetTraits."""
    obj = mod.CustomCallTargetTraits(name="CustomCallTargetTraits", value=1)
    assert isinstance(obj, mod.CustomCallTargetTraits)
    assert obj.name == "CustomCallTargetTraits"
    assert obj.value == 1


def test_class_Device() -> None:
    """Test Device."""
    obj = mod.Device(name="Device", value=1)
    assert isinstance(obj, mod.Device)
    assert obj.name == "Device"
    assert obj.value == 1


def test_class_DeviceAssignment() -> None:
    """Test DeviceAssignment."""
    obj = mod.DeviceAssignment(name="DeviceAssignment", value=1)
    assert isinstance(obj, mod.DeviceAssignment)
    assert obj.name == "DeviceAssignment"
    assert obj.value == 1


def test_class_DeviceList() -> None:
    """Test DeviceList."""
    obj = mod.DeviceList(name="DeviceList", value=1)
    assert isinstance(obj, mod.DeviceList)
    assert obj.name == "DeviceList"
    assert obj.value == 1


def test_class_DeviceTopology() -> None:
    """Test DeviceTopology."""
    obj = mod.DeviceTopology(name="DeviceTopology", value=1)
    assert isinstance(obj, mod.DeviceTopology)
    assert obj.name == "DeviceTopology"
    assert obj.value == 1


def test_class_DotDimensionNumbers() -> None:
    """Test DotDimensionNumbers."""
    obj = mod.DotDimensionNumbers(name="DotDimensionNumbers", value=1)
    assert isinstance(obj, mod.DotDimensionNumbers)
    assert obj.name == "DotDimensionNumbers"
    assert obj.value == 1


def test_class_FftType() -> None:
    """Test FftType."""
    obj = mod.FftType(name="FftType", value=1)
    assert isinstance(obj, mod.FftType)
    assert obj.name == "FftType"
    assert obj.value == 1


def test_class_Frame() -> None:
    """Test Frame."""
    obj = mod.Frame(name="Frame", value=1)
    assert isinstance(obj, mod.Frame)
    assert obj.name == "Frame"
    assert obj.value == 1


def test_class_GSPMDSharding() -> None:
    """Test GSPMDSharding."""
    obj = mod.GSPMDSharding(name="GSPMDSharding", value=1)
    assert isinstance(obj, mod.GSPMDSharding)
    assert obj.name == "GSPMDSharding"
    assert obj.value == 1


def test_class_GatherDimensionNumbers() -> None:
    """Test GatherDimensionNumbers."""
    obj = mod.GatherDimensionNumbers(name="GatherDimensionNumbers", value=1)
    assert isinstance(obj, mod.GatherDimensionNumbers)
    assert obj.name == "GatherDimensionNumbers"
    assert obj.value == 1


def test_class_HloSharding() -> None:
    """Test HloSharding."""
    obj = mod.HloSharding(name="HloSharding", value=1)
    assert isinstance(obj, mod.HloSharding)
    assert obj.name == "HloSharding"
    assert obj.value == 1


def test_class_HostBufferSemantics() -> None:
    """Test HostBufferSemantics."""
    obj = mod.HostBufferSemantics(name="HostBufferSemantics", value=1)
    assert isinstance(obj, mod.HostBufferSemantics)
    assert obj.name == "HostBufferSemantics"
    assert obj.value == 1


def test_class_Layout() -> None:
    """Test Layout."""
    obj = mod.Layout(name="Layout", value=1)
    assert isinstance(obj, mod.Layout)
    assert obj.name == "Layout"
    assert obj.value == 1


def test_class_LoadedExecutable() -> None:
    """Test LoadedExecutable."""
    obj = mod.LoadedExecutable(name="LoadedExecutable", value=1)
    assert isinstance(obj, mod.LoadedExecutable)
    assert obj.name == "LoadedExecutable"
    assert obj.value == 1


def test_LoadedExecutable_execute() -> None:
    """Test LoadedExecutable_execute."""
    with pytest.raises(AttributeError):
        mod.LoadedExecutable_execute()


def test_LoadedExecutable_execute_with_token() -> None:
    """Test LoadedExecutable_execute_with_token."""
    with pytest.raises(AttributeError):
        mod.LoadedExecutable_execute_with_token()


def test_class_Mapping() -> None:
    """Test Mapping."""
    obj = mod.Mapping(name="Mapping", value=1)
    assert isinstance(obj, mod.Mapping)
    assert obj.name == "Mapping"
    assert obj.value == 1


def test_class_Memory() -> None:
    """Test Memory."""
    obj = mod.Memory(name="Memory", value=1)
    assert isinstance(obj, mod.Memory)
    assert obj.name == "Memory"
    assert obj.value == 1


def test_class_NamedSharding() -> None:
    """Test NamedSharding."""
    obj = mod.NamedSharding(name="NamedSharding", value=1)
    assert isinstance(obj, mod.NamedSharding)
    assert obj.name == "NamedSharding"
    assert obj.value == 1


def test_class_OpMetadata() -> None:
    """Test OpMetadata."""
    obj = mod.OpMetadata(name="OpMetadata", value=1)
    assert isinstance(obj, mod.OpMetadata)
    assert obj.name == "OpMetadata"
    assert obj.value == 1


def test_class_OpSharding() -> None:
    """Test OpSharding."""
    obj = mod.OpSharding(name="OpSharding", value=1)
    assert isinstance(obj, mod.OpSharding)
    assert obj.name == "OpSharding"
    assert obj.value == 1


def test_class_PaddingConfig() -> None:
    """Test PaddingConfig."""
    obj = mod.PaddingConfig(name="PaddingConfig", value=1)
    assert isinstance(obj, mod.PaddingConfig)
    assert obj.name == "PaddingConfig"
    assert obj.value == 1


def test_class_PaddingConfigDimension() -> None:
    """Test PaddingConfigDimension."""
    obj = mod.PaddingConfigDimension(name="PaddingConfigDimension", value=1)
    assert isinstance(obj, mod.PaddingConfigDimension)
    assert obj.name == "PaddingConfigDimension"
    assert obj.value == 1


def test_class_PaddingType() -> None:
    """Test PaddingType."""
    obj = mod.PaddingType(name="PaddingType", value=1)
    assert isinstance(obj, mod.PaddingType)
    assert obj.name == "PaddingType"
    assert obj.value == 1


def test_class_PjRtLayout() -> None:
    """Test PjRtLayout."""
    obj = mod.PjRtLayout(name="PjRtLayout", value=1)
    assert isinstance(obj, mod.PjRtLayout)
    assert obj.name == "PjRtLayout"
    assert obj.value == 1


def test_class_PmapSharding() -> None:
    """Test PmapSharding."""
    obj = mod.PmapSharding(name="PmapSharding", value=1)
    assert isinstance(obj, mod.PmapSharding)
    assert obj.name == "PmapSharding"
    assert obj.value == 1


def test_class_PrecisionConfig() -> None:
    """Test PrecisionConfig."""
    obj = mod.PrecisionConfig(name="PrecisionConfig", value=1)
    assert isinstance(obj, mod.PrecisionConfig)
    assert obj.name == "PrecisionConfig"
    assert obj.value == 1


def test_class_PrimitiveType() -> None:
    """Test PrimitiveType."""
    obj = mod.PrimitiveType(name="PrimitiveType", value=1)
    assert isinstance(obj, mod.PrimitiveType)
    assert obj.name == "PrimitiveType"
    assert obj.value == 1


def test_class_ProgramShape() -> None:
    """Test ProgramShape."""
    obj = mod.ProgramShape(name="ProgramShape", value=1)
    assert isinstance(obj, mod.ProgramShape)
    assert obj.name == "ProgramShape"
    assert obj.value == 1


def test_class_Protocol() -> None:
    """Test Protocol."""
    obj = mod.Protocol(name="Protocol", value=1)
    assert isinstance(obj, mod.Protocol)
    assert obj.name == "Protocol"
    assert obj.value == 1


def test_class_ReplicaGroup() -> None:
    """Test ReplicaGroup."""
    obj = mod.ReplicaGroup(name="ReplicaGroup", value=1)
    assert isinstance(obj, mod.ReplicaGroup)
    assert obj.name == "ReplicaGroup"
    assert obj.value == 1


def test_class_ScatterDimensionNumbers() -> None:
    """Test ScatterDimensionNumbers."""
    obj = mod.ScatterDimensionNumbers(name="ScatterDimensionNumbers", value=1)
    assert isinstance(obj, mod.ScatterDimensionNumbers)
    assert obj.name == "ScatterDimensionNumbers"
    assert obj.value == 1


def test_class_Sequence() -> None:
    """Test Sequence."""
    obj = mod.Sequence(name="Sequence", value=1)
    assert isinstance(obj, mod.Sequence)
    assert obj.name == "Sequence"
    assert obj.value == 1


def test_class_Shape() -> None:
    """Test Shape."""
    obj = mod.Shape(name="Shape", value=1)
    assert isinstance(obj, mod.Shape)
    assert obj.name == "Shape"
    assert obj.value == 1


def test_class_ShapeIndex() -> None:
    """Test ShapeIndex."""
    obj = mod.ShapeIndex(name="ShapeIndex", value=1)
    assert isinstance(obj, mod.ShapeIndex)
    assert obj.name == "ShapeIndex"
    assert obj.value == 1


def test_class_Sharding() -> None:
    """Test Sharding."""
    obj = mod.Sharding(name="Sharding", value=1)
    assert isinstance(obj, mod.Sharding)
    assert obj.name == "Sharding"
    assert obj.value == 1


def test_class_SingleDeviceSharding() -> None:
    """Test SingleDeviceSharding."""
    obj = mod.SingleDeviceSharding(name="SingleDeviceSharding", value=1)
    assert isinstance(obj, mod.SingleDeviceSharding)
    assert obj.name == "SingleDeviceSharding"
    assert obj.value == 1


def test_class_Traceback() -> None:
    """Test Traceback."""
    obj = mod.Traceback(name="Traceback", value=1)
    assert isinstance(obj, mod.Traceback)
    assert obj.name == "Traceback"
    assert obj.value == 1


def test_class_XlaBuilder() -> None:
    """Test XlaBuilder."""
    obj = mod.XlaBuilder(name="XlaBuilder", value=1)
    assert isinstance(obj, mod.XlaBuilder)
    assert obj.name == "XlaBuilder"
    assert obj.value == 1


def test_class_XlaComputation() -> None:
    """Test XlaComputation."""
    obj = mod.XlaComputation(name="XlaComputation", value=1)
    assert isinstance(obj, mod.XlaComputation)
    assert obj.name == "XlaComputation"
    assert obj.value == 1


def test_class_XlaOp() -> None:
    """Test XlaOp."""
    obj = mod.XlaOp(name="XlaOp", value=1)
    assert isinstance(obj, mod.XlaOp)
    assert obj.name == "XlaOp"
    assert obj.value == 1


def test_class_XlaRuntimeError() -> None:
    """Test XlaRuntimeError."""
    obj = mod.XlaRuntimeError(name="XlaRuntimeError", value=1)
    assert isinstance(obj, mod.XlaRuntimeError)
    assert obj.name == "XlaRuntimeError"
    assert obj.value == 1


def test_class_bfloat16() -> None:
    """Test bfloat16."""
    obj = mod.bfloat16(name="bfloat16", value=1)
    assert isinstance(obj, mod.bfloat16)
    assert obj.name == "bfloat16"
    assert obj.value == 1


def test_dtype_to_etype() -> None:
    """Test dtype_to_etype."""
    with pytest.raises(AttributeError):
        mod.dtype_to_etype()


def test_execute_with_python_values() -> None:
    """Test execute_with_python_values."""
    with pytest.raises(AttributeError):
        mod.execute_with_python_values()


def test_execute_with_python_values_replicated() -> None:
    """Test execute_with_python_values_replicated."""
    with pytest.raises(AttributeError):
        mod.execute_with_python_values_replicated()


def test_class_float8_e4m3b11fnuz() -> None:
    """Test float8_e4m3b11fnuz."""
    obj = mod.float8_e4m3b11fnuz(name="float8_e4m3b11fnuz", value=1)
    assert isinstance(obj, mod.float8_e4m3b11fnuz)
    assert obj.name == "float8_e4m3b11fnuz"
    assert obj.value == 1


def test_class_float8_e4m3fn() -> None:
    """Test float8_e4m3fn."""
    obj = mod.float8_e4m3fn(name="float8_e4m3fn", value=1)
    assert isinstance(obj, mod.float8_e4m3fn)
    assert obj.name == "float8_e4m3fn"
    assert obj.value == 1


def test_class_float8_e4m3fnuz() -> None:
    """Test float8_e4m3fnuz."""
    obj = mod.float8_e4m3fnuz(name="float8_e4m3fnuz", value=1)
    assert isinstance(obj, mod.float8_e4m3fnuz)
    assert obj.name == "float8_e4m3fnuz"
    assert obj.value == 1


def test_class_float8_e5m2() -> None:
    """Test float8_e5m2."""
    obj = mod.float8_e5m2(name="float8_e5m2", value=1)
    assert isinstance(obj, mod.float8_e5m2)
    assert obj.name == "float8_e5m2"
    assert obj.value == 1


def test_class_float8_e5m2fnuz() -> None:
    """Test float8_e5m2fnuz."""
    obj = mod.float8_e5m2fnuz(name="float8_e5m2fnuz", value=1)
    assert isinstance(obj, mod.float8_e5m2fnuz)
    assert obj.name == "float8_e5m2fnuz"
    assert obj.value == 1


def test_generate_pjrt_gpu_plugin_options() -> None:
    """Test generate_pjrt_gpu_plugin_options."""
    with pytest.raises(AttributeError):
        mod.generate_pjrt_gpu_plugin_options()


def test_heap_profile() -> None:
    """Test heap_profile."""
    with pytest.raises(AttributeError):
        mod.heap_profile()


def test_initialize_pjrt_plugin() -> None:
    """Test initialize_pjrt_plugin."""
    with pytest.raises(AttributeError):
        mod.initialize_pjrt_plugin()


def test_load_pjrt_plugin_dynamically() -> None:
    """Test load_pjrt_plugin_dynamically."""
    with pytest.raises(AttributeError):
        mod.load_pjrt_plugin_dynamically()


def test_load_pjrt_plugin_with_c_api() -> None:
    """Test load_pjrt_plugin_with_c_api."""
    with pytest.raises(AttributeError):
        mod.load_pjrt_plugin_with_c_api()


def test_make_c_api_client() -> None:
    """Test make_c_api_client."""
    with pytest.raises(AttributeError):
        mod.make_c_api_client()


def test_make_c_api_device_topology() -> None:
    """Test make_c_api_device_topology."""
    with pytest.raises(AttributeError):
        mod.make_c_api_device_topology()


def test_make_convolution_dimension_numbers() -> None:
    """Test make_convolution_dimension_numbers."""
    with pytest.raises(AttributeError):
        mod.make_convolution_dimension_numbers()


def test_make_cpu_client() -> None:
    """Test make_cpu_client."""
    with pytest.raises(AttributeError):
        mod.make_cpu_client()


def test_make_dot_dimension_numbers() -> None:
    """Test make_dot_dimension_numbers."""
    with pytest.raises(AttributeError):
        mod.make_dot_dimension_numbers()


def test_make_gpu_client() -> None:
    """Test make_gpu_client."""
    with pytest.raises(AttributeError):
        mod.make_gpu_client()


def test_make_padding_config() -> None:
    """Test make_padding_config."""
    with pytest.raises(AttributeError):
        mod.make_padding_config()


def test_make_replica_groups() -> None:
    """Test make_replica_groups."""
    with pytest.raises(AttributeError):
        mod.make_replica_groups()


def test_make_tfrt_tpu_c_api_client() -> None:
    """Test make_tfrt_tpu_c_api_client."""
    with pytest.raises(AttributeError):
        mod.make_tfrt_tpu_c_api_client()


def test_make_tfrt_tpu_c_api_device_topology() -> None:
    """Test make_tfrt_tpu_c_api_device_topology."""
    with pytest.raises(AttributeError):
        mod.make_tfrt_tpu_c_api_device_topology()


def test_make_tpu_client() -> None:
    """Test make_tpu_client."""
    with pytest.raises(AttributeError):
        mod.make_tpu_client()


def test_pjrt_plugin_initialized() -> None:
    """Test pjrt_plugin_initialized."""
    with pytest.raises(AttributeError):
        mod.pjrt_plugin_initialized()


def test_pjrt_plugin_loaded() -> None:
    """Test pjrt_plugin_loaded."""
    with pytest.raises(AttributeError):
        mod.pjrt_plugin_loaded()


def test_register_custom_call_handler() -> None:
    """Test register_custom_call_handler."""
    with pytest.raises(AttributeError):
        mod.register_custom_call_handler()


def test_register_custom_call_target() -> None:
    """Test register_custom_call_target."""
    with pytest.raises(AttributeError):
        mod.register_custom_call_target()


def test_shape_from_pyval() -> None:
    """Test shape_from_pyval."""
    with pytest.raises(AttributeError):
        mod.shape_from_pyval()


def test_tracebacks() -> None:
    """Test tracebacks."""
    with pytest.raises(AttributeError):
        mod.tracebacks()


def test_window_padding_type_to_pad_values() -> None:
    """Test window_padding_type_to_pad_values."""
    with pytest.raises(AttributeError):
        mod.window_padding_type_to_pad_values()
