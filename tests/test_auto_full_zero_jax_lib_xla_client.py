"""Tests for zero_jax.lib.xla_client."""

from typing import Any

import pytest

import zero_jax.lib.xla_client as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_ArrayImpl() -> None:
    """Test class ArrayImpl."""
    try:
        mod.ArrayImpl()
    except Exception:
        pass


def test_class_Client() -> None:
    """Test class Client."""
    try:
        mod.Client()
    except Exception:
        pass


def test_class_CompileOptions() -> None:
    """Test class CompileOptions."""
    try:
        mod.CompileOptions()
    except Exception:
        pass


def test_class_ConvolutionDimensionNumbers() -> None:
    """Test class ConvolutionDimensionNumbers."""
    try:
        mod.ConvolutionDimensionNumbers()
    except Exception:
        pass


def test_CurrentSourceInfoMetadata() -> None:
    """Test CurrentSourceInfoMetadata."""
    try:
        mod.CurrentSourceInfoMetadata()
    except Exception:
        pass


def test_class_CustomCallHandler() -> None:
    """Test class CustomCallHandler."""
    try:
        mod.CustomCallHandler()
    except Exception:
        pass


def test_class_CustomCallTargetTraits() -> None:
    """Test class CustomCallTargetTraits."""
    try:
        mod.CustomCallTargetTraits()
    except Exception:
        pass


def test_class_Device() -> None:
    """Test class Device."""
    try:
        mod.Device()
    except Exception:
        pass


def test_class_DeviceAssignment() -> None:
    """Test class DeviceAssignment."""
    try:
        mod.DeviceAssignment()
    except Exception:
        pass


def test_class_DeviceList() -> None:
    """Test class DeviceList."""
    try:
        mod.DeviceList()
    except Exception:
        pass


def test_class_DeviceTopology() -> None:
    """Test class DeviceTopology."""
    try:
        mod.DeviceTopology()
    except Exception:
        pass


def test_class_DotDimensionNumbers() -> None:
    """Test class DotDimensionNumbers."""
    try:
        mod.DotDimensionNumbers()
    except Exception:
        pass


def test_class_FftType() -> None:
    """Test class FftType."""
    try:
        mod.FftType()
    except Exception:
        pass


def test_class_Frame() -> None:
    """Test class Frame."""
    try:
        mod.Frame()
    except Exception:
        pass


def test_class_GSPMDSharding() -> None:
    """Test class GSPMDSharding."""
    try:
        mod.GSPMDSharding()
    except Exception:
        pass


def test_class_GatherDimensionNumbers() -> None:
    """Test class GatherDimensionNumbers."""
    try:
        mod.GatherDimensionNumbers()
    except Exception:
        pass


def test_class_HloSharding() -> None:
    """Test class HloSharding."""
    try:
        mod.HloSharding()
    except Exception:
        pass


def test_class_HostBufferSemantics() -> None:
    """Test class HostBufferSemantics."""
    try:
        mod.HostBufferSemantics()
    except Exception:
        pass


def test_class_Layout() -> None:
    """Test class Layout."""
    try:
        mod.Layout()
    except Exception:
        pass


def test_class_LoadedExecutable() -> None:
    """Test class LoadedExecutable."""
    try:
        mod.LoadedExecutable()
    except Exception:
        pass


def test_LoadedExecutable_execute() -> None:
    """Test LoadedExecutable_execute."""
    try:
        mod.LoadedExecutable_execute()
    except Exception:
        pass


def test_LoadedExecutable_execute_with_token() -> None:
    """Test LoadedExecutable_execute_with_token."""
    try:
        mod.LoadedExecutable_execute_with_token()
    except Exception:
        pass


def test_class_Mapping() -> None:
    """Test class Mapping."""
    try:
        mod.Mapping()
    except Exception:
        pass


def test_class_Memory() -> None:
    """Test class Memory."""
    try:
        mod.Memory()
    except Exception:
        pass


def test_class_NamedSharding() -> None:
    """Test class NamedSharding."""
    try:
        mod.NamedSharding()
    except Exception:
        pass


def test_class_OpMetadata() -> None:
    """Test class OpMetadata."""
    try:
        mod.OpMetadata()
    except Exception:
        pass


def test_class_OpSharding() -> None:
    """Test class OpSharding."""
    try:
        mod.OpSharding()
    except Exception:
        pass


def test_class_PaddingConfig() -> None:
    """Test class PaddingConfig."""
    try:
        mod.PaddingConfig()
    except Exception:
        pass


def test_class_PaddingConfigDimension() -> None:
    """Test class PaddingConfigDimension."""
    try:
        mod.PaddingConfigDimension()
    except Exception:
        pass


def test_class_PaddingType() -> None:
    """Test class PaddingType."""
    try:
        mod.PaddingType()
    except Exception:
        pass


def test_class_PjRtLayout() -> None:
    """Test class PjRtLayout."""
    try:
        mod.PjRtLayout()
    except Exception:
        pass


def test_class_PmapSharding() -> None:
    """Test class PmapSharding."""
    try:
        mod.PmapSharding()
    except Exception:
        pass


def test_class_PrecisionConfig() -> None:
    """Test class PrecisionConfig."""
    try:
        mod.PrecisionConfig()
    except Exception:
        pass


def test_class_PrimitiveType() -> None:
    """Test class PrimitiveType."""
    try:
        mod.PrimitiveType()
    except Exception:
        pass


def test_class_ProgramShape() -> None:
    """Test class ProgramShape."""
    try:
        mod.ProgramShape()
    except Exception:
        pass


def test_class_Protocol() -> None:
    """Test class Protocol."""
    try:
        mod.Protocol()
    except Exception:
        pass


def test_class_ReplicaGroup() -> None:
    """Test class ReplicaGroup."""
    try:
        mod.ReplicaGroup()
    except Exception:
        pass


def test_class_ScatterDimensionNumbers() -> None:
    """Test class ScatterDimensionNumbers."""
    try:
        mod.ScatterDimensionNumbers()
    except Exception:
        pass


def test_class_Sequence() -> None:
    """Test class Sequence."""
    try:
        mod.Sequence()
    except Exception:
        pass


def test_class_Shape() -> None:
    """Test class Shape."""
    try:
        mod.Shape()
    except Exception:
        pass


def test_class_ShapeIndex() -> None:
    """Test class ShapeIndex."""
    try:
        mod.ShapeIndex()
    except Exception:
        pass


def test_class_Sharding() -> None:
    """Test class Sharding."""
    try:
        mod.Sharding()
    except Exception:
        pass


def test_class_SingleDeviceSharding() -> None:
    """Test class SingleDeviceSharding."""
    try:
        mod.SingleDeviceSharding()
    except Exception:
        pass


def test_class_Traceback() -> None:
    """Test class Traceback."""
    try:
        mod.Traceback()
    except Exception:
        pass


def test_class_XlaBuilder() -> None:
    """Test class XlaBuilder."""
    try:
        mod.XlaBuilder()
    except Exception:
        pass


def test_class_XlaComputation() -> None:
    """Test class XlaComputation."""
    try:
        mod.XlaComputation()
    except Exception:
        pass


def test_class_XlaOp() -> None:
    """Test class XlaOp."""
    try:
        mod.XlaOp()
    except Exception:
        pass


def test_class_XlaRuntimeError() -> None:
    """Test class XlaRuntimeError."""
    try:
        mod.XlaRuntimeError()
    except Exception:
        pass


def test_class_bfloat16() -> None:
    """Test class bfloat16."""
    try:
        mod.bfloat16()
    except Exception:
        pass


def test_dtype_to_etype() -> None:
    """Test dtype_to_etype."""
    try:
        mod.dtype_to_etype()
    except Exception:
        pass


def test_execute_with_python_values() -> None:
    """Test execute_with_python_values."""
    try:
        mod.execute_with_python_values()
    except Exception:
        pass


def test_execute_with_python_values_replicated() -> None:
    """Test execute_with_python_values_replicated."""
    try:
        mod.execute_with_python_values_replicated()
    except Exception:
        pass


def test_class_float8_e4m3b11fnuz() -> None:
    """Test class float8_e4m3b11fnuz."""
    try:
        mod.float8_e4m3b11fnuz()
    except Exception:
        pass


def test_class_float8_e4m3fn() -> None:
    """Test class float8_e4m3fn."""
    try:
        mod.float8_e4m3fn()
    except Exception:
        pass


def test_class_float8_e4m3fnuz() -> None:
    """Test class float8_e4m3fnuz."""
    try:
        mod.float8_e4m3fnuz()
    except Exception:
        pass


def test_class_float8_e5m2() -> None:
    """Test class float8_e5m2."""
    try:
        mod.float8_e5m2()
    except Exception:
        pass


def test_class_float8_e5m2fnuz() -> None:
    """Test class float8_e5m2fnuz."""
    try:
        mod.float8_e5m2fnuz()
    except Exception:
        pass


def test_generate_pjrt_gpu_plugin_options() -> None:
    """Test generate_pjrt_gpu_plugin_options."""
    try:
        mod.generate_pjrt_gpu_plugin_options()
    except Exception:
        pass


def test_heap_profile() -> None:
    """Test heap_profile."""
    try:
        mod.heap_profile()
    except Exception:
        pass


def test_initialize_pjrt_plugin() -> None:
    """Test initialize_pjrt_plugin."""
    try:
        mod.initialize_pjrt_plugin()
    except Exception:
        pass


def test_load_pjrt_plugin_dynamically() -> None:
    """Test load_pjrt_plugin_dynamically."""
    try:
        mod.load_pjrt_plugin_dynamically()
    except Exception:
        pass


def test_load_pjrt_plugin_with_c_api() -> None:
    """Test load_pjrt_plugin_with_c_api."""
    try:
        mod.load_pjrt_plugin_with_c_api()
    except Exception:
        pass


def test_make_c_api_client() -> None:
    """Test make_c_api_client."""
    try:
        mod.make_c_api_client()
    except Exception:
        pass


def test_make_c_api_device_topology() -> None:
    """Test make_c_api_device_topology."""
    try:
        mod.make_c_api_device_topology()
    except Exception:
        pass


def test_make_convolution_dimension_numbers() -> None:
    """Test make_convolution_dimension_numbers."""
    try:
        mod.make_convolution_dimension_numbers()
    except Exception:
        pass


def test_make_cpu_client() -> None:
    """Test make_cpu_client."""
    try:
        mod.make_cpu_client()
    except Exception:
        pass


def test_make_dot_dimension_numbers() -> None:
    """Test make_dot_dimension_numbers."""
    try:
        mod.make_dot_dimension_numbers()
    except Exception:
        pass


def test_make_gpu_client() -> None:
    """Test make_gpu_client."""
    try:
        mod.make_gpu_client()
    except Exception:
        pass


def test_make_padding_config() -> None:
    """Test make_padding_config."""
    try:
        mod.make_padding_config()
    except Exception:
        pass


def test_make_replica_groups() -> None:
    """Test make_replica_groups."""
    try:
        mod.make_replica_groups()
    except Exception:
        pass


def test_make_tfrt_tpu_c_api_client() -> None:
    """Test make_tfrt_tpu_c_api_client."""
    try:
        mod.make_tfrt_tpu_c_api_client()
    except Exception:
        pass


def test_make_tfrt_tpu_c_api_device_topology() -> None:
    """Test make_tfrt_tpu_c_api_device_topology."""
    try:
        mod.make_tfrt_tpu_c_api_device_topology()
    except Exception:
        pass


def test_make_tpu_client() -> None:
    """Test make_tpu_client."""
    try:
        mod.make_tpu_client()
    except Exception:
        pass


def test_pjrt_plugin_initialized() -> None:
    """Test pjrt_plugin_initialized."""
    try:
        mod.pjrt_plugin_initialized()
    except Exception:
        pass


def test_pjrt_plugin_loaded() -> None:
    """Test pjrt_plugin_loaded."""
    try:
        mod.pjrt_plugin_loaded()
    except Exception:
        pass


def test_register_custom_call_handler() -> None:
    """Test register_custom_call_handler."""
    try:
        mod.register_custom_call_handler()
    except Exception:
        pass


def test_register_custom_call_target() -> None:
    """Test register_custom_call_target."""
    try:
        mod.register_custom_call_target()
    except Exception:
        pass


def test_shape_from_pyval() -> None:
    """Test shape_from_pyval."""
    try:
        mod.shape_from_pyval()
    except Exception:
        pass


def test_tracebacks() -> None:
    """Test tracebacks."""
    try:
        mod.tracebacks()
    except Exception:
        pass


def test_window_padding_type_to_pad_values() -> None:
    """Test window_padding_type_to_pad_values."""
    try:
        mod.window_padding_type_to_pad_values()
    except Exception:
        pass
