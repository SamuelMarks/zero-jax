"""Module xla_client."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class Client:
    """Frontend class for ``Client``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


class FftType:
    """Frontend class for ``FftType``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


class Frame:
    """Frontend class for ``Frame``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


class Layout:
    """Frontend class for ``Layout``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


class Mapping:
    """Frontend class for ``Mapping``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


class Memory:
    """Frontend class for ``Memory``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


class Shape:
    """Frontend class for ``Shape``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


class XlaOp:
    """Frontend class for ``XlaOp``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


class ArrayImpl:
    """Stub class for ArrayImpl."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ArrayImpl.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class CompileOptions:
    """Stub class for CompileOptions."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize CompileOptions.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ConvolutionDimensionNumbers:
    """Stub class for ConvolutionDimensionNumbers."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ConvolutionDimensionNumbers.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class CurrentSourceInfoMetadata:
    """Stub class for CurrentSourceInfoMetadata."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize CurrentSourceInfoMetadata.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class CustomCallHandler:
    """Stub class for CustomCallHandler."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize CustomCallHandler.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class CustomCallTargetTraits:
    """Stub class for CustomCallTargetTraits."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize CustomCallTargetTraits.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class DTYPE_TO_XLA_ELEMENT_TYPE:
    """Stub class for DTYPE_TO_XLA_ELEMENT_TYPE."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize DTYPE_TO_XLA_ELEMENT_TYPE.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Device:
    """Stub class for Device."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize Device.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class DeviceAssignment:
    """Stub class for DeviceAssignment."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize DeviceAssignment.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class DeviceList:
    """Stub class for DeviceList."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize DeviceList.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class DeviceTopology:
    """Stub class for DeviceTopology."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize DeviceTopology.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class DotDimensionNumbers:
    """Stub class for DotDimensionNumbers."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize DotDimensionNumbers.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class GSPMDSharding:
    """Stub class for GSPMDSharding."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize GSPMDSharding.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class GatherDimensionNumbers:
    """Stub class for GatherDimensionNumbers."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize GatherDimensionNumbers.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class HloSharding:
    """Stub class for HloSharding."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize HloSharding.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class HostBufferSemantics:
    """Stub class for HostBufferSemantics."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize HostBufferSemantics.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class LoadedExecutable:
    """Stub class for LoadedExecutable."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize LoadedExecutable.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class LoadedExecutable_execute:
    """Stub class for LoadedExecutable_execute."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize LoadedExecutable_execute.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class LoadedExecutable_execute_with_token:
    """Stub class for LoadedExecutable_execute_with_token."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize LoadedExecutable_execute_with_token.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class NamedSharding:
    """Stub class for NamedSharding."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize NamedSharding.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class OpMetadata:
    """Stub class for OpMetadata."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize OpMetadata.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class OpSharding:
    """Stub class for OpSharding."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize OpSharding.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class PaddingConfig:
    """Stub class for PaddingConfig."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize PaddingConfig.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class PaddingConfigDimension:
    """Stub class for PaddingConfigDimension."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize PaddingConfigDimension.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class PaddingType:
    """Stub class for PaddingType."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize PaddingType.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class PjRtLayout:
    """Stub class for PjRtLayout."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize PjRtLayout.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class PmapSharding:
    """Stub class for PmapSharding."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize PmapSharding.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class PrecisionConfig:
    """Stub class for PrecisionConfig."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize PrecisionConfig.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class PrimitiveType:
    """Stub class for PrimitiveType."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize PrimitiveType.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ProgramShape:
    """Stub class for ProgramShape."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ProgramShape.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Protocol:
    """Stub class for Protocol."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize Protocol.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ReplicaGroup:
    """Stub class for ReplicaGroup."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ReplicaGroup.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ScatterDimensionNumbers:
    """Stub class for ScatterDimensionNumbers."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ScatterDimensionNumbers.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Sequence:
    """Stub class for Sequence."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize Sequence.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ShapeIndex:
    """Stub class for ShapeIndex."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ShapeIndex.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Sharding:
    """Stub class for Sharding."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize Sharding.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class SingleDeviceSharding:
    """Stub class for SingleDeviceSharding."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize SingleDeviceSharding.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Traceback:
    """Stub class for Traceback."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize Traceback.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Union:
    """Stub class for Union."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize Union.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class XLA_ELEMENT_TYPE_TO_DTYPE:
    """Stub class for XLA_ELEMENT_TYPE_TO_DTYPE."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize XLA_ELEMENT_TYPE_TO_DTYPE.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class XlaBuilder:
    """Stub class for XlaBuilder."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize XlaBuilder.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class XlaComputation:
    """Stub class for XlaComputation."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize XlaComputation.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


class XlaRuntimeError:
    """Stub class for XlaRuntimeError."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize XlaRuntimeError.

        Args:
            *args: Positional configuration arguments.
            **kwargs: Keyword configuration arguments.
        """
        self.args = args
        self.kwargs = kwargs


def annotations(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route annotations to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.annotations(*args, **kwargs)


def array_result_handler(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for array_result_handler.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def batched_block_until_ready(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for batched_block_until_ready.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def batched_copy_array_to_devices_with_sharding(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for batched_copy_array_to_devices_with_sharding.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def batched_device_put(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for batched_device_put.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def bfloat16(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for bfloat16.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def check_and_canonicalize_memory_kind(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for check_and_canonicalize_memory_kind.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def custom_call_targets(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for custom_call_targets.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def dtype_to_etype(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route dtype_to_etype to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.dtype_to_etype(*args, **kwargs)


def encode_inspect_sharding_callback(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for encode_inspect_sharding_callback.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def execute_with_python_values(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route execute_with_python_values to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.execute_with_python_values(*args, **kwargs)


def execute_with_python_values_replicated(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route execute_with_python_values_replicated to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.execute_with_python_values_replicated(*args, **kwargs)


def float8_e4m3b11fnuz(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for float8_e4m3b11fnuz.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def float8_e4m3fn(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for float8_e4m3fn.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def float8_e4m3fnuz(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for float8_e4m3fnuz.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def float8_e5m2(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for float8_e5m2.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def float8_e5m2fnuz(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for float8_e5m2fnuz.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def generate_pjrt_gpu_plugin_options(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route generate_pjrt_gpu_plugin_options to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.generate_pjrt_gpu_plugin_options(*args, **kwargs)


def get_topology_for_devices(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for get_topology_for_devices.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def heap_profile(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route heap_profile to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.heap_profile(*args, **kwargs)


def initialize_pjrt_plugin(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route initialize_pjrt_plugin to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.initialize_pjrt_plugin(*args, **kwargs)


def load_pjrt_plugin_dynamically(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route load_pjrt_plugin_dynamically to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.load_pjrt_plugin_dynamically(*args, **kwargs)


def load_pjrt_plugin_with_c_api(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route load_pjrt_plugin_with_c_api to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.load_pjrt_plugin_with_c_api(*args, **kwargs)


def logger(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for logger.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def make_c_api_client(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_c_api_client to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_c_api_client(*args, **kwargs)


def make_c_api_device_topology(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_c_api_device_topology to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_c_api_device_topology(*args, **kwargs)


def make_convolution_dimension_numbers(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_convolution_dimension_numbers to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_convolution_dimension_numbers(*args, **kwargs)


def make_cpu_client(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_cpu_client to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_cpu_client(*args, **kwargs)


def make_dot_dimension_numbers(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_dot_dimension_numbers to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_dot_dimension_numbers(*args, **kwargs)


def make_gpu_client(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_gpu_client to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_gpu_client(*args, **kwargs)


def make_padding_config(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_padding_config to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_padding_config(*args, **kwargs)


def make_replica_groups(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_replica_groups to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_replica_groups(*args, **kwargs)


def make_tfrt_tpu_c_api_client(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_tfrt_tpu_c_api_client to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_tfrt_tpu_c_api_client(*args, **kwargs)


def make_tfrt_tpu_c_api_device_topology(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_tfrt_tpu_c_api_device_topology to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_tfrt_tpu_c_api_device_topology(*args, **kwargs)


def make_tpu_client(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route make_tpu_client to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.make_tpu_client(*args, **kwargs)


def mlir_api_version(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for mlir_api_version.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def pjrt_plugin_initialized(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route pjrt_plugin_initialized to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.pjrt_plugin_initialized(*args, **kwargs)


def pjrt_plugin_loaded(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route pjrt_plugin_loaded to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.pjrt_plugin_loaded(*args, **kwargs)


def register_custom_call_handler(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route register_custom_call_handler to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.register_custom_call_handler(*args, **kwargs)


def register_custom_call_partitioner(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for register_custom_call_partitioner.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None


def register_custom_call_target(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route register_custom_call_target to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.register_custom_call_target(*args, **kwargs)


def shape_from_pyval(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route shape_from_pyval to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.shape_from_pyval(*args, **kwargs)


def tracebacks(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route tracebacks to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.tracebacks(*args, **kwargs)


def weakref_lru_cache(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route weakref_lru_cache to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.weakref_lru_cache(*args, **kwargs)


def window_padding_type_to_pad_values(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Route window_padding_type_to_pad_values to the compiler proxy operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: Result of the compiler operation.
    """
    return _ops.window_padding_type_to_pad_values(*args, **kwargs)


def xla_platform_names(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for xla_platform_names.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None
