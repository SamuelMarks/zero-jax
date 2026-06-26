"""Mock implementation for jax.lib.xla_client."""

from typing import Any

DTYPE_TO_XLA_ELEMENT_TYPE: Any = None
Union: Any = None
XLA_ELEMENT_TYPE_TO_DTYPE: Any = None
annotations: Any = None
array_result_handler: Any = None
batched_block_until_ready: Any = None
batched_copy_array_to_devices_with_sharding: Any = None
batched_device_put: Any = None
check_and_canonicalize_memory_kind: Any = None
custom_call_targets: Any = None
encode_inspect_sharding_callback: Any = None
get_topology_for_devices: Any = None
logger: Any = None
mlir_api_version: Any = None
register_custom_call_partitioner: Any = None
weakref_lru_cache: Any = None
xla_platform_names: Any = None


class ArrayImpl:
    """Mock implementation for ArrayImpl."""

    pass


class Client:
    """Mock implementation for Client."""

    pass


class CompileOptions:
    """Mock implementation for CompileOptions."""

    pass


class ConvolutionDimensionNumbers:
    """Mock implementation for ConvolutionDimensionNumbers."""

    pass


class CustomCallHandler:
    """Mock implementation for CustomCallHandler."""

    pass


class CustomCallTargetTraits:
    """Mock implementation for CustomCallTargetTraits."""

    pass


class Device:
    """Mock implementation for Device."""

    pass


class DeviceAssignment:
    """Mock implementation for DeviceAssignment."""

    pass


class DeviceList:
    """Mock implementation for DeviceList."""

    pass


class DeviceTopology:
    """Mock implementation for DeviceTopology."""

    pass


class DotDimensionNumbers:
    """Mock implementation for DotDimensionNumbers."""

    pass


class FftType:
    """Mock implementation for FftType."""

    pass


class Frame:
    """Mock implementation for Frame."""

    pass


class GSPMDSharding:
    """Mock implementation for GSPMDSharding."""

    pass


class GatherDimensionNumbers:
    """Mock implementation for GatherDimensionNumbers."""

    pass


class HloSharding:
    """Mock implementation for HloSharding."""

    pass


class HostBufferSemantics:
    """Mock implementation for HostBufferSemantics."""

    pass


class Layout:
    """Mock implementation for Layout."""

    pass


class LoadedExecutable:
    """Mock implementation for LoadedExecutable."""

    pass


class Mapping:
    """Mock implementation for Mapping."""

    pass


class Memory:
    """Mock implementation for Memory."""

    pass


class NamedSharding:
    """Mock implementation for NamedSharding."""

    pass


class OpMetadata:
    """Mock implementation for OpMetadata."""

    pass


class OpSharding:
    """Mock implementation for OpSharding."""

    pass


class PaddingConfig:
    """Mock implementation for PaddingConfig."""

    pass


class PaddingConfigDimension:
    """Mock implementation for PaddingConfigDimension."""

    pass


class PaddingType:
    """Mock implementation for PaddingType."""

    pass


class PjRtLayout:
    """Mock implementation for PjRtLayout."""

    pass


class PmapSharding:
    """Mock implementation for PmapSharding."""

    pass


class PrecisionConfig:
    """Mock implementation for PrecisionConfig."""

    pass


class PrimitiveType:
    """Mock implementation for PrimitiveType."""

    pass


class ProgramShape:
    """Mock implementation for ProgramShape."""

    pass


class Protocol:
    """Mock implementation for Protocol."""

    pass


class ReplicaGroup:
    """Mock implementation for ReplicaGroup."""

    pass


class ScatterDimensionNumbers:
    """Mock implementation for ScatterDimensionNumbers."""

    pass


class Sequence:
    """Mock implementation for Sequence."""

    pass


class Shape:
    """Mock implementation for Shape."""

    pass


class ShapeIndex:
    """Mock implementation for ShapeIndex."""

    pass


class Sharding:
    """Mock implementation for Sharding."""

    pass


class SingleDeviceSharding:
    """Mock implementation for SingleDeviceSharding."""

    pass


class Traceback:
    """Mock implementation for Traceback."""

    pass


class XlaBuilder:
    """Mock implementation for XlaBuilder."""

    pass


class XlaComputation:
    """Mock implementation for XlaComputation."""

    pass


class XlaOp:
    """Mock implementation for XlaOp."""

    pass


class XlaRuntimeError:
    """Mock implementation for XlaRuntimeError."""

    pass


class bfloat16:
    """Mock implementation for bfloat16."""

    pass


class float8_e4m3b11fnuz:
    """Mock implementation for float8_e4m3b11fnuz."""

    pass


class float8_e4m3fn:
    """Mock implementation for float8_e4m3fn."""

    pass


class float8_e4m3fnuz:
    """Mock implementation for float8_e4m3fnuz."""

    pass


class float8_e5m2:
    """Mock implementation for float8_e5m2."""

    pass


class float8_e5m2fnuz:
    """Mock implementation for float8_e5m2fnuz."""

    pass


def CurrentSourceInfoMetadata(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for CurrentSourceInfoMetadata."""
    raise NotImplementedError(
        "CurrentSourceInfoMetadata not yet implemented in zero-jax"
    )


def LoadedExecutable_execute(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for LoadedExecutable_execute."""
    raise NotImplementedError(
        "LoadedExecutable_execute not yet implemented in zero-jax"
    )


def LoadedExecutable_execute_with_token(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for LoadedExecutable_execute_with_token."""
    raise NotImplementedError(
        "LoadedExecutable_execute_with_token not yet implemented in zero-jax"
    )


def dtype_to_etype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dtype_to_etype."""
    raise NotImplementedError("dtype_to_etype not yet implemented in zero-jax")


def execute_with_python_values(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for execute_with_python_values."""
    raise NotImplementedError(
        "execute_with_python_values not yet implemented in zero-jax"
    )


def execute_with_python_values_replicated(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for execute_with_python_values_replicated."""
    raise NotImplementedError(
        "execute_with_python_values_replicated not yet implemented in zero-jax"
    )


def generate_pjrt_gpu_plugin_options(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for generate_pjrt_gpu_plugin_options."""
    raise NotImplementedError(
        "generate_pjrt_gpu_plugin_options not yet implemented in zero-jax"
    )


def heap_profile(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for heap_profile."""
    raise NotImplementedError("heap_profile not yet implemented in zero-jax")


def initialize_pjrt_plugin(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for initialize_pjrt_plugin."""
    raise NotImplementedError("initialize_pjrt_plugin not yet implemented in zero-jax")


def load_pjrt_plugin_dynamically(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for load_pjrt_plugin_dynamically."""
    raise NotImplementedError(
        "load_pjrt_plugin_dynamically not yet implemented in zero-jax"
    )


def load_pjrt_plugin_with_c_api(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for load_pjrt_plugin_with_c_api."""
    raise NotImplementedError(
        "load_pjrt_plugin_with_c_api not yet implemented in zero-jax"
    )


def make_c_api_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_c_api_client."""
    raise NotImplementedError("make_c_api_client not yet implemented in zero-jax")


def make_c_api_device_topology(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_c_api_device_topology."""
    raise NotImplementedError(
        "make_c_api_device_topology not yet implemented in zero-jax"
    )


def make_convolution_dimension_numbers(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_convolution_dimension_numbers."""
    raise NotImplementedError(
        "make_convolution_dimension_numbers not yet implemented in zero-jax"
    )


def make_cpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_cpu_client."""
    raise NotImplementedError("make_cpu_client not yet implemented in zero-jax")


def make_dot_dimension_numbers(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_dot_dimension_numbers."""
    raise NotImplementedError(
        "make_dot_dimension_numbers not yet implemented in zero-jax"
    )


def make_gpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_gpu_client."""
    raise NotImplementedError("make_gpu_client not yet implemented in zero-jax")


def make_padding_config(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_padding_config."""
    raise NotImplementedError("make_padding_config not yet implemented in zero-jax")


def make_replica_groups(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_replica_groups."""
    raise NotImplementedError("make_replica_groups not yet implemented in zero-jax")


def make_tfrt_tpu_c_api_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_tfrt_tpu_c_api_client."""
    raise NotImplementedError(
        "make_tfrt_tpu_c_api_client not yet implemented in zero-jax"
    )


def make_tfrt_tpu_c_api_device_topology(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_tfrt_tpu_c_api_device_topology."""
    raise NotImplementedError(
        "make_tfrt_tpu_c_api_device_topology not yet implemented in zero-jax"
    )


def make_tpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_tpu_client."""
    raise NotImplementedError("make_tpu_client not yet implemented in zero-jax")


def pjrt_plugin_initialized(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for pjrt_plugin_initialized."""
    raise NotImplementedError("pjrt_plugin_initialized not yet implemented in zero-jax")


def pjrt_plugin_loaded(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for pjrt_plugin_loaded."""
    raise NotImplementedError("pjrt_plugin_loaded not yet implemented in zero-jax")


def register_custom_call_handler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_custom_call_handler."""
    raise NotImplementedError(
        "register_custom_call_handler not yet implemented in zero-jax"
    )


def register_custom_call_target(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_custom_call_target."""
    raise NotImplementedError(
        "register_custom_call_target not yet implemented in zero-jax"
    )


def shape_from_pyval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for shape_from_pyval."""
    raise NotImplementedError("shape_from_pyval not yet implemented in zero-jax")


def tracebacks(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tracebacks."""
    raise NotImplementedError("tracebacks not yet implemented in zero-jax")


def window_padding_type_to_pad_values(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for window_padding_type_to_pad_values."""
    raise NotImplementedError(
        "window_padding_type_to_pad_values not yet implemented in zero-jax"
    )


atexit: Any = None  # Mock module
contextlib: Any = None  # Mock module
enum: Any = None  # Mock module
gzip: Any = None  # Mock module
hlo_sharding_util: Any = None  # Mock module
ifrt_programs: Any = None  # Mock module
inspect: Any = None  # Mock module
logging: Any = None  # Mock module
ml_dtypes: Any = None  # Mock module
np: Any = None  # Mock module
ops: Any = None  # Mock module
os: Any = None  # Mock module
profiler: Any = None  # Mock module
threading: Any = None  # Mock module
