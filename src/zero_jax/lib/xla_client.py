"""Mock implementation for jax.lib.xla_client."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops

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
    return getattr(_ops, "CurrentSourceInfoMetadata")(*args, **kwargs)


def LoadedExecutable_execute(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for LoadedExecutable_execute."""
    return getattr(_ops, "LoadedExecutable_execute")(*args, **kwargs)


def LoadedExecutable_execute_with_token(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for LoadedExecutable_execute_with_token."""
    return getattr(_ops, "LoadedExecutable_execute_with_token")(*args, **kwargs)


def dtype_to_etype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dtype_to_etype."""
    return getattr(_ops, "dtype_to_etype")(*args, **kwargs)


def execute_with_python_values(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for execute_with_python_values."""
    return getattr(_ops, "execute_with_python_values")(*args, **kwargs)


def execute_with_python_values_replicated(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for execute_with_python_values_replicated."""
    return getattr(_ops, "execute_with_python_values_replicated")(*args, **kwargs)


def generate_pjrt_gpu_plugin_options(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for generate_pjrt_gpu_plugin_options."""
    return getattr(_ops, "generate_pjrt_gpu_plugin_options")(*args, **kwargs)


def heap_profile(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for heap_profile."""
    return getattr(_ops, "heap_profile")(*args, **kwargs)


def initialize_pjrt_plugin(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for initialize_pjrt_plugin."""
    return getattr(_ops, "initialize_pjrt_plugin")(*args, **kwargs)


def load_pjrt_plugin_dynamically(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for load_pjrt_plugin_dynamically."""
    return getattr(_ops, "load_pjrt_plugin_dynamically")(*args, **kwargs)


def load_pjrt_plugin_with_c_api(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for load_pjrt_plugin_with_c_api."""
    return getattr(_ops, "load_pjrt_plugin_with_c_api")(*args, **kwargs)


def make_c_api_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_c_api_client."""
    return getattr(_ops, "make_c_api_client")(*args, **kwargs)


def make_c_api_device_topology(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_c_api_device_topology."""
    return getattr(_ops, "make_c_api_device_topology")(*args, **kwargs)


def make_convolution_dimension_numbers(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_convolution_dimension_numbers."""
    return getattr(_ops, "make_convolution_dimension_numbers")(*args, **kwargs)


def make_cpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_cpu_client."""
    return getattr(_ops, "make_cpu_client")(*args, **kwargs)


def make_dot_dimension_numbers(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_dot_dimension_numbers."""
    return getattr(_ops, "make_dot_dimension_numbers")(*args, **kwargs)


def make_gpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_gpu_client."""
    return getattr(_ops, "make_gpu_client")(*args, **kwargs)


def make_padding_config(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_padding_config."""
    return getattr(_ops, "make_padding_config")(*args, **kwargs)


def make_replica_groups(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_replica_groups."""
    return getattr(_ops, "make_replica_groups")(*args, **kwargs)


def make_tfrt_tpu_c_api_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_tfrt_tpu_c_api_client."""
    return getattr(_ops, "make_tfrt_tpu_c_api_client")(*args, **kwargs)


def make_tfrt_tpu_c_api_device_topology(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_tfrt_tpu_c_api_device_topology."""
    return getattr(_ops, "make_tfrt_tpu_c_api_device_topology")(*args, **kwargs)


def make_tpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_tpu_client."""
    return getattr(_ops, "make_tpu_client")(*args, **kwargs)


def pjrt_plugin_initialized(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for pjrt_plugin_initialized."""
    return getattr(_ops, "pjrt_plugin_initialized")(*args, **kwargs)


def pjrt_plugin_loaded(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for pjrt_plugin_loaded."""
    return getattr(_ops, "pjrt_plugin_loaded")(*args, **kwargs)


def register_custom_call_handler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_custom_call_handler."""
    return getattr(_ops, "register_custom_call_handler")(*args, **kwargs)


def register_custom_call_target(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_custom_call_target."""
    return getattr(_ops, "register_custom_call_target")(*args, **kwargs)


def shape_from_pyval(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for shape_from_pyval."""
    return getattr(_ops, "shape_from_pyval")(*args, **kwargs)


def tracebacks(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tracebacks."""
    return getattr(_ops, "tracebacks")(*args, **kwargs)


def window_padding_type_to_pad_values(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for window_padding_type_to_pad_values."""
    return getattr(_ops, "window_padding_type_to_pad_values")(*args, **kwargs)


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
