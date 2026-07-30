"""Frontend API routing for jax.lib.xla_client."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops

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
    """Frontend API routing for ArrayImpl."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Client:
    """Frontend API routing for Client."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class CompileOptions:
    """Frontend API routing for CompileOptions."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class ConvolutionDimensionNumbers:
    """Frontend API routing for ConvolutionDimensionNumbers."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class CustomCallHandler:
    """Frontend API routing for CustomCallHandler."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class CustomCallTargetTraits:
    """Frontend API routing for CustomCallTargetTraits."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Device:
    """Frontend API routing for Device."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class DeviceAssignment:
    """Frontend API routing for DeviceAssignment."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class DeviceList:
    """Frontend API routing for DeviceList."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class DeviceTopology:
    """Frontend API routing for DeviceTopology."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class DotDimensionNumbers:
    """Frontend API routing for DotDimensionNumbers."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class FftType:
    """Frontend API routing for FftType."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Frame:
    """Frontend API routing for Frame."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class GSPMDSharding:
    """Frontend API routing for GSPMDSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class GatherDimensionNumbers:
    """Frontend API routing for GatherDimensionNumbers."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class HloSharding:
    """Frontend API routing for HloSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class HostBufferSemantics:
    """Frontend API routing for HostBufferSemantics."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Layout:
    """Frontend API routing for Layout."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class LoadedExecutable:
    """Frontend API routing for LoadedExecutable."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Mapping:
    """Frontend API routing for Mapping."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Memory:
    """Frontend API routing for Memory."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class NamedSharding:
    """Frontend API routing for NamedSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class OpMetadata:
    """Frontend API routing for OpMetadata."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class OpSharding:
    """Frontend API routing for OpSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PaddingConfig:
    """Frontend API routing for PaddingConfig."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PaddingConfigDimension:
    """Frontend API routing for PaddingConfigDimension."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PaddingType:
    """Frontend API routing for PaddingType."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PjRtLayout:
    """Frontend API routing for PjRtLayout."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PmapSharding:
    """Frontend API routing for PmapSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PrecisionConfig:
    """Frontend API routing for PrecisionConfig."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PrimitiveType:
    """Frontend API routing for PrimitiveType."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class ProgramShape:
    """Frontend API routing for ProgramShape."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Protocol:
    """Frontend API routing for Protocol."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class ReplicaGroup:
    """Frontend API routing for ReplicaGroup."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class ScatterDimensionNumbers:
    """Frontend API routing for ScatterDimensionNumbers."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Sequence:
    """Frontend API routing for Sequence."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Shape:
    """Frontend API routing for Shape."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class ShapeIndex:
    """Frontend API routing for ShapeIndex."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Sharding:
    """Frontend API routing for Sharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class SingleDeviceSharding:
    """Frontend API routing for SingleDeviceSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Traceback:
    """Frontend API routing for Traceback."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class XlaBuilder:
    """Frontend API routing for XlaBuilder."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class XlaComputation:
    """Frontend API routing for XlaComputation."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class XlaOp:
    """Frontend API routing for XlaOp."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class XlaRuntimeError:
    """Frontend API routing for XlaRuntimeError."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class bfloat16:
    """Frontend API routing for bfloat16."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class float8_e4m3b11fnuz:
    """Frontend API routing for float8_e4m3b11fnuz."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class float8_e4m3fn:
    """Frontend API routing for float8_e4m3fn."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class float8_e4m3fnuz:
    """Frontend API routing for float8_e4m3fnuz."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class float8_e5m2:
    """Frontend API routing for float8_e5m2."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class float8_e5m2fnuz:
    """Frontend API routing for float8_e5m2fnuz."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def CurrentSourceInfoMetadata(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for CurrentSourceInfoMetadata.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.CurrentSourceInfoMetadata(*args, **kwargs)


def LoadedExecutable_execute(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for LoadedExecutable_execute.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.LoadedExecutable_execute(*args, **kwargs)


def LoadedExecutable_execute_with_token(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for LoadedExecutable_execute_with_token.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.LoadedExecutable_execute_with_token(*args, **kwargs)


def dtype_to_etype(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for dtype_to_etype.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.dtype_to_etype(*args, **kwargs)


def execute_with_python_values(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for execute_with_python_values.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.execute_with_python_values(*args, **kwargs)


def execute_with_python_values_replicated(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for execute_with_python_values_replicated.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.execute_with_python_values_replicated(*args, **kwargs)


def generate_pjrt_gpu_plugin_options(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for generate_pjrt_gpu_plugin_options.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.generate_pjrt_gpu_plugin_options(*args, **kwargs)


def heap_profile(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for heap_profile.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.heap_profile(*args, **kwargs)


def initialize_pjrt_plugin(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for initialize_pjrt_plugin.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.initialize_pjrt_plugin(*args, **kwargs)


def load_pjrt_plugin_dynamically(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for load_pjrt_plugin_dynamically.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.load_pjrt_plugin_dynamically(*args, **kwargs)


def load_pjrt_plugin_with_c_api(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for load_pjrt_plugin_with_c_api.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.load_pjrt_plugin_with_c_api(*args, **kwargs)


def make_c_api_client(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_c_api_client.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_c_api_client(*args, **kwargs)


def make_c_api_device_topology(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_c_api_device_topology.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_c_api_device_topology(*args, **kwargs)


def make_convolution_dimension_numbers(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_convolution_dimension_numbers.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_convolution_dimension_numbers(*args, **kwargs)


def make_cpu_client(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_cpu_client.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_cpu_client(*args, **kwargs)


def make_dot_dimension_numbers(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_dot_dimension_numbers.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_dot_dimension_numbers(*args, **kwargs)


def make_gpu_client(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_gpu_client.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_gpu_client(*args, **kwargs)


def make_padding_config(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_padding_config.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_padding_config(*args, **kwargs)


def make_replica_groups(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_replica_groups.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_replica_groups(*args, **kwargs)


def make_tfrt_tpu_c_api_client(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_tfrt_tpu_c_api_client.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_tfrt_tpu_c_api_client(*args, **kwargs)


def make_tfrt_tpu_c_api_device_topology(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_tfrt_tpu_c_api_device_topology.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_tfrt_tpu_c_api_device_topology(*args, **kwargs)


def make_tpu_client(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for make_tpu_client.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.make_tpu_client(*args, **kwargs)


def pjrt_plugin_initialized(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for pjrt_plugin_initialized.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.pjrt_plugin_initialized(*args, **kwargs)


def pjrt_plugin_loaded(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for pjrt_plugin_loaded.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.pjrt_plugin_loaded(*args, **kwargs)


def register_custom_call_handler(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for register_custom_call_handler.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.register_custom_call_handler(*args, **kwargs)


def register_custom_call_target(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for register_custom_call_target.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.register_custom_call_target(*args, **kwargs)


def shape_from_pyval(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for shape_from_pyval.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.shape_from_pyval(*args, **kwargs)


def tracebacks(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for tracebacks.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.tracebacks(*args, **kwargs)


def window_padding_type_to_pad_values(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for window_padding_type_to_pad_values.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.window_padding_type_to_pad_values(*args, **kwargs)


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
