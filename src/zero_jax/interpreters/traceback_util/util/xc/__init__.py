"""Frontend API routing for zero-jax.interpreters.traceback_util.util.xc."""

from typing import Any

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


class ArrayImpl:
    def __init__(self, *args, **kwargs):
        pass


class Client:
    def __init__(self, *args, **kwargs):
        pass


class CompileOptions:
    def __init__(self, *args, **kwargs):
        pass


class CustomCallHandler:
    def __init__(self, *args, **kwargs):
        pass


class CustomCallTargetTraits:
    def __init__(self, *args, **kwargs):
        pass


class DeviceAssignment:
    def __init__(self, *args, **kwargs):
        pass


class DeviceList:
    def __init__(self, *args, **kwargs):
        pass


class DeviceTopology:
    def __init__(self, *args, **kwargs):
        pass


class DotDimensionNumbers:
    def __init__(self, *args, **kwargs):
        pass


class FftType:
    def __init__(self, *args, **kwargs):
        pass


class Frame:
    def __init__(self, *args, **kwargs):
        pass


class GSPMDSharding:
    def __init__(self, *args, **kwargs):
        pass


class GatherDimensionNumbers:
    def __init__(self, *args, **kwargs):
        pass


class HloSharding:
    def __init__(self, *args, **kwargs):
        pass


class HostBufferSemantics:
    def __init__(self, *args, **kwargs):
        pass


class Layout:
    def __init__(self, *args, **kwargs):
        pass


class LoadedExecutable:
    def __init__(self, *args, **kwargs):
        pass


class Mapping:
    def __init__(self, *args, **kwargs):
        pass


class Memory:
    def __init__(self, *args, **kwargs):
        pass


class NamedSharding:
    def __init__(self, *args, **kwargs):
        pass


class OpMetadata:
    def __init__(self, *args, **kwargs):
        pass


class OpSharding:
    def __init__(self, *args, **kwargs):
        pass


class PaddingConfig:
    def __init__(self, *args, **kwargs):
        pass


class PaddingConfigDimension:
    def __init__(self, *args, **kwargs):
        pass


class PaddingType:
    def __init__(self, *args, **kwargs):
        pass


class PjRtLayout:
    def __init__(self, *args, **kwargs):
        pass


class PmapSharding:
    def __init__(self, *args, **kwargs):
        pass


class PrecisionConfig:
    def __init__(self, *args, **kwargs):
        pass


class PrimitiveType:
    def __init__(self, *args, **kwargs):
        pass


class ProgramShape:
    def __init__(self, *args, **kwargs):
        pass


class Protocol:
    def __init__(self, *args, **kwargs):
        pass


class ReplicaGroup:
    def __init__(self, *args, **kwargs):
        pass


class Shape:
    def __init__(self, *args, **kwargs):
        pass


class ShapeIndex:
    def __init__(self, *args, **kwargs):
        pass


class Sharding:
    def __init__(self, *args, **kwargs):
        pass


class SingleDeviceSharding:
    def __init__(self, *args, **kwargs):
        pass


class Traceback:
    def __init__(self, *args, **kwargs):
        pass


class XlaBuilder:
    def __init__(self, *args, **kwargs):
        pass


class XlaComputation:
    def __init__(self, *args, **kwargs):
        pass


class XlaOp:
    def __init__(self, *args, **kwargs):
        pass


class XlaRuntimeError:
    def __init__(self, *args, **kwargs):
        pass


def atexit(*args: Any, **kwargs: Any) -> Any:
    pass


def batched_device_put(*args: Any, **kwargs: Any) -> Any:
    pass


def bfloat16(*args: Any, **kwargs: Any) -> Any:
    pass


def contextlib(*args: Any, **kwargs: Any) -> Any:
    pass


def custom_call_targets(*args: Any, **kwargs: Any) -> Any:
    pass


def dtype_to_etype(*args: Any, **kwargs: Any) -> Any:
    pass


def enum(*args: Any, **kwargs: Any) -> Any:
    pass


def float8_e4m3b11fnuz(*args: Any, **kwargs: Any) -> Any:
    pass


def float8_e4m3fn(*args: Any, **kwargs: Any) -> Any:
    pass


def float8_e4m3fnuz(*args: Any, **kwargs: Any) -> Any:
    pass


def float8_e5m2(*args: Any, **kwargs: Any) -> Any:
    pass


def float8_e5m2fnuz(*args: Any, **kwargs: Any) -> Any:
    pass


def gzip(*args: Any, **kwargs: Any) -> Any:
    pass


def heap_profile(*args: Any, **kwargs: Any) -> Any:
    pass


def inspect(*args: Any, **kwargs: Any) -> Any:
    pass


def logger(*args: Any, **kwargs: Any) -> Any:
    pass


def logging(*args: Any, **kwargs: Any) -> Any:
    pass


def make_c_api_client(*args: Any, **kwargs: Any) -> Any:
    pass


def make_cpu_client(*args: Any, **kwargs: Any) -> Any:
    pass


def make_gpu_client(*args: Any, **kwargs: Any) -> Any:
    pass


def make_padding_config(*args: Any, **kwargs: Any) -> Any:
    pass


def make_replica_groups(*args: Any, **kwargs: Any) -> Any:
    pass


def make_tpu_client(*args: Any, **kwargs: Any) -> Any:
    pass


def ml_dtypes(*args: Any, **kwargs: Any) -> Any:
    pass


def mlir_api_version(*args: Any, **kwargs: Any) -> Any:
    pass


def os(*args: Any, **kwargs: Any) -> Any:
    pass


def pjrt_plugin_loaded(*args: Any, **kwargs: Any) -> Any:
    pass


def shape_from_pyval(*args: Any, **kwargs: Any) -> Any:
    pass


def threading(*args: Any, **kwargs: Any) -> Any:
    pass


def tracebacks(*args: Any, **kwargs: Any) -> Any:
    pass


def weakref_lru_cache(*args: Any, **kwargs: Any) -> Any:
    pass


def xla_platform_names(*args: Any, **kwargs: Any) -> Any:
    pass


def __getattr__(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        def stub(*args, **kwargs):
            raise NotImplementedError(f"Stub for {name} is not implemented in backend")

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
