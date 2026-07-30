"""Frontend API routing for jax.lib.xla_extension."""

from typing import Any

array_result_handler: Any = None
batched_block_until_ready: Any = None
batched_copy_array_to_devices_with_sharding: Any = None
batched_device_put: Any = None
buffer_to_dlpack_managed_tensor: Any = None
check_and_canonicalize_memory_kind: Any = None
collect_garbage: Any = None
create_preemption_sync_manager: Any = None
cuda_array_interface_to_buffer: Any = None
custom_call_targets: Any = None
dlpack_managed_tensor_to_buffer: Any = None
encode_inspect_sharding_callback: Any = None
get_c_api_client: Any = None
get_c_api_topology: Any = None
get_default_c_api_topology: Any = None
get_distributed_runtime_client: Any = None
get_distributed_runtime_service: Any = None
get_tfrt_cpu_client: Any = None
get_topology_for_devices: Any = None
hlo_module_cost_analysis: Any = None
hlo_module_from_text: Any = None
hlo_module_to_dot_graph: Any = None
initialize_pjrt_plugin: Any = None
is_asan: Any = None
is_msan: Any = None
is_optimized_build: Any = None
is_sanitized: Any = None
is_tsan: Any = None
json_to_pprof_profile: Any = None
load_pjrt_plugin: Any = None
make_gloo_tcp_collectives: Any = None
make_mpi_collectives: Any = None
pjit: Any = None
pjrt_plugin_initialized: Any = None
pjrt_plugin_loaded: Any = None
pprof_profile_to_json: Any = None
register_custom_call_partitioner: Any = None
register_custom_call_target: Any = None
replace_thread_exc_traceback: Any = None
weakref_lru_cache: Any = None


class ArrayImpl:
    """Frontend API routing for ArrayImpl."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class CallInliner:
    """Frontend API routing for CallInliner."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Client:
    """Frontend API routing for Client."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class CompileOnlyPyClient:
    """Frontend API routing for CompileOnlyPyClient."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class CompileOptions:
    """Frontend API routing for CompileOptions."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class CompiledMemoryStats:
    """Frontend API routing for CompiledMemoryStats."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class CpuCollectives:
    """Frontend API routing for CpuCollectives."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class DebugOptions:
    """Frontend API routing for DebugOptions."""

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


class DistributedRuntimeClient:
    """Frontend API routing for DistributedRuntimeClient."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class DistributedRuntimeService:
    """Frontend API routing for DistributedRuntimeService."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Executable:
    """Frontend API routing for Executable."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class ExecutableBuildOptions:
    """Frontend API routing for ExecutableBuildOptions."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class ExecuteResults:
    """Frontend API routing for ExecuteResults."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class FftType:
    """Frontend API routing for FftType."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class FlattenCallGraph:
    """Frontend API routing for FlattenCallGraph."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Frame:
    """Frontend API routing for Frame."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class FrontendAttributes:
    """Frontend API routing for FrontendAttributes."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class GSPMDSharding:
    """Frontend API routing for GSPMDSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class HloComputation:
    """Frontend API routing for HloComputation."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class HloDCE:
    """Frontend API routing for HloDCE."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class HloModule:
    """Frontend API routing for HloModule."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class HloModuleGroup:
    """Frontend API routing for HloModuleGroup."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class HloPassInterface:
    """Frontend API routing for HloPassInterface."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class HloPrintOptions:
    """Frontend API routing for HloPrintOptions."""

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


class Literal:
    """Frontend API routing for Literal."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class LoadedExecutable:
    """Frontend API routing for LoadedExecutable."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Memory:
    """Frontend API routing for Memory."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class MpiCollectives:
    """Frontend API routing for MpiCollectives."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class NamedSharding:
    """Frontend API routing for NamedSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class OpSharding:
    """Frontend API routing for OpSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class OpSharding_ShardGroupType:
    """Frontend API routing for OpSharding_ShardGroupType."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class OpSharding_Type:
    """Frontend API routing for OpSharding_Type."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PjRtLayout:
    """Frontend API routing for PjRtLayout."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PjRtXlaLayout:
    """Frontend API routing for PjRtXlaLayout."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PjitFunction:
    """Frontend API routing for PjitFunction."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PjitFunctionCache:
    """Frontend API routing for PjitFunctionCache."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PmapFunction:
    """Frontend API routing for PmapFunction."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PmapSharding:
    """Frontend API routing for PmapSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PrecisionConfig_Precision:
    """Frontend API routing for PrecisionConfig_Precision."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PreemptionSyncManager:
    """Frontend API routing for PreemptionSyncManager."""

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


class PyTreeRegistry:
    """Frontend API routing for PyTreeRegistry."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class ResultHandler:
    """Frontend API routing for ResultHandler."""

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


class ShardedToken:
    """Frontend API routing for ShardedToken."""

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


class Token:
    """Frontend API routing for Token."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Traceback:
    """Frontend API routing for Traceback."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class TupleSimplifier:
    """Frontend API routing for TupleSimplifier."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class WeakrefLRUCache:
    """Frontend API routing for WeakrefLRUCache."""

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


hlo_sharding_util: Any = None  # Mock module
ifrt_programs: Any = None  # Mock module
ifrt_proxy: Any = None  # Mock module
jax_jit: Any = None  # Mock module
mlir: Any = None  # Mock module
ops: Any = None  # Mock module
outfeed_receiver: Any = None  # Mock module
pmap_lib: Any = None  # Mock module
profiler: Any = None  # Mock module
pytree: Any = None  # Mock module
transfer_guard_lib: Any = None  # Mock module
