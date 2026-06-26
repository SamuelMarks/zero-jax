"""Mock implementation for jax.lib.xla_extension."""

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
    """Mock implementation for ArrayImpl."""

    pass


class CallInliner:
    """Mock implementation for CallInliner."""

    pass


class Client:
    """Mock implementation for Client."""

    pass


class CompileOnlyPyClient:
    """Mock implementation for CompileOnlyPyClient."""

    pass


class CompileOptions:
    """Mock implementation for CompileOptions."""

    pass


class CompiledMemoryStats:
    """Mock implementation for CompiledMemoryStats."""

    pass


class CpuCollectives:
    """Mock implementation for CpuCollectives."""

    pass


class DebugOptions:
    """Mock implementation for DebugOptions."""

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


class DistributedRuntimeClient:
    """Mock implementation for DistributedRuntimeClient."""

    pass


class DistributedRuntimeService:
    """Mock implementation for DistributedRuntimeService."""

    pass


class Executable:
    """Mock implementation for Executable."""

    pass


class ExecutableBuildOptions:
    """Mock implementation for ExecutableBuildOptions."""

    pass


class ExecuteResults:
    """Mock implementation for ExecuteResults."""

    pass


class FftType:
    """Mock implementation for FftType."""

    pass


class FlattenCallGraph:
    """Mock implementation for FlattenCallGraph."""

    pass


class Frame:
    """Mock implementation for Frame."""

    pass


class FrontendAttributes:
    """Mock implementation for FrontendAttributes."""

    pass


class GSPMDSharding:
    """Mock implementation for GSPMDSharding."""

    pass


class HloComputation:
    """Mock implementation for HloComputation."""

    pass


class HloDCE:
    """Mock implementation for HloDCE."""

    pass


class HloModule:
    """Mock implementation for HloModule."""

    pass


class HloModuleGroup:
    """Mock implementation for HloModuleGroup."""

    pass


class HloPassInterface:
    """Mock implementation for HloPassInterface."""

    pass


class HloPrintOptions:
    """Mock implementation for HloPrintOptions."""

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


class Literal:
    """Mock implementation for Literal."""

    pass


class LoadedExecutable:
    """Mock implementation for LoadedExecutable."""

    pass


class Memory:
    """Mock implementation for Memory."""

    pass


class MpiCollectives:
    """Mock implementation for MpiCollectives."""

    pass


class NamedSharding:
    """Mock implementation for NamedSharding."""

    pass


class OpSharding:
    """Mock implementation for OpSharding."""

    pass


class OpSharding_ShardGroupType:
    """Mock implementation for OpSharding_ShardGroupType."""

    pass


class OpSharding_Type:
    """Mock implementation for OpSharding_Type."""

    pass


class PjRtLayout:
    """Mock implementation for PjRtLayout."""

    pass


class PjRtXlaLayout:
    """Mock implementation for PjRtXlaLayout."""

    pass


class PjitFunction:
    """Mock implementation for PjitFunction."""

    pass


class PjitFunctionCache:
    """Mock implementation for PjitFunctionCache."""

    pass


class PmapFunction:
    """Mock implementation for PmapFunction."""

    pass


class PmapSharding:
    """Mock implementation for PmapSharding."""

    pass


class PrecisionConfig_Precision:
    """Mock implementation for PrecisionConfig_Precision."""

    pass


class PreemptionSyncManager:
    """Mock implementation for PreemptionSyncManager."""

    pass


class PrimitiveType:
    """Mock implementation for PrimitiveType."""

    pass


class ProgramShape:
    """Mock implementation for ProgramShape."""

    pass


class PyTreeRegistry:
    """Mock implementation for PyTreeRegistry."""

    pass


class ResultHandler:
    """Mock implementation for ResultHandler."""

    pass


class Shape:
    """Mock implementation for Shape."""

    pass


class ShapeIndex:
    """Mock implementation for ShapeIndex."""

    pass


class ShardedToken:
    """Mock implementation for ShardedToken."""

    pass


class Sharding:
    """Mock implementation for Sharding."""

    pass


class SingleDeviceSharding:
    """Mock implementation for SingleDeviceSharding."""

    pass


class Token:
    """Mock implementation for Token."""

    pass


class Traceback:
    """Mock implementation for Traceback."""

    pass


class TupleSimplifier:
    """Mock implementation for TupleSimplifier."""

    pass


class WeakrefLRUCache:
    """Mock implementation for WeakrefLRUCache."""

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
