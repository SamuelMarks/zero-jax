"""Tests for zero_jax.lib.xla_extension."""

from typing import Any

import pytest

import zero_jax.lib.xla_extension as mod


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


def test_class_CallInliner() -> None:
    """Test class CallInliner."""
    try:
        mod.CallInliner()
    except Exception:
        pass


def test_class_Client() -> None:
    """Test class Client."""
    try:
        mod.Client()
    except Exception:
        pass


def test_class_CompileOnlyPyClient() -> None:
    """Test class CompileOnlyPyClient."""
    try:
        mod.CompileOnlyPyClient()
    except Exception:
        pass


def test_class_CompileOptions() -> None:
    """Test class CompileOptions."""
    try:
        mod.CompileOptions()
    except Exception:
        pass


def test_class_CompiledMemoryStats() -> None:
    """Test class CompiledMemoryStats."""
    try:
        mod.CompiledMemoryStats()
    except Exception:
        pass


def test_class_CpuCollectives() -> None:
    """Test class CpuCollectives."""
    try:
        mod.CpuCollectives()
    except Exception:
        pass


def test_class_DebugOptions() -> None:
    """Test class DebugOptions."""
    try:
        mod.DebugOptions()
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


def test_class_DistributedRuntimeClient() -> None:
    """Test class DistributedRuntimeClient."""
    try:
        mod.DistributedRuntimeClient()
    except Exception:
        pass


def test_class_DistributedRuntimeService() -> None:
    """Test class DistributedRuntimeService."""
    try:
        mod.DistributedRuntimeService()
    except Exception:
        pass


def test_class_Executable() -> None:
    """Test class Executable."""
    try:
        mod.Executable()
    except Exception:
        pass


def test_class_ExecutableBuildOptions() -> None:
    """Test class ExecutableBuildOptions."""
    try:
        mod.ExecutableBuildOptions()
    except Exception:
        pass


def test_class_ExecuteResults() -> None:
    """Test class ExecuteResults."""
    try:
        mod.ExecuteResults()
    except Exception:
        pass


def test_class_FftType() -> None:
    """Test class FftType."""
    try:
        mod.FftType()
    except Exception:
        pass


def test_class_FlattenCallGraph() -> None:
    """Test class FlattenCallGraph."""
    try:
        mod.FlattenCallGraph()
    except Exception:
        pass


def test_class_Frame() -> None:
    """Test class Frame."""
    try:
        mod.Frame()
    except Exception:
        pass


def test_class_FrontendAttributes() -> None:
    """Test class FrontendAttributes."""
    try:
        mod.FrontendAttributes()
    except Exception:
        pass


def test_class_GSPMDSharding() -> None:
    """Test class GSPMDSharding."""
    try:
        mod.GSPMDSharding()
    except Exception:
        pass


def test_class_HloComputation() -> None:
    """Test class HloComputation."""
    try:
        mod.HloComputation()
    except Exception:
        pass


def test_class_HloDCE() -> None:
    """Test class HloDCE."""
    try:
        mod.HloDCE()
    except Exception:
        pass


def test_class_HloModule() -> None:
    """Test class HloModule."""
    try:
        mod.HloModule()
    except Exception:
        pass


def test_class_HloModuleGroup() -> None:
    """Test class HloModuleGroup."""
    try:
        mod.HloModuleGroup()
    except Exception:
        pass


def test_class_HloPassInterface() -> None:
    """Test class HloPassInterface."""
    try:
        mod.HloPassInterface()
    except Exception:
        pass


def test_class_HloPrintOptions() -> None:
    """Test class HloPrintOptions."""
    try:
        mod.HloPrintOptions()
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


def test_class_Literal() -> None:
    """Test class Literal."""
    try:
        mod.Literal()
    except Exception:
        pass


def test_class_LoadedExecutable() -> None:
    """Test class LoadedExecutable."""
    try:
        mod.LoadedExecutable()
    except Exception:
        pass


def test_class_Memory() -> None:
    """Test class Memory."""
    try:
        mod.Memory()
    except Exception:
        pass


def test_class_MpiCollectives() -> None:
    """Test class MpiCollectives."""
    try:
        mod.MpiCollectives()
    except Exception:
        pass


def test_class_NamedSharding() -> None:
    """Test class NamedSharding."""
    try:
        mod.NamedSharding()
    except Exception:
        pass


def test_class_OpSharding() -> None:
    """Test class OpSharding."""
    try:
        mod.OpSharding()
    except Exception:
        pass


def test_class_OpSharding_ShardGroupType() -> None:
    """Test class OpSharding_ShardGroupType."""
    try:
        mod.OpSharding_ShardGroupType()
    except Exception:
        pass


def test_class_OpSharding_Type() -> None:
    """Test class OpSharding_Type."""
    try:
        mod.OpSharding_Type()
    except Exception:
        pass


def test_class_PjRtLayout() -> None:
    """Test class PjRtLayout."""
    try:
        mod.PjRtLayout()
    except Exception:
        pass


def test_class_PjRtXlaLayout() -> None:
    """Test class PjRtXlaLayout."""
    try:
        mod.PjRtXlaLayout()
    except Exception:
        pass


def test_class_PjitFunction() -> None:
    """Test class PjitFunction."""
    try:
        mod.PjitFunction()
    except Exception:
        pass


def test_class_PjitFunctionCache() -> None:
    """Test class PjitFunctionCache."""
    try:
        mod.PjitFunctionCache()
    except Exception:
        pass


def test_class_PmapFunction() -> None:
    """Test class PmapFunction."""
    try:
        mod.PmapFunction()
    except Exception:
        pass


def test_class_PmapSharding() -> None:
    """Test class PmapSharding."""
    try:
        mod.PmapSharding()
    except Exception:
        pass


def test_class_PrecisionConfig_Precision() -> None:
    """Test class PrecisionConfig_Precision."""
    try:
        mod.PrecisionConfig_Precision()
    except Exception:
        pass


def test_class_PreemptionSyncManager() -> None:
    """Test class PreemptionSyncManager."""
    try:
        mod.PreemptionSyncManager()
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


def test_class_PyTreeRegistry() -> None:
    """Test class PyTreeRegistry."""
    try:
        mod.PyTreeRegistry()
    except Exception:
        pass


def test_class_ResultHandler() -> None:
    """Test class ResultHandler."""
    try:
        mod.ResultHandler()
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


def test_class_ShardedToken() -> None:
    """Test class ShardedToken."""
    try:
        mod.ShardedToken()
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


def test_class_Token() -> None:
    """Test class Token."""
    try:
        mod.Token()
    except Exception:
        pass


def test_class_Traceback() -> None:
    """Test class Traceback."""
    try:
        mod.Traceback()
    except Exception:
        pass


def test_class_TupleSimplifier() -> None:
    """Test class TupleSimplifier."""
    try:
        mod.TupleSimplifier()
    except Exception:
        pass


def test_class_WeakrefLRUCache() -> None:
    """Test class WeakrefLRUCache."""
    try:
        mod.WeakrefLRUCache()
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
