"""Tests for zero_jax.lib.xla_extension."""

from __future__ import annotations

import pytest

import zero_jax.lib.xla_extension as mod


def test_class_ArrayImpl() -> None:
    """Test ArrayImpl."""
    obj = mod.ArrayImpl(name="ArrayImpl", value=1)
    assert isinstance(obj, mod.ArrayImpl)
    assert obj.name == "ArrayImpl"
    assert obj.value == 1


def test_class_CallInliner() -> None:
    """Test CallInliner."""
    obj = mod.CallInliner(name="CallInliner", value=1)
    assert isinstance(obj, mod.CallInliner)
    assert obj.name == "CallInliner"
    assert obj.value == 1


def test_class_Client() -> None:
    """Test Client."""
    obj = mod.Client(name="Client", value=1)
    assert isinstance(obj, mod.Client)
    assert obj.name == "Client"
    assert obj.value == 1


def test_class_CompileOnlyPyClient() -> None:
    """Test CompileOnlyPyClient."""
    obj = mod.CompileOnlyPyClient(name="CompileOnlyPyClient", value=1)
    assert isinstance(obj, mod.CompileOnlyPyClient)
    assert obj.name == "CompileOnlyPyClient"
    assert obj.value == 1


def test_class_CompileOptions() -> None:
    """Test CompileOptions."""
    obj = mod.CompileOptions(name="CompileOptions", value=1)
    assert isinstance(obj, mod.CompileOptions)
    assert obj.name == "CompileOptions"
    assert obj.value == 1


def test_class_CompiledMemoryStats() -> None:
    """Test CompiledMemoryStats."""
    obj = mod.CompiledMemoryStats(name="CompiledMemoryStats", value=1)
    assert isinstance(obj, mod.CompiledMemoryStats)
    assert obj.name == "CompiledMemoryStats"
    assert obj.value == 1


def test_class_CpuCollectives() -> None:
    """Test CpuCollectives."""
    obj = mod.CpuCollectives(name="CpuCollectives", value=1)
    assert isinstance(obj, mod.CpuCollectives)
    assert obj.name == "CpuCollectives"
    assert obj.value == 1


def test_class_DebugOptions() -> None:
    """Test DebugOptions."""
    obj = mod.DebugOptions(name="DebugOptions", value=1)
    assert isinstance(obj, mod.DebugOptions)
    assert obj.name == "DebugOptions"
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


def test_class_DistributedRuntimeClient() -> None:
    """Test DistributedRuntimeClient."""
    obj = mod.DistributedRuntimeClient(name="DistributedRuntimeClient", value=1)
    assert isinstance(obj, mod.DistributedRuntimeClient)
    assert obj.name == "DistributedRuntimeClient"
    assert obj.value == 1


def test_class_DistributedRuntimeService() -> None:
    """Test DistributedRuntimeService."""
    obj = mod.DistributedRuntimeService(name="DistributedRuntimeService", value=1)
    assert isinstance(obj, mod.DistributedRuntimeService)
    assert obj.name == "DistributedRuntimeService"
    assert obj.value == 1


def test_class_Executable() -> None:
    """Test Executable."""
    obj = mod.Executable(name="Executable", value=1)
    assert isinstance(obj, mod.Executable)
    assert obj.name == "Executable"
    assert obj.value == 1


def test_class_ExecutableBuildOptions() -> None:
    """Test ExecutableBuildOptions."""
    obj = mod.ExecutableBuildOptions(name="ExecutableBuildOptions", value=1)
    assert isinstance(obj, mod.ExecutableBuildOptions)
    assert obj.name == "ExecutableBuildOptions"
    assert obj.value == 1


def test_class_ExecuteResults() -> None:
    """Test ExecuteResults."""
    obj = mod.ExecuteResults(name="ExecuteResults", value=1)
    assert isinstance(obj, mod.ExecuteResults)
    assert obj.name == "ExecuteResults"
    assert obj.value == 1


def test_class_FftType() -> None:
    """Test FftType."""
    obj = mod.FftType(name="FftType", value=1)
    assert isinstance(obj, mod.FftType)
    assert obj.name == "FftType"
    assert obj.value == 1


def test_class_FlattenCallGraph() -> None:
    """Test FlattenCallGraph."""
    obj = mod.FlattenCallGraph(name="FlattenCallGraph", value=1)
    assert isinstance(obj, mod.FlattenCallGraph)
    assert obj.name == "FlattenCallGraph"
    assert obj.value == 1


def test_class_Frame() -> None:
    """Test Frame."""
    obj = mod.Frame(name="Frame", value=1)
    assert isinstance(obj, mod.Frame)
    assert obj.name == "Frame"
    assert obj.value == 1


def test_class_FrontendAttributes() -> None:
    """Test FrontendAttributes."""
    obj = mod.FrontendAttributes(name="FrontendAttributes", value=1)
    assert isinstance(obj, mod.FrontendAttributes)
    assert obj.name == "FrontendAttributes"
    assert obj.value == 1


def test_class_GSPMDSharding() -> None:
    """Test GSPMDSharding."""
    obj = mod.GSPMDSharding(name="GSPMDSharding", value=1)
    assert isinstance(obj, mod.GSPMDSharding)
    assert obj.name == "GSPMDSharding"
    assert obj.value == 1


def test_class_HloComputation() -> None:
    """Test HloComputation."""
    obj = mod.HloComputation(name="HloComputation", value=1)
    assert isinstance(obj, mod.HloComputation)
    assert obj.name == "HloComputation"
    assert obj.value == 1


def test_class_HloDCE() -> None:
    """Test HloDCE."""
    obj = mod.HloDCE(name="HloDCE", value=1)
    assert isinstance(obj, mod.HloDCE)
    assert obj.name == "HloDCE"
    assert obj.value == 1


def test_class_HloModule() -> None:
    """Test HloModule."""
    obj = mod.HloModule(name="HloModule", value=1)
    assert isinstance(obj, mod.HloModule)
    assert obj.name == "HloModule"
    assert obj.value == 1


def test_class_HloModuleGroup() -> None:
    """Test HloModuleGroup."""
    obj = mod.HloModuleGroup(name="HloModuleGroup", value=1)
    assert isinstance(obj, mod.HloModuleGroup)
    assert obj.name == "HloModuleGroup"
    assert obj.value == 1


def test_class_HloPassInterface() -> None:
    """Test HloPassInterface."""
    obj = mod.HloPassInterface(name="HloPassInterface", value=1)
    assert isinstance(obj, mod.HloPassInterface)
    assert obj.name == "HloPassInterface"
    assert obj.value == 1


def test_class_HloPrintOptions() -> None:
    """Test HloPrintOptions."""
    obj = mod.HloPrintOptions(name="HloPrintOptions", value=1)
    assert isinstance(obj, mod.HloPrintOptions)
    assert obj.name == "HloPrintOptions"
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


def test_class_Literal() -> None:
    """Test Literal."""
    obj = mod.Literal(name="Literal", value=1)
    assert isinstance(obj, mod.Literal)
    assert obj.name == "Literal"
    assert obj.value == 1


def test_class_LoadedExecutable() -> None:
    """Test LoadedExecutable."""
    obj = mod.LoadedExecutable(name="LoadedExecutable", value=1)
    assert isinstance(obj, mod.LoadedExecutable)
    assert obj.name == "LoadedExecutable"
    assert obj.value == 1


def test_class_Memory() -> None:
    """Test Memory."""
    obj = mod.Memory(name="Memory", value=1)
    assert isinstance(obj, mod.Memory)
    assert obj.name == "Memory"
    assert obj.value == 1


def test_class_MpiCollectives() -> None:
    """Test MpiCollectives."""
    obj = mod.MpiCollectives(name="MpiCollectives", value=1)
    assert isinstance(obj, mod.MpiCollectives)
    assert obj.name == "MpiCollectives"
    assert obj.value == 1


def test_class_NamedSharding() -> None:
    """Test NamedSharding."""
    obj = mod.NamedSharding(name="NamedSharding", value=1)
    assert isinstance(obj, mod.NamedSharding)
    assert obj.name == "NamedSharding"
    assert obj.value == 1


def test_class_OpSharding() -> None:
    """Test OpSharding."""
    obj = mod.OpSharding(name="OpSharding", value=1)
    assert isinstance(obj, mod.OpSharding)
    assert obj.name == "OpSharding"
    assert obj.value == 1


def test_class_OpSharding_ShardGroupType() -> None:
    """Test OpSharding_ShardGroupType."""
    obj = mod.OpSharding_ShardGroupType(name="OpSharding_ShardGroupType", value=1)
    assert isinstance(obj, mod.OpSharding_ShardGroupType)
    assert obj.name == "OpSharding_ShardGroupType"
    assert obj.value == 1


def test_class_OpSharding_Type() -> None:
    """Test OpSharding_Type."""
    obj = mod.OpSharding_Type(name="OpSharding_Type", value=1)
    assert isinstance(obj, mod.OpSharding_Type)
    assert obj.name == "OpSharding_Type"
    assert obj.value == 1


def test_class_PjRtLayout() -> None:
    """Test PjRtLayout."""
    obj = mod.PjRtLayout(name="PjRtLayout", value=1)
    assert isinstance(obj, mod.PjRtLayout)
    assert obj.name == "PjRtLayout"
    assert obj.value == 1


def test_class_PjRtXlaLayout() -> None:
    """Test PjRtXlaLayout."""
    obj = mod.PjRtXlaLayout(name="PjRtXlaLayout", value=1)
    assert isinstance(obj, mod.PjRtXlaLayout)
    assert obj.name == "PjRtXlaLayout"
    assert obj.value == 1


def test_class_PjitFunction() -> None:
    """Test PjitFunction."""
    obj = mod.PjitFunction(name="PjitFunction", value=1)
    assert isinstance(obj, mod.PjitFunction)
    assert obj.name == "PjitFunction"
    assert obj.value == 1


def test_class_PjitFunctionCache() -> None:
    """Test PjitFunctionCache."""
    obj = mod.PjitFunctionCache(name="PjitFunctionCache", value=1)
    assert isinstance(obj, mod.PjitFunctionCache)
    assert obj.name == "PjitFunctionCache"
    assert obj.value == 1


def test_class_PmapFunction() -> None:
    """Test PmapFunction."""
    obj = mod.PmapFunction(name="PmapFunction", value=1)
    assert isinstance(obj, mod.PmapFunction)
    assert obj.name == "PmapFunction"
    assert obj.value == 1


def test_class_PmapSharding() -> None:
    """Test PmapSharding."""
    obj = mod.PmapSharding(name="PmapSharding", value=1)
    assert isinstance(obj, mod.PmapSharding)
    assert obj.name == "PmapSharding"
    assert obj.value == 1


def test_class_PrecisionConfig_Precision() -> None:
    """Test PrecisionConfig_Precision."""
    obj = mod.PrecisionConfig_Precision(name="PrecisionConfig_Precision", value=1)
    assert isinstance(obj, mod.PrecisionConfig_Precision)
    assert obj.name == "PrecisionConfig_Precision"
    assert obj.value == 1


def test_class_PreemptionSyncManager() -> None:
    """Test PreemptionSyncManager."""
    obj = mod.PreemptionSyncManager(name="PreemptionSyncManager", value=1)
    assert isinstance(obj, mod.PreemptionSyncManager)
    assert obj.name == "PreemptionSyncManager"
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


def test_class_PyTreeRegistry() -> None:
    """Test PyTreeRegistry."""
    obj = mod.PyTreeRegistry(name="PyTreeRegistry", value=1)
    assert isinstance(obj, mod.PyTreeRegistry)
    assert obj.name == "PyTreeRegistry"
    assert obj.value == 1


def test_class_ResultHandler() -> None:
    """Test ResultHandler."""
    obj = mod.ResultHandler(name="ResultHandler", value=1)
    assert isinstance(obj, mod.ResultHandler)
    assert obj.name == "ResultHandler"
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


def test_class_ShardedToken() -> None:
    """Test ShardedToken."""
    obj = mod.ShardedToken(name="ShardedToken", value=1)
    assert isinstance(obj, mod.ShardedToken)
    assert obj.name == "ShardedToken"
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


def test_class_Token() -> None:
    """Test Token."""
    obj = mod.Token(name="Token", value=1)
    assert isinstance(obj, mod.Token)
    assert obj.name == "Token"
    assert obj.value == 1


def test_class_Traceback() -> None:
    """Test Traceback."""
    obj = mod.Traceback(name="Traceback", value=1)
    assert isinstance(obj, mod.Traceback)
    assert obj.name == "Traceback"
    assert obj.value == 1


def test_class_TupleSimplifier() -> None:
    """Test TupleSimplifier."""
    obj = mod.TupleSimplifier(name="TupleSimplifier", value=1)
    assert isinstance(obj, mod.TupleSimplifier)
    assert obj.name == "TupleSimplifier"
    assert obj.value == 1


def test_class_WeakrefLRUCache() -> None:
    """Test WeakrefLRUCache."""
    obj = mod.WeakrefLRUCache(name="WeakrefLRUCache", value=1)
    assert isinstance(obj, mod.WeakrefLRUCache)
    assert obj.name == "WeakrefLRUCache"
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
