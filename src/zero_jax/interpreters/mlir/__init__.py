"""Frontend API routing for jax.interpreters.mlir."""

from typing import Any


def AxisContext(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for AxisContext."""
    raise NotImplementedError("AxisContext not yet implemented in zero-jax")


class ConstantHandler:
    """Mock implementation for ConstantHandler."""

    pass


DEVICE_TO_DEVICE_TYPE: Any = None


class LoweringParameters:
    """LoweringParameters(override_lowering_rules: 'tuple[tuple[core.Primitive, LoweringRule]] | None' = None, global_constant_computation: 'bool' = False, for_export: 'bool' = False)"""

    pass


class LoweringResult:
    """LoweringResult(module, keepalive, host_callbacks, shape_poly_state)"""

    pass


class LoweringRule:
    """Mock implementation for LoweringRule."""

    pass


class LoweringRuleContext:
    """Per-rule context information for MLIR lowering."""

    pass


class Mesh:
    """Declare the hardware resources available in the scope of this manager."""

    pass


def MeshAxisName(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    raise NotImplementedError("MeshAxisName not yet implemented in zero-jax")


class ModuleContext:
    """Module-wide context information for MLIR lowering."""

    pass


RECV_FROM_HOST_TYPE: Any = None


class ReplicaAxisContext:
    """A hardware axis context for parallel computations that are partitioned by JAX."""

    pass


SEND_TO_HOST_TYPE: Any = None


class SPMDAxisContext:
    """A hardware axis context for parallel computations that use the GSPMD partitioner."""

    pass


class ShapePolyLoweringState:
    """Mock implementation for ShapePolyLoweringState."""

    pass


class ShardingContext:
    """A hardware axis context for parallel computations that use the sharding"""

    pass


class Token:
    """All the operations on a read-only sequence."""

    pass


class TokenSet:
    """An immutable container of tokens to be used to lower effectful jaxprs. When lowering"""

    pass


def Value(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    raise NotImplementedError("Value not yet implemented in zero-jax")


def aval_to_ir_type(*args: Any, **kwargs: Any) -> Any:
    """Convenience wrapper around aval_to_ir_types for single types."""
    raise NotImplementedError("aval_to_ir_type not yet implemented in zero-jax")


def aval_to_ir_types(*args: Any, **kwargs: Any) -> Any:
    """Converts a JAX aval to zero or more MLIR IR types."""
    raise NotImplementedError("aval_to_ir_types not yet implemented in zero-jax")


def core_call_lowering(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for core_call_lowering."""
    raise NotImplementedError("core_call_lowering not yet implemented in zero-jax")


def custom_call(*args: Any, **kwargs: Any) -> Any:
    """Helper function for building an hlo.CustomCall."""
    raise NotImplementedError("custom_call not yet implemented in zero-jax")


def dense_bool_array(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dense_bool_array."""
    raise NotImplementedError("dense_bool_array not yet implemented in zero-jax")


def dense_bool_elements(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dense_bool_elements."""
    raise NotImplementedError("dense_bool_elements not yet implemented in zero-jax")


def dense_int_array(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dense_int_array."""
    raise NotImplementedError("dense_int_array not yet implemented in zero-jax")


def dense_int_elements(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dense_int_elements."""
    raise NotImplementedError("dense_int_elements not yet implemented in zero-jax")


def dtype_to_ir_type(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dtype_to_ir_type."""
    raise NotImplementedError("dtype_to_ir_type not yet implemented in zero-jax")


def emit_python_callback(*args: Any, **kwargs: Any) -> Any:
    """Emits MLIR that calls back to a provided Python function."""
    raise NotImplementedError("emit_python_callback not yet implemented in zero-jax")


def flatten_lowering_ir_args(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for flatten_lowering_ir_args."""
    raise NotImplementedError(
        "flatten_lowering_ir_args not yet implemented in zero-jax"
    )


from . import func_dialect
from . import hlo


def i32_attr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for i32_attr."""
    raise NotImplementedError("i32_attr not yet implemented in zero-jax")


def i64_attr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for i64_attr."""
    raise NotImplementedError("i64_attr not yet implemented in zero-jax")


from . import ir


def ir_constant(*args: Any, **kwargs: Any) -> Any:
    """Convenience wrapper around ir_constants for singleton values."""
    raise NotImplementedError("ir_constant not yet implemented in zero-jax")


def ir_constants(*args: Any, **kwargs: Any) -> Any:
    """Translate a Python `val` to an IR constant, canonicalizing its dtype."""
    raise NotImplementedError("ir_constants not yet implemented in zero-jax")


ir_type_handlers: Any = None


def jaxpr_subcomp(*args: Any, **kwargs: Any) -> Any:
    """Lowers a jaxpr into MLIR, inlined into an existing function."""
    raise NotImplementedError("jaxpr_subcomp not yet implemented in zero-jax")


def lower_fun(*args: Any, **kwargs: Any) -> Any:
    """Converts a traceable JAX function `fun` into a lowering rule."""
    raise NotImplementedError("lower_fun not yet implemented in zero-jax")


def lower_jaxpr_to_fun(*args: Any, **kwargs: Any) -> Any:
    """Lowers jaxpr and its callees to an IR function."""
    raise NotImplementedError("lower_jaxpr_to_fun not yet implemented in zero-jax")


def lower_jaxpr_to_module(*args: Any, **kwargs: Any) -> Any:
    """Lowers a top-level jaxpr to an MLIR module."""
    raise NotImplementedError("lower_jaxpr_to_module not yet implemented in zero-jax")


lowerable_effects: Any = None


def make_ir_context(*args: Any, **kwargs: Any) -> Any:
    """Creates an MLIR context suitable for JAX IR."""
    raise NotImplementedError("make_ir_context not yet implemented in zero-jax")


def merge_mlir_modules(*args: Any, **kwargs: Any) -> Any:
    """Args:"""
    raise NotImplementedError("merge_mlir_modules not yet implemented in zero-jax")


def module_to_bytecode(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for module_to_bytecode."""
    raise NotImplementedError("module_to_bytecode not yet implemented in zero-jax")


def module_to_string(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for module_to_string."""
    raise NotImplementedError("module_to_string not yet implemented in zero-jax")


def register_constant_handler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_constant_handler."""
    raise NotImplementedError(
        "register_constant_handler not yet implemented in zero-jax"
    )


def register_lowering(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_lowering."""
    raise NotImplementedError("register_lowering not yet implemented in zero-jax")


def shape_tensor(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for shape_tensor."""
    raise NotImplementedError("shape_tensor not yet implemented in zero-jax")


def token_type(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for token_type."""
    raise NotImplementedError("token_type not yet implemented in zero-jax")


def xla_computation_to_mlir_module(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for xla_computation_to_mlir_module."""
    raise NotImplementedError(
        "xla_computation_to_mlir_module not yet implemented in zero-jax"
    )
