"""Frontend API routing for jax.interpreters.mlir."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def AxisContext(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for AxisContext."""
    return getattr(_ops, "AxisContext")(*args, **kwargs)


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
    return getattr(_ops, "MeshAxisName")(*args, **kwargs)


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
    return getattr(_ops, "Value")(*args, **kwargs)


def aval_to_ir_type(*args: Any, **kwargs: Any) -> Any:
    """Convenience wrapper around aval_to_ir_types for single types."""
    return getattr(_ops, "aval_to_ir_type")(*args, **kwargs)


def aval_to_ir_types(*args: Any, **kwargs: Any) -> Any:
    """Converts a JAX aval to zero or more MLIR IR types."""
    return getattr(_ops, "aval_to_ir_types")(*args, **kwargs)


def core_call_lowering(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for core_call_lowering."""
    return getattr(_ops, "core_call_lowering")(*args, **kwargs)


def custom_call(*args: Any, **kwargs: Any) -> Any:
    """Helper function for building an hlo.CustomCall."""
    return getattr(_ops, "custom_call")(*args, **kwargs)


def dense_bool_array(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dense_bool_array."""
    return getattr(_ops, "dense_bool_array")(*args, **kwargs)


def dense_bool_elements(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dense_bool_elements."""
    return getattr(_ops, "dense_bool_elements")(*args, **kwargs)


def dense_int_array(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dense_int_array."""
    return getattr(_ops, "dense_int_array")(*args, **kwargs)


def dense_int_elements(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dense_int_elements."""
    return getattr(_ops, "dense_int_elements")(*args, **kwargs)


def dtype_to_ir_type(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for dtype_to_ir_type."""
    return getattr(_ops, "dtype_to_ir_type")(*args, **kwargs)


def emit_python_callback(*args: Any, **kwargs: Any) -> Any:
    """Emits MLIR that calls back to a provided Python function."""
    return getattr(_ops, "emit_python_callback")(*args, **kwargs)


def flatten_lowering_ir_args(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for flatten_lowering_ir_args."""
    return getattr(_ops, "flatten_lowering_ir_args")(*args, **kwargs)


from . import func_dialect
from . import hlo


def i32_attr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for i32_attr."""
    return getattr(_ops, "i32_attr")(*args, **kwargs)


def i64_attr(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for i64_attr."""
    return getattr(_ops, "i64_attr")(*args, **kwargs)


from . import ir


def ir_constant(*args: Any, **kwargs: Any) -> Any:
    """Convenience wrapper around ir_constants for singleton values."""
    return getattr(_ops, "ir_constant")(*args, **kwargs)


def ir_constants(*args: Any, **kwargs: Any) -> Any:
    """Translate a Python `val` to an IR constant, canonicalizing its dtype."""
    return getattr(_ops, "ir_constants")(*args, **kwargs)


ir_type_handlers: Any = None


def jaxpr_subcomp(*args: Any, **kwargs: Any) -> Any:
    """Lowers a jaxpr into MLIR, inlined into an existing function."""
    return getattr(_ops, "jaxpr_subcomp")(*args, **kwargs)


def lower_fun(*args: Any, **kwargs: Any) -> Any:
    """Converts a traceable JAX function `fun` into a lowering rule."""
    return getattr(_ops, "lower_fun")(*args, **kwargs)


def lower_jaxpr_to_fun(*args: Any, **kwargs: Any) -> Any:
    """Lowers jaxpr and its callees to an IR function."""
    return getattr(_ops, "lower_jaxpr_to_fun")(*args, **kwargs)


def lower_jaxpr_to_module(*args: Any, **kwargs: Any) -> Any:
    """Lowers a top-level jaxpr to an MLIR module."""
    return getattr(_ops, "lower_jaxpr_to_module")(*args, **kwargs)


lowerable_effects: Any = None


def make_ir_context(*args: Any, **kwargs: Any) -> Any:
    """Creates an MLIR context suitable for JAX IR."""
    return getattr(_ops, "make_ir_context")(*args, **kwargs)


def merge_mlir_modules(*args: Any, **kwargs: Any) -> Any:
    """Args:"""
    return getattr(_ops, "merge_mlir_modules")(*args, **kwargs)


def module_to_bytecode(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for module_to_bytecode."""
    return getattr(_ops, "module_to_bytecode")(*args, **kwargs)


def module_to_string(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for module_to_string."""
    return getattr(_ops, "module_to_string")(*args, **kwargs)


def register_constant_handler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_constant_handler."""
    return getattr(_ops, "register_constant_handler")(*args, **kwargs)


def register_lowering(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_lowering."""
    return getattr(_ops, "register_lowering")(*args, **kwargs)


def shape_tensor(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for shape_tensor."""
    return getattr(_ops, "shape_tensor")(*args, **kwargs)


def token_type(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for token_type."""
    return getattr(_ops, "token_type")(*args, **kwargs)


def xla_computation_to_mlir_module(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for xla_computation_to_mlir_module."""
    return getattr(_ops, "xla_computation_to_mlir_module")(*args, **kwargs)
