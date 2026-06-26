"""Frontend API routing for jax.experimental.x64_context.config.jax_jit."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class JitState:
    """Mock implementation for JitState."""

    pass


class PyArgSignature:
    """Mock implementation for PyArgSignature."""

    pass


def get_enable_x64(*args: Any, **kwargs: Any) -> Any:
    """get_enable_x64() -> bool"""
    return getattr(_ops, "get_enable_x64")(*args, **kwargs)


def global_state(*args: Any, **kwargs: Any) -> Any:
    """global_state() -> jaxlib.xla_extension.jax_jit.JitState"""
    return getattr(_ops, "global_state")(*args, **kwargs)


def set_thread_local_state_initialization_callback(*args: Any, **kwargs: Any) -> Any:
    """set_thread_local_state_initialization_callback(arg: object, /) -> None"""
    return getattr(_ops, "set_thread_local_state_initialization_callback")(
        *args, **kwargs
    )


def swap_thread_local_state_disable_jit(*args: Any, **kwargs: Any) -> Any:
    """swap_thread_local_state_disable_jit(value: Optional[bool]) -> Optional[bool]"""
    return getattr(_ops, "swap_thread_local_state_disable_jit")(*args, **kwargs)


def thread_local_state(*args: Any, **kwargs: Any) -> Any:
    """thread_local_state() -> jaxlib.xla_extension.jax_jit.JitState"""
    return getattr(_ops, "thread_local_state")(*args, **kwargs)
