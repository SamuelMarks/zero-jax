"""Frontend API routing for jax.experimental.x64_context.config.jax_jit."""

from typing import Any


class JitState:
    """Mock implementation for JitState."""

    pass


class PyArgSignature:
    """Mock implementation for PyArgSignature."""

    pass


def get_enable_x64(*args: Any, **kwargs: Any) -> Any:
    """get_enable_x64() -> bool"""
    raise NotImplementedError("get_enable_x64 not yet implemented in zero-jax")


def global_state(*args: Any, **kwargs: Any) -> Any:
    """global_state() -> jaxlib.xla_extension.jax_jit.JitState"""
    raise NotImplementedError("global_state not yet implemented in zero-jax")


def set_thread_local_state_initialization_callback(*args: Any, **kwargs: Any) -> Any:
    """set_thread_local_state_initialization_callback(arg: object, /) -> None"""
    raise NotImplementedError(
        "set_thread_local_state_initialization_callback not yet implemented in zero-jax"
    )


def swap_thread_local_state_disable_jit(*args: Any, **kwargs: Any) -> Any:
    """swap_thread_local_state_disable_jit(value: Optional[bool]) -> Optional[bool]"""
    raise NotImplementedError(
        "swap_thread_local_state_disable_jit not yet implemented in zero-jax"
    )


def thread_local_state(*args: Any, **kwargs: Any) -> Any:
    """thread_local_state() -> jaxlib.xla_extension.jax_jit.JitState"""
    raise NotImplementedError("thread_local_state not yet implemented in zero-jax")
