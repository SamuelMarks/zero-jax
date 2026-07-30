"""Frontend API routing for jax.experimental.x64_context.config.jax_jit."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class JitState:
    """Frontend state holder for JitState."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PyArgSignature:
    """Frontend state holder for PyArgSignature."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def get_enable_x64(*args: Any, **kwargs: Any) -> Any:
    """get_enable_x64() -> bool"""
    return _ops.get_enable_x64(*args, **kwargs)


def global_state(*args: Any, **kwargs: Any) -> Any:
    """global_state() -> jaxlib.xla_extension.jax_jit.JitState"""
    return _ops.global_state(*args, **kwargs)


def set_thread_local_state_initialization_callback(*args: Any, **kwargs: Any) -> Any:
    """set_thread_local_state_initialization_callback(arg: object, /) -> None"""
    return _ops.set_thread_local_state_initialization_callback(*args, **kwargs)


def swap_thread_local_state_disable_jit(*args: Any, **kwargs: Any) -> Any:
    """swap_thread_local_state_disable_jit(value: Optional[bool]) -> Optional[bool]"""
    return _ops.swap_thread_local_state_disable_jit(*args, **kwargs)


def thread_local_state(*args: Any, **kwargs: Any) -> Any:
    """thread_local_state() -> jaxlib.xla_extension.jax_jit.JitState"""
    return _ops.thread_local_state(*args, **kwargs)
