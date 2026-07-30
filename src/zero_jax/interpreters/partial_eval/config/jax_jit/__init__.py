"""Module jax_jit."""

import typing
from typing import Any

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


def JitState(*args: Any, **kwargs: Any) -> Any:
    return None


def PyArgSignature(*args: Any, **kwargs: Any) -> Any:
    return None


def get_enable_x64(*args: Any, **kwargs: Any) -> Any:
    return None


def global_state(*args: Any, **kwargs: Any) -> Any:
    return None


def set_thread_local_state_initialization_callback(*args: Any, **kwargs: Any) -> Any:
    return None


def swap_thread_local_state_disable_jit(*args: Any, **kwargs: Any) -> Any:
    return None


def thread_local_state(*args: Any, **kwargs: Any) -> Any:
    return None
