"""Frontend API routing for jax.interpreters.xla.xb."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops

BACKEND_TARGET: Any = None


def BackendFactory(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for BackendFactory."""
    return getattr(_ops, "BackendFactory")(*args, **kwargs)


class BackendRegistration:
    """BackendRegistration(factory: 'BackendFactory', priority: 'int', fail_quietly: 'bool' = False, experimental: 'bool' = False, c_api: 'Any | None' = None)"""

    pass


CUDA_VISIBLE_DEVICES: Any = None

MIN_COMPUTE_CAPABILITY: Any = None


class Mapping:
    """Mock implementation for Mapping."""

    pass


def TopologyFactory(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for TopologyFactory."""
    return getattr(_ops, "TopologyFactory")(*args, **kwargs)


class XlaBackend:
    """Mock implementation for XlaBackend."""

    pass


annotations: Any = None
from . import atexit


def backend_pjrt_c_api_version(*args: Any, **kwargs: Any) -> Any:
    """Returns the PJRT C API version of the backend."""
    return getattr(_ops, "backend_pjrt_c_api_version")(*args, **kwargs)


def backend_xla_version(*args: Any, **kwargs: Any) -> Any:
    """Returns the XLA version of the backend."""
    return getattr(_ops, "backend_xla_version")(*args, **kwargs)


def backends(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for backends."""
    return getattr(_ops, "backends")(*args, **kwargs)


def backends_are_initialized(*args: Any, **kwargs: Any) -> Any:
    """Returns true if backends have already been initialized."""
    return getattr(_ops, "backends_are_initialized")(*args, **kwargs)


def canonicalize_platform(*args: Any, **kwargs: Any) -> Any:
    """Replaces platform aliases with their concrete equivalent."""
    return getattr(_ops, "canonicalize_platform")(*args, **kwargs)


from . import config
from . import dataclasses


def default_backend(*args: Any, **kwargs: Any) -> Any:
    """Returns the platform name of the default XLA backend."""
    return getattr(_ops, "default_backend")(*args, **kwargs)


def device_count(*args: Any, **kwargs: Any) -> Any:
    """Returns the total number of devices."""
    return getattr(_ops, "device_count")(*args, **kwargs)


def devices(*args: Any, **kwargs: Any) -> Any:
    """Returns a list of all devices for a given backend."""
    return getattr(_ops, "devices")(*args, **kwargs)


def discover_pjrt_plugins(*args: Any, **kwargs: Any) -> Any:
    """Discovers plugins in the namespace package `jax_plugins` and import them."""
    return getattr(_ops, "discover_pjrt_plugins")(*args, **kwargs)


from . import distributed


def expand_platform_alias(*args: Any, **kwargs: Any) -> Any:
    """Expands, e.g., "gpu" to ["cuda", "rocm"]."""
    return getattr(_ops, "expand_platform_alias")(*args, **kwargs)


def get_backend(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_backend."""
    return getattr(_ops, "get_backend")(*args, **kwargs)


def get_device_backend(*args: Any, **kwargs: Any) -> Any:
    """Returns the Backend associated with `device`, or the default Backend."""
    return getattr(_ops, "get_device_backend")(*args, **kwargs)


def get_tpu_library_path(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_tpu_library_path."""
    return getattr(_ops, "get_tpu_library_path")(*args, **kwargs)


from . import hardware_utils


def host_count(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for host_count."""
    return getattr(_ops, "host_count")(*args, **kwargs)


def host_id(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for host_id."""
    return getattr(_ops, "host_id")(*args, **kwargs)


def host_ids(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for host_ids."""
    return getattr(_ops, "host_ids")(*args, **kwargs)


from . import importlib


def is_gpu(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_gpu."""
    return getattr(_ops, "is_gpu")(*args, **kwargs)


def is_known_platform(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_known_platform."""
    return getattr(_ops, "is_known_platform")(*args, **kwargs)


from . import jax_plugins
from . import json


def local_device_count(*args: Any, **kwargs: Any) -> Any:
    """Returns the number of devices addressable by this process."""
    return getattr(_ops, "local_device_count")(*args, **kwargs)


def local_devices(*args: Any, **kwargs: Any) -> Any:
    """Like :py:func:`jax.devices`, but only returns devices local to a given process."""
    return getattr(_ops, "local_devices")(*args, **kwargs)


logger: Any = None
from . import logging


def lru_cache(*args: Any, **kwargs: Any) -> Any:
    """Least-recently-used cache decorator."""
    return getattr(_ops, "lru_cache")(*args, **kwargs)


def make_cpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_cpu_client."""
    return getattr(_ops, "make_cpu_client")(*args, **kwargs)


def make_gpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_gpu_client."""
    return getattr(_ops, "make_gpu_client")(*args, **kwargs)


def make_pjrt_topology(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_pjrt_topology."""
    return getattr(_ops, "make_pjrt_topology")(*args, **kwargs)


def make_pjrt_tpu_topology(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_pjrt_tpu_topology."""
    return getattr(_ops, "make_pjrt_tpu_topology")(*args, **kwargs)


from . import os


class partial:
    """partial(func, *args, **keywords) - new function with partial application"""

    pass


from . import pkgutil


def process_count(*args: Any, **kwargs: Any) -> Any:
    """Returns the number of JAX processes associated with the backend."""
    return getattr(_ops, "process_count")(*args, **kwargs)


def process_index(*args: Any, **kwargs: Any) -> Any:
    """Returns the integer process index of this process."""
    return getattr(_ops, "process_index")(*args, **kwargs)


from . import py_platform


def register_backend_factory(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_backend_factory."""
    return getattr(_ops, "register_backend_factory")(*args, **kwargs)


def register_pjrt_plugin_factories_from_env(*args: Any, **kwargs: Any) -> Any:
    """Registers backend factories for PJRT plugins."""
    return getattr(_ops, "register_pjrt_plugin_factories_from_env")(*args, **kwargs)


def register_plugin(*args: Any, **kwargs: Any) -> Any:
    """Registers a backend factory for the PJRT plugin."""
    return getattr(_ops, "register_plugin")(*args, **kwargs)


def register_plugin_callbacks(*args: Any, **kwargs: Any) -> Any:
    """Registers a callback to be called with c_api after plugins discovery."""
    return getattr(_ops, "register_plugin_callbacks")(*args, **kwargs)


from . import sys
from . import threading


def tpu_client_timer_callback(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tpu_client_timer_callback."""
    return getattr(_ops, "tpu_client_timer_callback")(*args, **kwargs)


from . import traceback
from . import traceback_util


def using_pjrt_c_api(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for using_pjrt_c_api."""
    return getattr(_ops, "using_pjrt_c_api")(*args, **kwargs)


from . import util
from . import warnings
from . import xla_client
from . import xla_extension

xla_extension_version: Any = None
