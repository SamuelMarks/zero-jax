"""Frontend API routing for jax.interpreters.xla.xb."""

from typing import Any

BACKEND_TARGET: Any = None


def BackendFactory(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for BackendFactory."""
    raise NotImplementedError("BackendFactory not yet implemented in zero-jax")


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
    raise NotImplementedError("TopologyFactory not yet implemented in zero-jax")


class XlaBackend:
    """Mock implementation for XlaBackend."""

    pass


annotations: Any = None
from . import atexit


def backend_pjrt_c_api_version(*args: Any, **kwargs: Any) -> Any:
    """Returns the PJRT C API version of the backend."""
    raise NotImplementedError(
        "backend_pjrt_c_api_version not yet implemented in zero-jax"
    )


def backend_xla_version(*args: Any, **kwargs: Any) -> Any:
    """Returns the XLA version of the backend."""
    raise NotImplementedError("backend_xla_version not yet implemented in zero-jax")


def backends(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for backends."""
    raise NotImplementedError("backends not yet implemented in zero-jax")


def backends_are_initialized(*args: Any, **kwargs: Any) -> Any:
    """Returns true if backends have already been initialized."""
    raise NotImplementedError(
        "backends_are_initialized not yet implemented in zero-jax"
    )


def canonicalize_platform(*args: Any, **kwargs: Any) -> Any:
    """Replaces platform aliases with their concrete equivalent."""
    raise NotImplementedError("canonicalize_platform not yet implemented in zero-jax")


from . import config
from . import dataclasses


def default_backend(*args: Any, **kwargs: Any) -> Any:
    """Returns the platform name of the default XLA backend."""
    raise NotImplementedError("default_backend not yet implemented in zero-jax")


def device_count(*args: Any, **kwargs: Any) -> Any:
    """Returns the total number of devices."""
    raise NotImplementedError("device_count not yet implemented in zero-jax")


def devices(*args: Any, **kwargs: Any) -> Any:
    """Returns a list of all devices for a given backend."""
    raise NotImplementedError("devices not yet implemented in zero-jax")


def discover_pjrt_plugins(*args: Any, **kwargs: Any) -> Any:
    """Discovers plugins in the namespace package `jax_plugins` and import them."""
    raise NotImplementedError("discover_pjrt_plugins not yet implemented in zero-jax")


from . import distributed


def expand_platform_alias(*args: Any, **kwargs: Any) -> Any:
    """Expands, e.g., "gpu" to ["cuda", "rocm"]."""
    raise NotImplementedError("expand_platform_alias not yet implemented in zero-jax")


def get_backend(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_backend."""
    raise NotImplementedError("get_backend not yet implemented in zero-jax")


def get_device_backend(*args: Any, **kwargs: Any) -> Any:
    """Returns the Backend associated with `device`, or the default Backend."""
    raise NotImplementedError("get_device_backend not yet implemented in zero-jax")


def get_tpu_library_path(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_tpu_library_path."""
    raise NotImplementedError("get_tpu_library_path not yet implemented in zero-jax")


from . import hardware_utils


def host_count(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for host_count."""
    raise NotImplementedError("host_count not yet implemented in zero-jax")


def host_id(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for host_id."""
    raise NotImplementedError("host_id not yet implemented in zero-jax")


def host_ids(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for host_ids."""
    raise NotImplementedError("host_ids not yet implemented in zero-jax")


from . import importlib


def is_gpu(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_gpu."""
    raise NotImplementedError("is_gpu not yet implemented in zero-jax")


def is_known_platform(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_known_platform."""
    raise NotImplementedError("is_known_platform not yet implemented in zero-jax")


from . import jax_plugins
from . import json


def local_device_count(*args: Any, **kwargs: Any) -> Any:
    """Returns the number of devices addressable by this process."""
    raise NotImplementedError("local_device_count not yet implemented in zero-jax")


def local_devices(*args: Any, **kwargs: Any) -> Any:
    """Like :py:func:`jax.devices`, but only returns devices local to a given process."""
    raise NotImplementedError("local_devices not yet implemented in zero-jax")


logger: Any = None
from . import logging


def lru_cache(*args: Any, **kwargs: Any) -> Any:
    """Least-recently-used cache decorator."""
    raise NotImplementedError("lru_cache not yet implemented in zero-jax")


def make_cpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_cpu_client."""
    raise NotImplementedError("make_cpu_client not yet implemented in zero-jax")


def make_gpu_client(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_gpu_client."""
    raise NotImplementedError("make_gpu_client not yet implemented in zero-jax")


def make_pjrt_topology(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_pjrt_topology."""
    raise NotImplementedError("make_pjrt_topology not yet implemented in zero-jax")


def make_pjrt_tpu_topology(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for make_pjrt_tpu_topology."""
    raise NotImplementedError("make_pjrt_tpu_topology not yet implemented in zero-jax")


from . import os


class partial:
    """partial(func, *args, **keywords) - new function with partial application"""

    pass


from . import pkgutil


def process_count(*args: Any, **kwargs: Any) -> Any:
    """Returns the number of JAX processes associated with the backend."""
    raise NotImplementedError("process_count not yet implemented in zero-jax")


def process_index(*args: Any, **kwargs: Any) -> Any:
    """Returns the integer process index of this process."""
    raise NotImplementedError("process_index not yet implemented in zero-jax")


from . import py_platform


def register_backend_factory(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_backend_factory."""
    raise NotImplementedError(
        "register_backend_factory not yet implemented in zero-jax"
    )


def register_pjrt_plugin_factories_from_env(*args: Any, **kwargs: Any) -> Any:
    """Registers backend factories for PJRT plugins."""
    raise NotImplementedError(
        "register_pjrt_plugin_factories_from_env not yet implemented in zero-jax"
    )


def register_plugin(*args: Any, **kwargs: Any) -> Any:
    """Registers a backend factory for the PJRT plugin."""
    raise NotImplementedError("register_plugin not yet implemented in zero-jax")


def register_plugin_callbacks(*args: Any, **kwargs: Any) -> Any:
    """Registers a callback to be called with c_api after plugins discovery."""
    raise NotImplementedError(
        "register_plugin_callbacks not yet implemented in zero-jax"
    )


from . import sys
from . import threading


def tpu_client_timer_callback(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for tpu_client_timer_callback."""
    raise NotImplementedError(
        "tpu_client_timer_callback not yet implemented in zero-jax"
    )


from . import traceback
from . import traceback_util


def using_pjrt_c_api(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for using_pjrt_c_api."""
    raise NotImplementedError("using_pjrt_c_api not yet implemented in zero-jax")


from . import util
from . import warnings
from . import xla_client
from . import xla_extension

xla_extension_version: Any = None
