"""Frontend API routing for jax.interpreters.xla.xb.distributed.clusters.cloud_tpu_cluster."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class BaseTpuCluster:
    """Abstract cluster supports both single and multislice TPU environments."""

    pass


class GceTpuCluster:
    """Mock implementation for GceTpuCluster."""

    pass


class GkeTpuCluster:
    """Mock implementation for GkeTpuCluster."""

    pass


annotations: Any = None
from . import clusters

coordinator_port: Any = None


def get_metadata(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_metadata."""
    return getattr(_ops, "get_metadata")(*args, **kwargs)


def get_tpu_env_value(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_tpu_env_value."""
    return getattr(_ops, "get_tpu_env_value")(*args, **kwargs)


def has_megascale_address(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for has_megascale_address."""
    return getattr(_ops, "has_megascale_address")(*args, **kwargs)


logger: Any = None
from . import logging

metadata_response_code_success: Any = None
from . import os
from . import re

running_in_cloud_tpu_vm: Any = None
from . import socket
from . import time
