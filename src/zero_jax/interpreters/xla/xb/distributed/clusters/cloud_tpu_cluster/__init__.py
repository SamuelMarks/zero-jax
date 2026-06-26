"""Frontend API routing for jax.interpreters.xla.xb.distributed.clusters.cloud_tpu_cluster."""

from typing import Any


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
    raise NotImplementedError("get_metadata not yet implemented in zero-jax")


def get_tpu_env_value(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_tpu_env_value."""
    raise NotImplementedError("get_tpu_env_value not yet implemented in zero-jax")


def has_megascale_address(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for has_megascale_address."""
    raise NotImplementedError("has_megascale_address not yet implemented in zero-jax")


logger: Any = None
from . import logging

metadata_response_code_success: Any = None
from . import os
from . import re

running_in_cloud_tpu_vm: Any = None
from . import socket
from . import time
