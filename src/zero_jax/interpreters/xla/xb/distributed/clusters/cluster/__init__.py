"""Frontend API routing for jax.interpreters.xla.xb.distributed.clusters.cluster."""

from typing import Any


class ClusterEnv:
    """Interface for defining a cluster environment."""

    pass


annotations: Any = None

logger: Any = None
from . import logging

running_in_cloud_tpu_vm: Any = None
