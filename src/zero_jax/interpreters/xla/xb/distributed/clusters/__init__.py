"""Frontend API routing for jax.interpreters.xla.xb.distributed.clusters."""

from typing import Any


class ClusterEnv:
    """Interface for defining a cluster environment."""

    pass


class GceTpuCluster:
    """Mock implementation for GceTpuCluster."""

    pass


class GkeTpuCluster:
    """Mock implementation for GkeTpuCluster."""

    pass


class OmpiCluster:
    """Mock implementation for OmpiCluster."""

    pass


class SlurmCluster:
    """Mock implementation for SlurmCluster."""

    pass


from . import cloud_tpu_cluster
from . import cluster
from . import ompi_cluster
from . import slurm_cluster
