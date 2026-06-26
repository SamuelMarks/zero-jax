"""Frontend API routing for jax.interpreters.xla.xb.distributed."""

from typing import Any


class State:
    """Mock implementation for State."""

    pass


annotations: Any = None
from . import atexit
from . import clusters
from . import config

global_state: Any = None


def initialize(*args: Any, **kwargs: Any) -> Any:
    """Initializes the JAX distributed system."""
    raise NotImplementedError("initialize not yet implemented in zero-jax")


logger: Any = None
from . import logging
from . import os


def shutdown(*args: Any, **kwargs: Any) -> Any:
    """Shuts down the distributed system."""
    raise NotImplementedError("shutdown not yet implemented in zero-jax")


from . import xla_bridge
from . import xla_extension
