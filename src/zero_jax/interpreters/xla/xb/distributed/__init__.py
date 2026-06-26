"""Frontend API routing for jax.interpreters.xla.xb.distributed."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


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
    return getattr(_ops, "initialize")(*args, **kwargs)


logger: Any = None
from . import logging
from . import os


def shutdown(*args: Any, **kwargs: Any) -> Any:
    """Shuts down the distributed system."""
    return getattr(_ops, "shutdown")(*args, **kwargs)


from . import xla_bridge
from . import xla_extension
