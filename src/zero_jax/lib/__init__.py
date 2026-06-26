"""Mock implementation for jax.lib."""

from . import xla_bridge
from . import xla_client
from . import xla_extension

__all__ = ["xla_bridge", "xla_client", "xla_extension"]
