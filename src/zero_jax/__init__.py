"""Module docstring."""

from . import numpy
from . import tree_util
from . import lax
from .api import jit, grad, value_and_grad, vmap, disable_jit, pmap
from . import random
from . import nn
from . import experimental

__all__ = [
    "numpy",
    "tree_util",
    "lax",
    "jit",
    "grad",
    "value_and_grad",
    "vmap",
    "disable_jit",
    "pmap",
    "random",
    "nn",
    "experimental",
]
import numpy as np

Array = np.ndarray
__all__ += ["Array"]


class Device:
    """Docstring."""

    def __init__(self, platform="cpu"):
        """Docstring."""
        self.platform = platform


def devices(backend=None):
    """Docstring."""
    return [Device("cpu")]


def local_devices(backend=None):
    """Docstring."""
    return [Device("cpu")]


__all__ += ["Device", "devices", "local_devices"]


def device_get(x):
    """Docstring."""
    return x


__all__.append("device_get")
