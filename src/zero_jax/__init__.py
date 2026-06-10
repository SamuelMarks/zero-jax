"""Module docstring."""

from typing import Any
import ml_switcheroo

"Module docstring."
from . import numpy
from . import tree_util
from . import lax
from .api import jit, grad, value_and_grad, vmap, disable_jit, pmap, eval_shape
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
    "pmap, eval_shape",
    "random",
    "nn",
    "experimental",
]
from .numpy.lax_numpy import ndarray as Array

__all__ += ["Array"]


class Device:
    """Docstring."""

    def __init__(self, platform: Any = "cpu") -> None:
        """Initialize."""
        self.platform = platform


def devices(backend: Any = None) -> Any:
    """Devices function."""
    return [Device(platform="cpu")]


def local_devices(backend: Any = None) -> Any:
    """local_devices function."""
    return [Device(platform="cpu")]


__all__ += ["Device", "devices", "local_devices"]


def device_get(x: Any) -> Any:
    """device_get function."""
    return x


__all__.append("device_get")
