"""Main initialization module for zero_jax."""

from __future__ import annotations

from typing import Any
import ml_switcheroo

from . import tree_util
from . import lax
from .api import jit, grad, value_and_grad, vmap, disable_jit, pmap, eval_shape
from . import random
from . import nn
from . import experimental

__all__ = [
    "tree_util",
    "lax",
    "jit",
    "grad",
    "value_and_grad",
    "vmap",
    "disable_jit",
    "pmap",
    "eval_shape",
    "random",
    "nn",
    "experimental",
]

from zero_jax.numpy.lax_numpy import ndarray as Array

__all__ += ["Array"]


class Device:
    """Represents a computational device.

    Attributes:
        platform: The platform of the device (e.g., 'cpu', 'gpu').
    """

    def __init__(self, platform: Any = "cpu") -> None:
        """Initializes a Device.

        Args:
            platform: The hardware platform.
        """
        self.platform = platform


def devices(backend: Any = None) -> Any:
    """Gets the available devices.

    Args:
        backend: The backend to query for devices.

    Returns:
        A list of available Device objects.
    """
    return [Device(platform="cpu")]


def local_devices(backend: Any = None) -> Any:
    """Gets the available local devices.

    Args:
        backend: The backend to query for local devices.

    Returns:
        A list of available local Device objects.
    """
    return [Device(platform="cpu")]


__all__ += ["Device", "devices", "local_devices"]


def device_get(x: Any) -> Any:
    """Retrieves data from a device to the host.

    Args:
        x: The array or tree of arrays to retrieve.

    Returns:
        The host-backed array(s).
    """
    return x


__all__.append("device_get")
