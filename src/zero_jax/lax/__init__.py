"""Frontend API routing for zero-jax.lax."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ConfigValue
from zero_jax.numpy.lax_numpy import ndarray as Array

from . import linalg
from .control_flow import *
from .missing_funcs import *
from .mock_p import *
from .primitives import *
from .types import *


class Device:
    """Device descriptor for JAX computations."""

    def __init__(self, platform: str = "cpu") -> None:
        """Initialize Device.

        Args:
            platform: Platform identifier string.
        """
        self.platform = platform
