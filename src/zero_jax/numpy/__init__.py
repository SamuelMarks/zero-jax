"""numpy API routing for zero-jax."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue

from . import fft, linalg
from .config import *
from .constants import *
from .lax_numpy import *
from .lax_numpy import _to_tensor, _wrap
from .polynomial import *


class ComplexWarning(Warning):
    """Frontend class for ``ComplexWarning``."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """
        super().__init__(*args, **kwargs)
