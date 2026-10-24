"""Frontend API routing for zero-jax.random."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ConfigValue

from .missing_funcs import *
from .prng import *


def random_gamma_p(*args: ConfigValue, **kwargs: ConfigValue) -> ConfigValue:
    """Stub function for random_gamma_p.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ConfigValue: None stub return value.
    """
    return None
