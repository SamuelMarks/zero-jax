"""Tests for tests/test_eager_state.py."""

from __future__ import annotations

import math

from ml_switcheroo_compiler.core.config import config

from tests.conftest import assert_allclose
from zero_jax import numpy as jnp_zero
from zero_jax._types import ArrayLikeSpec


def test_jnp_sin(switcheroo_config: ArrayLikeSpec) -> None:
    """Execute ``test_jnp_sin``.

    Args:
        switcheroo_config: Parameter ``switcheroo_config``.
    """
    assert config.eager_mode is True
    vals = [0.0, 1.0, 3.14159]
    x = jnp_zero.array(vals)
    zero_val = jnp_zero.sin(x)
    assert_allclose(zero_val, [math.sin(v) for v in vals])
