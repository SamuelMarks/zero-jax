"""Tests for tests/test_eager_state.py."""

from __future__ import annotations

import numpy as np
from ml_switcheroo_compiler.core.config import config

from zero_jax import numpy as jnp_zero
from zero_jax._types import ArrayLikeSpec


def test_jnp_sin(switcheroo_config: ArrayLikeSpec) -> None:
    """Execute ``test_jnp_sin``."""
    print("In test, eager_mode:", config.eager_mode)
    x = np.array([0.0, 1.0, 3.14159])
    zero_val = jnp_zero.sin(x)
    print("zero_val:", np.array(zero_val))
