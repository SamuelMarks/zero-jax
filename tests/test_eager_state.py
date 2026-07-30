import numpy as np
import pytest
from ml_switcheroo_compiler.core.config import config

from zero_jax import numpy as jnp_zero


def test_jnp_sin(switcheroo_config):
    print("In test, eager_mode:", config.eager_mode)
    x = np.array([0.0, 1.0, 3.14159])
    zero_val = jnp_zero.sin(x)
    print("zero_val:", np.array(zero_val))
