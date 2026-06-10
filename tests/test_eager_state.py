import pytest
from zero_jax.numpy import lax_numpy as jnp_zero
import numpy as np
from ml_switcheroo.core.config import config


def test_jnp_sin(switcheroo_config):
    print("In test, eager_mode:", config.eager_mode)
    x = np.array([0.0, 1.0, 3.14159])
    zero_val = jnp_zero.sin(x)
    print("zero_val:", np.array(zero_val))
