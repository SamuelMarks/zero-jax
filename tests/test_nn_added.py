import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import zero_jax.nn as jnn
import zero_jax.numpy as jnp


def test_nn_added():
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    assert jnn.glu(jnp.array([1.0, 2.0, 3.0, 4.0])).shape is not None
    assert jnn.hard_silu(x).shape is not None
    assert jnn.hard_swish(x).shape is not None
    assert jnn.leaky_relu(x).shape is not None
    assert jnn.mish(x).shape is not None
    assert jnn.soft_sign(x).shape is not None
    assert jnn.softplus(x).shape is not None
    assert jnn.sparse_plus(x).shape is not None
    assert jnn.sparse_sigmoid(x).shape is not None
    assert jnn.squareplus(x).shape is not None
    assert jnn.standardize(x).shape is not None
