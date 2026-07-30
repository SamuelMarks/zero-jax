import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from ml_switcheroo_compiler.core.config import config

import zero_jax
from zero_jax import numpy as jnp


def test_devices():
    config.backend = "mlx"
    d = zero_jax.devices()
    assert len(d) == 1
    assert d[0].platform == "mlx"

    config.backend = "dask"
    assert zero_jax.local_devices()[0].platform == "dask"

    config.backend = "cupy"
    assert zero_jax.devices()[0].platform == "cupy"

    config.backend = "numpy"


def test_from_dlpack():
    try:
        import torch
    except ImportError:
        return

    x = torch.tensor([1, 2, 3])

    # zero_jax.dlpack.from_dlpack
    out = zero_jax.dlpack.from_dlpack(x)
    assert np.allclose(np.array(out), np.array([1, 2, 3]))

    # zero_jax.numpy.from_dlpack
    out2 = jnp.from_dlpack(x)
    assert np.allclose(np.array(out2), np.array([1, 2, 3]))
