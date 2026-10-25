"""Tests for tests/test_backends.py."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from ml_switcheroo_compiler.core.config import config

import zero_jax
from tests.conftest import assert_allclose, assert_array_equal
from zero_jax import _compiler_patches, _compiler_proxy_creation
from zero_jax import numpy as jnp
from zero_jax import random as jrandom


def test_devices() -> None:
    """Verify device platform reporting across backends."""
    config.backend = "cpu"
    d = zero_jax.devices()
    assert len(d) == 1
    assert d[0].platform == "cpu"

    config.backend = "dask"
    assert zero_jax.local_devices()[0].platform == "dask"

    config.backend = "cupy"
    assert zero_jax.devices()[0].platform == "cupy"

    config.backend = "numpy"


def test_deterministic_prng_compiler_patches() -> None:
    """Verify deterministic PRNG behavior and key seeding in ``_compiler_patches``."""
    _compiler_patches.register_compiler_patches()
    assert jrandom.normal(jnp.array([], dtype=jnp.uint32), shape=(2,)).shape == (2,)
    assert _compiler_patches.numpy_eager_registry.get("RandomNormal")(jnp, shape=(2,)).shape == (2,)
    assert _compiler_patches.numpy_eager_registry.get("RandomRandint")(jnp, shape=(2,), minval=0, maxval=5).shape == (2,)
    assert _compiler_patches.numpy_eager_registry.get("RandomBernoulli")(jnp, shape=(2,), p=0.5).shape == (2,)
    k1 = jrandom.PRNGKey(42)
    k2 = jrandom.PRNGKey(42)
    k3 = jrandom.PRNGKey(99)

    n1 = jrandom.normal(k1, shape=(4,))
    n2 = jrandom.normal(k2, shape=(4,))
    n3 = jrandom.normal(k3, shape=(4,))
    assert_allclose(n1, n2)
    assert n1.tolist() != n3.tolist()

    r1 = jrandom.randint(k1, shape=(6,), minval=0, maxval=100)
    r2 = jrandom.randint(k2, shape=(6,), minval=0, maxval=100)
    r3 = jrandom.randint(k3, shape=(6,), minval=0, maxval=100)
    assert_array_equal(r1, r2)
    assert r1.tolist() != r3.tolist()

    b1 = jrandom.bernoulli(k1, p=0.5, shape=(16,))
    b2 = jrandom.bernoulli(k2, p=0.5, shape=(16,))
    b3 = jrandom.bernoulli(k3, p=0.5, shape=(16,))
    assert_array_equal(b1, b2)
    assert b1.tolist() != b3.tolist()


def test_compiler_proxy_creation_shape_validation() -> None:
    """Verify shape validation in ``_compiler_proxy_creation``."""
    assert _compiler_proxy_creation.zeros(3).shape == (3,)
    assert _compiler_proxy_creation.ones([2, 3]).shape == (2, 3)

    bad_shapes: list[zero_jax._types.ConfigValue] = ["invalid", 3.14]
    for bad_s in bad_shapes:
        with pytest.raises(TypeError, match="Shape must be an int or sequence of ints"):
            _compiler_proxy_creation._validate_shape(bad_s)

    bad_seq_shapes: list[zero_jax._types.ConfigValue] = [["bad"], [1.5]]
    for bad_seq in bad_seq_shapes:
        with pytest.raises(TypeError, match="Shape dimensions must be integers"):
            _compiler_proxy_creation._validate_shape(bad_seq)


def test_from_dlpack() -> None:
    """Execute ``test_from_dlpack``."""
    x = jnp.array([1, 2, 3])
    dl = zero_jax.dlpack.to_dlpack(x)
    out = zero_jax.dlpack.from_dlpack(dl)
    assert_allclose(out, [1, 2, 3])

    dl2 = zero_jax.dlpack.to_dlpack(x)
    out2 = jnp.from_dlpack(dl2)
    assert_allclose(out2, [1, 2, 3])
