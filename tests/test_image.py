"""Tests for zero_jax.image."""

from __future__ import annotations

import zero_jax.image as mod
import zero_jax.numpy as jnp
from tests.conftest import assert_allclose


def test_ResizeMethod() -> None:
    """Test ResizeMethod."""
    obj = mod.ResizeMethod(value=mod.ResizeMethod.LINEAR, name="linear")
    assert obj.value == 1
    assert obj.name == "linear"


def test_resize() -> None:
    """Test resize."""
    img = jnp.array([[0.0, 10.0], [20.0, 30.0]])
    res_lin = mod.resize(img, (2, 3), "linear")
    assert res_lin.shape == (2, 3)
    assert_allclose(res_lin[0], [0.0, 5.0, 10.0])
    res_near = mod.resize(img, (3, 2), "nearest")
    assert res_near.shape == (3, 2)
    res_one = mod.resize(jnp.array([5.0]), (3,), "linear")
    assert_allclose(res_one, [5.0, 5.0, 5.0])


def test_scale_and_translate() -> None:
    """Test scale_and_translate."""
    img = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = mod.scale_and_translate(img, (2, 2), (0, 1), jnp.ones(2), jnp.zeros(2), "linear")
    assert_allclose(out, img)
