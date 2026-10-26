"""Tests for zero_jax.scipy.fft."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.fft as mod
from tests.conftest import assert_allclose


def test_dct() -> None:
    """Test dct."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    out = mod.dct(x)
    assert out.shape == (4,)
    assert_allclose(out[0], 20.0, atol=1e-4)
    out_ortho = mod.dct(x, norm="ortho")
    assert out_ortho.shape == (4,)


def test_dctn() -> None:
    """Test dctn."""
    x2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = mod.dctn(x2d)
    assert out.shape == (2, 2)
    assert_allclose(out[0, 0], 40.0, atol=1e-4)
    out_axes = mod.dctn(x2d, 2, None, (0, 1))
    assert out_axes.shape == (2, 2)


def test_idct() -> None:
    """Test idct."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    recon = mod.idct(mod.dct(x))
    assert_allclose(recon, x, atol=1e-4)


def test_idctn() -> None:
    """Test idctn."""
    x2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = mod.idctn(mod.dctn(x2d))
    assert_allclose(out, x2d, atol=1e-4)
    out_axes = mod.idctn(x2d, 2, None, (0, 1))
    assert out_axes.shape == (2, 2)
