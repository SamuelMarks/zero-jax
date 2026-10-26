"""Tests for zero_jax.scipy.signal."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.signal as mod
from tests.conftest import assert_allclose


def test_convolve() -> None:
    """Test convolve."""
    assert_allclose(mod.convolve(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])), [3.0, 10.0, 8.0])


def test_convolve2d() -> None:
    """Test convolve2d."""
    a = jnp.ones((2, 2))
    out = mod.convolve2d(a, a)
    assert out.shape == (3, 3)
    assert_allclose(out[1, 1], 4.0)


def test_correlate() -> None:
    """Test correlate."""
    assert_allclose(mod.correlate(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])), [4.0, 11.0, 6.0])


def test_correlate2d() -> None:
    """Test correlate2d."""
    a = jnp.ones((2, 2))
    out = mod.correlate2d(a, a)
    assert out.shape == (3, 3)
    assert_allclose(out[1, 1], 4.0)


def test_csd() -> None:
    """Test csd."""
    freqs, pxy = mod.csd(jnp.ones((8,)), jnp.ones((8,)))
    assert freqs.shape == (5,)
    assert pxy.shape == (5,)


def test_detrend() -> None:
    """Test detrend."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert_allclose(mod.detrend(x, type="linear"), [0.0, 0.0, 0.0, 0.0], atol=1e-5)
    assert_allclose(mod.detrend(jnp.array([2.0, 2.0, 2.0]), type="constant"), [0.0, 0.0, 0.0], atol=1e-5)


def test_fftconvolve() -> None:
    """Test fftconvolve."""
    assert_allclose(mod.fftconvolve(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])), [3.0, 10.0, 8.0])
    assert mod.fftconvolve(jnp.ones((2, 2)), jnp.ones((2, 2))).shape == (3, 3)


def test_istft() -> None:
    """Test istft."""
    _, _, zxx = mod.stft(jnp.ones((8,)), nperseg=4)
    t, x_rec = mod.istft(zxx, nperseg=4)
    assert t.shape == x_rec.shape


def test_stft() -> None:
    """Test stft."""
    freqs, times, zxx = mod.stft(jnp.ones((8,)), nperseg=4)
    assert freqs.shape == (5,)
    assert times.shape == (1,)
    assert zxx.shape == (5, 1)


def test_welch() -> None:
    """Test welch."""
    freqs, pxx = mod.welch(jnp.ones((8,)))
    assert freqs.shape == (5,)
    assert pxx.shape == (5,)
    assert_allclose(pxx[0], 8.0, atol=1e-5)
