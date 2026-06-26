"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.signal as mod


def test_convolve() -> None:
    """Test convolve."""
    with pytest.raises(NotImplementedError):
        mod.convolve()


def test_convolve2d() -> None:
    """Test convolve2d."""
    with pytest.raises(NotImplementedError):
        mod.convolve2d()


def test_correlate() -> None:
    """Test correlate."""
    with pytest.raises(NotImplementedError):
        mod.correlate()


def test_correlate2d() -> None:
    """Test correlate2d."""
    with pytest.raises(NotImplementedError):
        mod.correlate2d()


def test_csd() -> None:
    """Test csd."""
    with pytest.raises(NotImplementedError):
        mod.csd()


def test_detrend() -> None:
    """Test detrend."""
    with pytest.raises(NotImplementedError):
        mod.detrend()


def test_fftconvolve() -> None:
    """Test fftconvolve."""
    with pytest.raises(NotImplementedError):
        mod.fftconvolve()


def test_istft() -> None:
    """Test istft."""
    with pytest.raises(NotImplementedError):
        mod.istft()


def test_stft() -> None:
    """Test stft."""
    with pytest.raises(NotImplementedError):
        mod.stft()


def test_welch() -> None:
    """Test welch."""
    with pytest.raises(NotImplementedError):
        mod.welch()
