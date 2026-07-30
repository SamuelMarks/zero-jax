"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.signal as mod


def test_convolve() -> None:
    """Test convolve."""
    with patch("zero_jax._compiler_proxy_ops.convolve", create=True) as mock_op:
        mod.convolve()
        mock_op.assert_called_once_with()


def test_convolve2d() -> None:
    """Test convolve2d."""
    with patch("zero_jax._compiler_proxy_ops.convolve2d", create=True) as mock_op:
        mod.convolve2d()
        mock_op.assert_called_once_with()


def test_correlate() -> None:
    """Test correlate."""
    with patch("zero_jax._compiler_proxy_ops.correlate", create=True) as mock_op:
        mod.correlate()
        mock_op.assert_called_once_with()


def test_correlate2d() -> None:
    """Test correlate2d."""
    with patch("zero_jax._compiler_proxy_ops.correlate2d", create=True) as mock_op:
        mod.correlate2d()
        mock_op.assert_called_once_with()


def test_csd() -> None:
    """Test csd."""
    with patch("zero_jax._compiler_proxy_ops.csd", create=True) as mock_op:
        mod.csd()
        mock_op.assert_called_once_with()


def test_detrend() -> None:
    """Test detrend."""
    with patch("zero_jax._compiler_proxy_ops.detrend", create=True) as mock_op:
        mod.detrend()
        mock_op.assert_called_once_with()


def test_fftconvolve() -> None:
    """Test fftconvolve."""
    with patch("zero_jax._compiler_proxy_ops.fftconvolve", create=True) as mock_op:
        mod.fftconvolve()
        mock_op.assert_called_once_with()


def test_istft() -> None:
    """Test istft."""
    with patch("zero_jax._compiler_proxy_ops.istft", create=True) as mock_op:
        mod.istft()
        mock_op.assert_called_once_with()


def test_stft() -> None:
    """Test stft."""
    with patch("zero_jax._compiler_proxy_ops.stft", create=True) as mock_op:
        mod.stft()
        mock_op.assert_called_once_with()


def test_welch() -> None:
    """Test welch."""
    with patch("zero_jax._compiler_proxy_ops.welch", create=True) as mock_op:
        mod.welch()
        mock_op.assert_called_once_with()
