"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.signal as mod


def test_convolve() -> None:
    """Test convolve."""
    with patch("ml_switcheroo_compiler.ops.convolve") as mock_op:
        mod.convolve()
        mock_op.assert_called_once_with()


def test_convolve2d() -> None:
    """Test convolve2d."""
    with patch("ml_switcheroo_compiler.ops.convolve2d") as mock_op:
        mod.convolve2d()
        mock_op.assert_called_once_with()


def test_correlate() -> None:
    """Test correlate."""
    with patch("ml_switcheroo_compiler.ops.correlate") as mock_op:
        mod.correlate()
        mock_op.assert_called_once_with()


def test_correlate2d() -> None:
    """Test correlate2d."""
    with patch("ml_switcheroo_compiler.ops.correlate2d") as mock_op:
        mod.correlate2d()
        mock_op.assert_called_once_with()


def test_csd() -> None:
    """Test csd."""
    with patch("ml_switcheroo_compiler.ops.csd") as mock_op:
        mod.csd()
        mock_op.assert_called_once_with()


def test_detrend() -> None:
    """Test detrend."""
    with patch("ml_switcheroo_compiler.ops.detrend") as mock_op:
        mod.detrend()
        mock_op.assert_called_once_with()


def test_fftconvolve() -> None:
    """Test fftconvolve."""
    with patch("ml_switcheroo_compiler.ops.fftconvolve") as mock_op:
        mod.fftconvolve()
        mock_op.assert_called_once_with()


def test_istft() -> None:
    """Test istft."""
    with patch("ml_switcheroo_compiler.ops.istft") as mock_op:
        mod.istft()
        mock_op.assert_called_once_with()


def test_stft() -> None:
    """Test stft."""
    with patch("ml_switcheroo_compiler.ops.stft") as mock_op:
        mod.stft()
        mock_op.assert_called_once_with()


def test_welch() -> None:
    """Test welch."""
    with patch("ml_switcheroo_compiler.ops.welch") as mock_op:
        mod.welch()
        mock_op.assert_called_once_with()
