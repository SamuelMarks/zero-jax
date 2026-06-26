"""Frontend API routing for jax.scipy.signal."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def convolve(*args: Any, **kwargs: Any) -> Any:
    """Convolution of two N-dimensional arrays."""
    return getattr(_ops, "convolve")(*args, **kwargs)


def convolve2d(*args: Any, **kwargs: Any) -> Any:
    """Convolution of two 2-dimensional arrays."""
    return getattr(_ops, "convolve2d")(*args, **kwargs)


def correlate(*args: Any, **kwargs: Any) -> Any:
    """Cross-correlation of two N-dimensional arrays."""
    return getattr(_ops, "correlate")(*args, **kwargs)


def correlate2d(*args: Any, **kwargs: Any) -> Any:
    """Cross-correlation of two 2-dimensional arrays."""
    return getattr(_ops, "correlate2d")(*args, **kwargs)


def csd(*args: Any, **kwargs: Any) -> Any:
    """Estimate cross power spectral density (CSD) using Welch's method."""
    return getattr(_ops, "csd")(*args, **kwargs)


def detrend(*args: Any, **kwargs: Any) -> Any:
    """Remove linear or piecewise linear trends from data."""
    return getattr(_ops, "detrend")(*args, **kwargs)


def fftconvolve(*args: Any, **kwargs: Any) -> Any:
    """Convolve two N-dimensional arrays using Fast Fourier Transform (FFT)."""
    return getattr(_ops, "fftconvolve")(*args, **kwargs)


def istft(*args: Any, **kwargs: Any) -> Any:
    """Perform the inverse short-time Fourier transform (ISTFT)."""
    return getattr(_ops, "istft")(*args, **kwargs)


def stft(*args: Any, **kwargs: Any) -> Any:
    """Compute the short-time Fourier transform (STFT)."""
    return getattr(_ops, "stft")(*args, **kwargs)


def welch(*args: Any, **kwargs: Any) -> Any:
    """Estimate power spectral density (PSD) using Welch's method."""
    return getattr(_ops, "welch")(*args, **kwargs)
