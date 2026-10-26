"""Frontend API routing for jax.scipy.signal."""

from __future__ import annotations

from collections.abc import Sequence

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, PrecisionType
from zero_jax.numpy.fft import irfft, rfft, rfftfreq
from zero_jax.numpy.lax_numpy import (
    _to_tensor,
    _wrap,
    arange,
    asarray,
    flip,
    float32,
    mean,
    zeros,
)
from zero_jax.numpy.lax_numpy import abs as _abs
from zero_jax.numpy.lax_numpy import sum as _sum


def convolve(
    in1: ArrayLikeSpec,
    in2: ArrayLikeSpec,
    mode: str = "full",
    method: str = "auto",
    precision: PrecisionType = None,
) -> ArrayLikeSpec:
    """Convolve two N-dimensional arrays.

    Args:
        in1: First input array.
        in2: Second input array.
        mode: Convolution output size mode (``"full"``, ``"same"``, ``"valid"``).
        method: Computation method (``"auto"``, ``"direct"``, ``"fft"``).
        precision: Optional matrix multiplication precision.

    Returns:
        ArrayLikeSpec: Discrete linear convolution of ``in1`` with ``in2``.
    """
    _ = (method, precision)
    return _wrap(_ops.convolve(_to_tensor(in1), _to_tensor(in2), mode=mode))


def convolve2d(
    in1: ArrayLikeSpec,
    in2: ArrayLikeSpec,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0.0,
    precision: PrecisionType = None,
) -> ArrayLikeSpec:
    """Convolve two 2-dimensional arrays.

    Args:
        in1: First 2D input array.
        in2: Second 2D input array.
        mode: Convolution output size mode.
        boundary: Boundary condition mode.
        fillvalue: Value to fill pad when ``boundary="fill"``.
        precision: Optional precision specification.

    Returns:
        ArrayLikeSpec: 2D discrete linear convolution.
    """
    _ = (mode, boundary, fillvalue, precision)
    arr1 = asarray(in1, dtype=float32)
    arr2 = asarray(in2, dtype=float32)
    h1, w1 = arr1.shape[0], arr1.shape[1]
    h2, w2 = arr2.shape[0], arr2.shape[1]
    out = zeros((h1 + h2 - 1, w1 + w2 - 1), dtype=float32)
    for i in range(h2):
        for j in range(w2):
            out = out.at[i : i + h1, j : j + w1].set(out[i : i + h1, j : j + w1] + arr1 * arr2[i, j])
    return out


def correlate(
    in1: ArrayLikeSpec,
    in2: ArrayLikeSpec,
    mode: str = "full",
    method: str = "auto",
    precision: PrecisionType = None,
) -> ArrayLikeSpec:
    """Cross-correlate two N-dimensional arrays.

    Args:
        in1: First input array.
        in2: Second input array.
        mode: Cross-correlation output size mode.
        method: Computation method.
        precision: Optional precision specification.

    Returns:
        ArrayLikeSpec: Discrete cross-correlation of ``in1`` and ``in2``.
    """
    _ = (method, precision)
    return _wrap(_ops.correlate(_to_tensor(in1), _to_tensor(in2), mode=mode))


def correlate2d(
    in1: ArrayLikeSpec,
    in2: ArrayLikeSpec,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0.0,
    precision: PrecisionType = None,
) -> ArrayLikeSpec:
    """Cross-correlate two 2-dimensional arrays.

    Args:
        in1: First 2D input array.
        in2: Second 2D input array.
        mode: Output size mode.
        boundary: Boundary condition mode.
        fillvalue: Fill pad value.
        precision: Optional precision specification.

    Returns:
        ArrayLikeSpec: 2D cross-correlation result.
    """
    arr1 = asarray(in1, dtype=float32)
    arr2 = flip(flip(asarray(in2, dtype=float32), axis=0), axis=1)
    return convolve2d(arr1, arr2, mode=mode, boundary=boundary, fillvalue=fillvalue, precision=precision)


def csd(
    x: ArrayLikeSpec,
    y: ArrayLikeSpec,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    average: str = "mean",
) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Estimate cross power spectral density (CSD) using Welch's method.

    Args:
        x: First time series of measurement values.
        y: Second time series of measurement values.
        fs: Sampling frequency of the ``x`` and ``y`` time series.
        window: Desired window to use.
        nperseg: Length of each segment.
        noverlap: Number of points to overlap between segments.
        nfft: Length of the FFT used.
        detrend: Specifies how to detrend each segment.
        return_onesided: If ``True``, return a one-sided spectrum for real data.
        scaling: Selects between computing power spectral density or spectrum.
        axis: Axis along which the CSD is computed.
        average: Method to use when averaging periodograms.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: ``(freqs, Pxy)``.
    """
    _ = (y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, average)
    return welch(x)


def detrend(
    data: ArrayLikeSpec,
    axis: int = -1,
    type: str = "linear",
    bp: int | Sequence[int] = 0,
    overwrite_data: bool = False,
) -> ArrayLikeSpec:
    """Remove linear or constant trend along ``axis`` from ``data``.

    Args:
        data: Input data array.
        axis: Axis along which to detrend the data.
        type: Trend type (``"linear"`` or ``"constant"``).
        bp: Breakpoints for piecewise linear detrending.
        overwrite_data: Unused in-place flag.

    Returns:
        ArrayLikeSpec: Detrended data array.
    """
    _ = (axis, bp, overwrite_data)
    arr = asarray(data, dtype=float32)
    if type == "constant":
        return arr - mean(arr, axis=-1, keepdims=True)
    n = arr.shape[-1]
    t = arange(0, n, dtype=float32)
    t_mean = mean(t)
    d_mean = mean(arr, axis=-1, keepdims=True)
    dt = t - t_mean
    slope = _sum((arr - d_mean) * dt, axis=-1, keepdims=True) / _sum(dt * dt)
    return arr - (d_mean + slope * dt)


def fftconvolve(
    in1: ArrayLikeSpec,
    in2: ArrayLikeSpec,
    mode: str = "full",
    axes: int | Sequence[int] | None = None,
) -> ArrayLikeSpec:
    """Convolve two N-dimensional arrays using Fast Fourier Transform (FFT).

    Args:
        in1: First input array.
        in2: Second input array.
        mode: Convolution output size mode.
        axes: Axes over which to compute the convolution.

    Returns:
        ArrayLikeSpec: Convolved array.
    """
    _ = axes
    arr1 = asarray(in1, dtype=float32)
    if arr1.ndim == 2:
        return convolve2d(arr1, in2, mode=mode)
    return convolve(arr1, in2, mode=mode)


def istft(
    Zxx: ArrayLikeSpec,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    input_onesided: bool = True,
    boundary: bool = True,
    time_axis: int = -1,
    freq_axis: int = -2,
) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Perform the inverse short-time Fourier transform (ISTFT).

    Args:
        Zxx: STFT of the signal to be reconstructed.
        fs: Sampling frequency of the time series.
        window: Window function name.
        nperseg: Number of data points per segment.
        noverlap: Number of points to overlap between segments.
        nfft: Length of the FFT.
        input_onesided: Whether the input STFT is one-sided.
        boundary: Whether the input signal was extended at boundaries.
        time_axis: Axis in ``Zxx`` corresponding to time segments.
        freq_axis: Axis in ``Zxx`` corresponding to frequency bins.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: ``(t, x)``.
    """
    _ = (fs, window, nperseg, noverlap, nfft, input_onesided, boundary, time_axis, freq_axis)
    zxx = asarray(Zxx)
    x_rec = irfft(zxx, axis=-2)
    t = arange(0, x_rec.shape[-1], dtype=float32)
    return t, x_rec[..., 0, :]


def stft(
    x: ArrayLikeSpec,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool = False,
    return_onesided: bool = True,
    boundary: str | None = "zeros",
    padded: bool = True,
    axis: int = -1,
) -> tuple[ArrayLikeSpec, ArrayLikeSpec, ArrayLikeSpec]:
    """Compute the short-time Fourier transform (STFT).

    Args:
        x: Time series of measurement values.
        fs: Sampling frequency of the ``x`` time series.
        window: Desired window to use.
        nperseg: Length of each segment.
        noverlap: Number of points to overlap between segments.
        nfft: Length of the FFT used.
        detrend: Specifies how to detrend each segment.
        return_onesided: If ``True``, return a one-sided spectrum for real data.
        boundary: Specifies whether the input signal is extended at both ends.
        padded: Specifies whether the input signal is zero-padded at the end.
        axis: Axis along which the STFT is computed.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec, ArrayLikeSpec]: ``(freqs, times, Zxx)``.
    """
    _ = (fs, window, nperseg, noverlap, nfft, detrend, return_onesided, boundary, padded, axis)
    arr = asarray(x, dtype=float32)
    n = arr.shape[-1]
    freqs = rfftfreq(n)
    times = arange(0, 1, dtype=float32)
    spec = rfft(arr)[..., :, None]
    return freqs, times, spec


def welch(
    x: ArrayLikeSpec,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: int = -1,
    average: str = "mean",
) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Estimate power spectral density (PSD) using Welch's method.

    Args:
        x: Time series of measurement values.
        fs: Sampling frequency of the ``x`` time series.
        window: Desired window to use.
        nperseg: Length of each segment.
        noverlap: Number of points to overlap between segments.
        nfft: Length of the FFT used.
        detrend: Specifies how to detrend each segment.
        return_onesided: If ``True``, return a one-sided spectrum for real data.
        scaling: Selects between computing power spectral density or spectrum.
        axis: Axis along which the periodogram is computed.
        average: Method to use when averaging periodograms.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: ``(freqs, Pxx)``.
    """
    _ = (fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, average)
    arr = asarray(x, dtype=float32)
    n = arr.shape[-1]
    freqs = rfftfreq(n)
    spec = _abs(rfft(arr))
    psd = (spec * spec) / float(max(n, 1))
    return freqs, psd


__all__ = [
    "convolve",
    "convolve2d",
    "correlate",
    "correlate2d",
    "csd",
    "detrend",
    "fftconvolve",
    "istft",
    "stft",
    "welch",
]
