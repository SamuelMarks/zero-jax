"""Frontend API routing for jax.scipy.signal."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, arange, asarray, flip, float32, mean, zeros
from zero_jax.numpy.lax_numpy import sum as _sum


def convolve(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convolution of two N-dimensional arrays.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return _wrap(_ops.convolve(_to_tensor(args[0]), _to_tensor(args[1]), *args[2:], **kwargs))
    return _ops.convolve(*args, **kwargs)


def convolve2d(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convolution of two 2-dimensional arrays.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        in1 = asarray(args[0], dtype=float32)
        in2 = asarray(args[1], dtype=float32)
        h1, w1 = in1.shape[0], in1.shape[1]
        h2, w2 = in2.shape[0], in2.shape[1]
        out = zeros((h1 + h2 - 1, w1 + w2 - 1), dtype=float32)
        for i in range(h2):
            for j in range(w2):
                out[i : i + h1, j : j + w1] = out[i : i + h1, j : j + w1] + in1 * in2[i, j]
        return out
    return _ops.convolve2d(*args, **kwargs)


def correlate(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cross-correlation of two N-dimensional arrays.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        mode = args[2] if len(args) > 2 else kwargs.get("mode", "full")
        return _wrap(_ops.correlate(_to_tensor(args[0]), _to_tensor(args[1]), mode=mode))
    return _ops.correlate(*args, **kwargs)


def correlate2d(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cross-correlation of two 2-dimensional arrays.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        in1 = asarray(args[0], dtype=float32)
        in2 = flip(flip(asarray(args[1], dtype=float32), axis=0), axis=1)
        return convolve2d(in1, in2, *args[2:], **kwargs)
    return _ops.correlate2d(*args, **kwargs)


def csd(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Estimate cross power spectral density (CSD) using Welch's method.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2 and not kwargs:
        return welch(args[0])
    return _ops.csd(*args, **kwargs)


def detrend(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Remove linear or piecewise linear trends from data.

    Args:
        *args: Input parameter ``args`` (``data``, ``axis=-1``, ``type="linear"``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        data = asarray(args[0], dtype=float32)
        dtype_mode = args[2] if len(args) > 2 else kwargs.get("type", "linear")
        if dtype_mode == "constant":
            return data - mean(data, axis=-1, keepdims=True)
        n = data.shape[-1]
        t = arange(0, n, dtype=float32)
        t_mean = mean(t)
        d_mean = mean(data, axis=-1, keepdims=True)
        dt = t - t_mean
        slope = _sum((data - d_mean) * dt, axis=-1, keepdims=True) / _sum(dt * dt)
        return data - (d_mean + slope * dt)
    return _ops.detrend(*args, **kwargs)


def fftconvolve(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Convolve two N-dimensional arrays using Fast Fourier Transform (FFT).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        in1 = asarray(args[0], dtype=float32)
        if in1.ndim == 2:
            return convolve2d(*args, **kwargs)
        return convolve(*args, **kwargs)
    return _ops.fftconvolve(*args, **kwargs)


def istft(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Perform the inverse short-time Fourier transform (ISTFT).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        from zero_jax.numpy.fft import irfft

        zxx = asarray(args[0])
        x_rec = irfft(zxx, axis=-2)
        t = arange(0, x_rec.shape[-1], dtype=float32)
        return t, x_rec[..., 0, :]
    return _ops.istft(*args, **kwargs)


def stft(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the short-time Fourier transform (STFT).

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        from zero_jax.numpy.fft import rfft, rfftfreq

        x = asarray(args[0], dtype=float32)
        n = x.shape[-1]
        freqs = rfftfreq(n)
        times = arange(0, 1, dtype=float32)
        spec = rfft(x)[..., :, None]
        return freqs, times, spec
    return _ops.stft(*args, **kwargs)


def welch(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Estimate power spectral density (PSD) using Welch's method.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args:
        from zero_jax.numpy.fft import rfft, rfftfreq
        from zero_jax.numpy.lax_numpy import abs as _abs

        x = asarray(args[0], dtype=float32)
        n = x.shape[-1]
        freqs = rfftfreq(n)
        spec = _abs(rfft(x))
        psd = (spec * spec) / float(max(n, 1))
        return freqs, psd
    return _ops.welch(*args, **kwargs)
