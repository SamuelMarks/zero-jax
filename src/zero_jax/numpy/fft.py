"""Fast Fourier Transform operations for ``zero_jax.numpy.fft``."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, astype


def fft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: int = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the one-dimensional discrete Fourier Transform.

    Args:
        a: Input array, can be complex.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the 1D discrete Fourier Transform.
    """
    return _wrap(ops.fft(_to_tensor(a)))


def rfft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: int = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the one-dimensional discrete Fourier Transform for real input.

    Args:
        a: Real-valued input array.
        n: Number of points along transformation axis in the input to use.
        axis: Axis over which to compute the RFFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the 1D real FFT.
    """
    return _wrap(ops.rfft(_to_tensor(a)))


def fft2(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = (-2, -1), norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the 2-dimensional discrete Fourier Transform.

    Args:
        a: Input array, can be complex.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the 2D FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the 2D FFT.
    """
    return _wrap(ops.fft2(_to_tensor(a), s, axes, norm))


def fftfreq(n: ArrayLikeSpec, d: ArrayLikeSpec = 1.0, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the Discrete Fourier Transform sample frequencies.

    Args:
        n: Window length.
        d: Sample spacing (inverse of the sampling rate).
        dtype: Optional data type of the returned frequency array.

    Returns:
        ArrayLikeSpec: Array of length ``n`` containing the sample frequencies.
    """
    res = _wrap(ops.fftfreq(n, d))
    if dtype is not None:
        return astype(res, dtype)
    return res


def fftn(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = None, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the N-dimensional discrete Fourier Transform.

    Args:
        a: Input array, can be complex.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the N-D FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the N-D FFT.
    """
    return _wrap(ops.fftn(_to_tensor(a), s, axes, norm))


def fftshift(x: ArrayLikeSpec, axes: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Shift the zero-frequency component to the center of the spectrum.

    Args:
        x: Input array.
        axes: Axes over which to shift (default is all axes).

    Returns:
        ArrayLikeSpec: The shifted array.
    """
    return _wrap(ops.fftshift(_to_tensor(x), axes))


def hfft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: ArrayLikeSpec = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum.

    Args:
        a: Input array with Hermitian symmetry.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the Hermitian FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Real ndarray containing the Hermitian FFT.
    """
    return _wrap(ops.hfft(_to_tensor(a), n, axis, norm))


def ifft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: ArrayLikeSpec = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the one-dimensional inverse discrete Fourier Transform.

    Args:
        a: Input complex array.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the inverse DFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the 1D inverse FFT.
    """
    return _wrap(ops.ifft(_to_tensor(a), n, axis, norm))


def ifft2(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = (-2, -1), norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the 2-dimensional inverse discrete Fourier Transform.

    Args:
        a: Input complex array.
        s: Shape of the transformed axes in the output.
        axes: Axes over which to compute the 2D inverse FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the 2D inverse FFT.
    """
    return _wrap(ops.ifft2(_to_tensor(a), s, axes, norm))


def ifftn(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = None, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the N-dimensional inverse discrete Fourier Transform.

    Args:
        a: Input complex array.
        s: Shape of the transformed axes in the output.
        axes: Axes over which to compute the N-D inverse FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the N-D inverse FFT.
    """
    return _wrap(ops.ifftn(_to_tensor(a), s, axes, norm))


def ifftshift(x: ArrayLikeSpec, axes: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the inverse of :func:`fftshift`.

    Args:
        x: Input array.
        axes: Axes over which to calculate the inverse shift.

    Returns:
        ArrayLikeSpec: The shifted array.
    """
    return _wrap(ops.ifftshift(_to_tensor(x), axes))


def ihfft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: ArrayLikeSpec = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the inverse FFT of a signal that has Hermitian symmetry.

    Args:
        a: Input array.
        n: Length of the inverse Hermitian FFT.
        axis: Axis over which to compute the inverse Hermitian FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the inverse Hermitian FFT.
    """
    return _wrap(ops.ihfft(_to_tensor(a), n, axis, norm))


def irfft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: ArrayLikeSpec = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the inverse of :func:`rfft`.

    Args:
        a: Input complex array.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the inverse real FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Real ndarray containing the 1D inverse real FFT.
    """
    return _wrap(ops.irfft(_to_tensor(a), n, axis, norm))


def irfft2(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = (-2, -1), norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the 2-dimensional inverse FFT of a real array.

    Args:
        a: Input complex array.
        s: Shape of the real output to the inverse 2D real FFT.
        axes: Axes over which to compute the inverse 2D real FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Real ndarray containing the 2D inverse real FFT.
    """
    return _wrap(ops.irfft2(_to_tensor(a), s, axes, norm))


def irfftn(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = None, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the N-dimensional inverse FFT of real input.

    Args:
        a: Input complex array.
        s: Shape of the real output to the inverse N-D real FFT.
        axes: Axes over which to compute the inverse N-D real FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Real ndarray containing the N-D inverse real FFT.
    """
    return _wrap(ops.irfftn(_to_tensor(a), s, axes, norm))


def rfft2(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = (-2, -1), norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the 2-dimensional FFT of a real array.

    Args:
        a: Real-valued input array.
        s: Shape of the FFT.
        axes: Axes over which to compute the 2D real FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the 2D real FFT.
    """
    return _wrap(ops.rfft2(_to_tensor(a), s, axes, norm))


def rfftfreq(n: ArrayLikeSpec, d: ArrayLikeSpec = 1.0, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the Discrete Fourier Transform sample frequencies for usage with :func:`rfft`.

    Args:
        n: Window length.
        d: Sample spacing (inverse of the sampling rate).
        dtype: Optional data type of the returned frequency array.

    Returns:
        ArrayLikeSpec: Array of length ``n // 2 + 1`` containing the sample frequencies.
    """
    res = _wrap(ops.rfftfreq(n, d))
    if dtype is not None:
        return astype(res, dtype)
    return res


def rfftn(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = None, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute the N-dimensional discrete Fourier Transform for real input.

    Args:
        a: Real-valued input array.
        s: Shape of the transformed axes in the output.
        axes: Axes over which to compute the N-D real FFT.
        norm: Normalization mode (``"backward"``, ``"ortho"``, or ``"forward"``).

    Returns:
        ArrayLikeSpec: Complex ndarray containing the N-D real FFT.
    """
    return _wrap(ops.rfftn(_to_tensor(a), s, axes, norm))


__all__ = [
    "fft",
    "fft2",
    "fftfreq",
    "fftn",
    "fftshift",
    "hfft",
    "ifft",
    "ifft2",
    "ifftn",
    "ifftshift",
    "ihfft",
    "irfft",
    "irfft2",
    "irfftn",
    "rfft",
    "rfft2",
    "rfftfreq",
    "rfftn",
]
