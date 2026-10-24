"""Module documentation."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def fft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: int = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for fft.

    Args:
        a: Argument a.
        n: Argument n.
        axis: Argument axis.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.fft(_to_tensor(a)))


def rfft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: int = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for rfft.

    Args:
        a: Argument a.
        n: Argument n.
        axis: Argument axis.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.rfft(_to_tensor(a)))


def fft2(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = (-2, -1), norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for fft2.

    Args:
        a: Argument a.
        s: Argument s.
        axes: Argument axes.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.fft2(_to_tensor(a), s, axes, norm))


def fftfreq(n: ArrayLikeSpec, d: ArrayLikeSpec = 1.0, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for fftfreq.

    Args:
        n: Argument n.
        d: Argument d.
        dtype: Argument dtype.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.fftfreq(n, d))


def fftn(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = None, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for fftn.

    Args:
        a: Argument a.
        s: Argument s.
        axes: Argument axes.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.fftn(_to_tensor(a), s, axes, norm))


def fftshift(x: ArrayLikeSpec, axes: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for fftshift.

    Args:
        x: Argument x.
        axes: Argument axes.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.fftshift(_to_tensor(x), axes))


def hfft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: ArrayLikeSpec = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for hfft.

    Args:
        a: Argument a.
        n: Argument n.
        axis: Argument axis.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.hfft(_to_tensor(a), n, axis, norm))


def ifft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: ArrayLikeSpec = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for ifft.

    Args:
        a: Argument a.
        n: Argument n.
        axis: Argument axis.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.ifft(_to_tensor(a), n, axis, norm))


def ifft2(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = (-2, -1), norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for ifft2.

    Args:
        a: Argument a.
        s: Argument s.
        axes: Argument axes.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.ifft2(_to_tensor(a), s, axes, norm))


def ifftn(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = None, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for ifftn.

    Args:
        a: Argument a.
        s: Argument s.
        axes: Argument axes.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.ifftn(_to_tensor(a), s, axes, norm))


def ifftshift(x: ArrayLikeSpec, axes: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for ifftshift.

    Args:
        x: Argument x.
        axes: Argument axes.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.ifftshift(_to_tensor(x), axes))


def ihfft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: ArrayLikeSpec = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for ihfft.

    Args:
        a: Argument a.
        n: Argument n.
        axis: Argument axis.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.ihfft(_to_tensor(a), n, axis, norm))


def irfft(a: ArrayLikeSpec, n: ArrayLikeSpec = None, axis: ArrayLikeSpec = -1, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for irfft.

    Args:
        a: Argument a.
        n: Argument n.
        axis: Argument axis.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.irfft(_to_tensor(a), n, axis, norm))


def irfft2(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = (-2, -1), norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for irfft2.

    Args:
        a: Argument a.
        s: Argument s.
        axes: Argument axes.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.irfft2(_to_tensor(a), s, axes, norm))


def irfftn(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = None, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for irfftn.

    Args:
        a: Argument a.
        s: Argument s.
        axes: Argument axes.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.irfftn(_to_tensor(a), s, axes, norm))


def rfft2(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = (-2, -1), norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for rfft2.

    Args:
        a: Argument a.
        s: Argument s.
        axes: Argument axes.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.rfft2(_to_tensor(a), s, axes, norm))


def rfftfreq(n: ArrayLikeSpec, d: ArrayLikeSpec = 1.0, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for rfftfreq.

    Args:
        n: Argument n.
        d: Argument d.
        dtype: Argument dtype.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
    return _wrap(ops.rfftfreq(n, d))


def rfftn(a: ArrayLikeSpec, s: ArrayLikeSpec = None, axes: ArrayLikeSpec = None, norm: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for rfftn.

    Args:
        a: Argument a.
        s: Argument s.
        axes: Argument axes.
        norm: Argument norm.

    Returns:
        ArrayLikeSpec: The result.
    """
    # Note: we drop kwargs that ml-switcheroo-compiler might not support yet
    # by catching TypeError and retrying, or just passing them if possible.
    # But for a robust API, we just pass what we can.
    # Actually, JAX's compiler_ops might not have all the args JAX has.
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
