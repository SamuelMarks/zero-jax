"""Frontend API routing for jax.dlpack."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, bool_, complex64, complex128, float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64

SUPPORTED_DTYPES: frozenset[ConfigValue] = frozenset(
    {
        bool_,
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float16,
        float32,
        float64,
        complex64,
        complex128,
    }
)


def from_dlpack(external_array: ArrayLikeSpec, device: ConfigValue = None, copy: bool | None = None) -> ArrayLikeSpec:
    """Create a JAX array from an object implementing the DLPack protocol.

    Args:
        external_array: Array or capsule object exposing DLPack or array data.
        device: Optional target device.
        copy: Optional copy behavior flag.

    Returns:
        ArrayLikeSpec: Constructed ``ndarray``.
    """
    _ = (device, copy)
    return asarray(external_array)


def to_dlpack(x: ArrayLikeSpec, take_ownership: bool = False, stream: int | None = None) -> ArrayLikeSpec:
    """Export a JAX array as a DLPack-compatible array representation.

    Args:
        x: Input array.
        take_ownership: Unused ownership flag.
        stream: Optional stream handle.

    Returns:
        ArrayLikeSpec: Exported array representation.
    """
    _ = (take_ownership, stream)
    return asarray(x)


__all__ = ["SUPPORTED_DTYPES", "from_dlpack", "to_dlpack"]
