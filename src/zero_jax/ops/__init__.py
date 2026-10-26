"""Frontend API routing for jax.ops."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def segment_max(
    data: ArrayLikeSpec,
    segment_ids: ArrayLikeSpec,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: str | None = None,
) -> ArrayLikeSpec:
    """Compute the maximum within segments of an array.

    Args:
        data: Array with the values to be reduced.
        segment_ids: Integer array of segment indices.
        num_segments: Optional total number of segments.
        indices_are_sorted: Whether ``segment_ids`` is known to be sorted.
        unique_indices: Whether ``segment_ids`` has no duplicates.
        bucket_size: Size of bucket for segmented reduction.
        mode: Out-of-bounds indexing mode.

    Returns:
        ArrayLikeSpec: Segment-wise maxima.
    """
    _ = (indices_are_sorted, unique_indices, bucket_size, mode)
    args = (_to_tensor(data), _to_tensor(segment_ids)) + ((num_segments,) if num_segments is not None else ())
    return _wrap(_ops.segment_max(*args))


def segment_min(
    data: ArrayLikeSpec,
    segment_ids: ArrayLikeSpec,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: str | None = None,
) -> ArrayLikeSpec:
    """Compute the minimum within segments of an array.

    Args:
        data: Array with the values to be reduced.
        segment_ids: Integer array of segment indices.
        num_segments: Optional total number of segments.
        indices_are_sorted: Whether ``segment_ids`` is known to be sorted.
        unique_indices: Whether ``segment_ids`` has no duplicates.
        bucket_size: Size of bucket for segmented reduction.
        mode: Out-of-bounds indexing mode.

    Returns:
        ArrayLikeSpec: Segment-wise minima.
    """
    _ = (indices_are_sorted, unique_indices, bucket_size, mode)
    args = (_to_tensor(data), _to_tensor(segment_ids)) + ((num_segments,) if num_segments is not None else ())
    return _wrap(_ops.segment_min(*args))


def segment_prod(
    data: ArrayLikeSpec,
    segment_ids: ArrayLikeSpec,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: str | None = None,
) -> ArrayLikeSpec:
    """Compute the product within segments of an array.

    Args:
        data: Array with the values to be reduced.
        segment_ids: Integer array of segment indices.
        num_segments: Optional total number of segments.
        indices_are_sorted: Whether ``segment_ids`` is known to be sorted.
        unique_indices: Whether ``segment_ids`` has no duplicates.
        bucket_size: Size of bucket for segmented reduction.
        mode: Out-of-bounds indexing mode.

    Returns:
        ArrayLikeSpec: Segment-wise products.
    """
    _ = (indices_are_sorted, unique_indices, bucket_size, mode)
    args = (_to_tensor(data), _to_tensor(segment_ids)) + ((num_segments,) if num_segments is not None else ())
    return _wrap(_ops.segment_prod(*args))


def segment_sum(
    data: ArrayLikeSpec,
    segment_ids: ArrayLikeSpec,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: str | None = None,
) -> ArrayLikeSpec:
    """Compute the sum within segments of an array.

    Args:
        data: Array with the values to be reduced.
        segment_ids: Integer array of segment indices.
        num_segments: Optional total number of segments.
        indices_are_sorted: Whether ``segment_ids`` is known to be sorted.
        unique_indices: Whether ``segment_ids`` has no duplicates.
        bucket_size: Size of bucket for segmented reduction.
        mode: Out-of-bounds indexing mode.

    Returns:
        ArrayLikeSpec: Segment-wise sums.
    """
    _ = (indices_are_sorted, unique_indices, bucket_size, mode)
    args = (_to_tensor(data), _to_tensor(segment_ids)) + ((num_segments,) if num_segments is not None else ())
    return _wrap(_ops.segment_sum(*args))


__all__ = ["segment_max", "segment_min", "segment_prod", "segment_sum"]
