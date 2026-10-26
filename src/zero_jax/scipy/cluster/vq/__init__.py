"""Frontend API routing for jax.scipy.cluster.vq."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import argmin, asarray, float32, sqrt
from zero_jax.numpy.lax_numpy import min as _min
from zero_jax.numpy.lax_numpy import sum as _sum


def vq(obs: ArrayLikeSpec, code_book: ArrayLikeSpec, check_finite: bool = True) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Assign observations to codes from a code book and return Euclidean distances.

    Args:
        obs: Observation matrix of shape ``(M, N)``.
        code_book: Code book matrix of shape ``(K, N)``.
        check_finite: Unused finite-value check flag.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: Tuple of ``(code_indices, min_distances)``.
    """
    _ = check_finite
    obs_arr = asarray(obs, dtype=float32)
    cb_arr = asarray(code_book, dtype=float32)
    diff = obs_arr[:, None, :] - cb_arr[None, :, :]
    dists = sqrt(_sum(diff * diff, axis=-1))
    codes = argmin(dists, axis=-1)
    min_dists = _min(dists, axis=-1)
    return codes, min_dists


__all__ = ["vq"]
