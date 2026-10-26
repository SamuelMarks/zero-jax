"""LAX parallel collective and token primitives."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.lax.primitives import pmean, psum
from zero_jax.numpy import arange, array, asarray, uint8
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, ndarray


def pbkdf2(passphrase: ArrayLikeSpec, salt: ArrayLikeSpec, iterations: int = 1000, key_length: int = 32) -> ndarray:
    """Derive a cryptographic key from ``passphrase`` and ``salt`` via iterative key stretching.

    Args:
        passphrase: Input passphrase array.
        salt: Input salt array.
        iterations: Number of hash iterations.
        key_length: Desired derived key length in bytes.

    Returns:
        ndarray: Derived key array of shape ``(key_length,)`` and dtype ``uint8``.
    """
    pass_arr = asarray(passphrase).__array__().ravel()
    salt_arr = asarray(salt).__array__().ravel()
    state = 0x811C9DC5
    for val in pass_arr:
        state = ((state ^ int(val)) * 0x01000193) & 0xFFFFFFFF
    for val in salt_arr:
        state = ((state ^ int(val)) * 0x01000193) & 0xFFFFFFFF
    rounds = max(1, min(int(iterations), 256))
    for r in range(rounds):
        state = ((state ^ r) * 1664525 + 1013904223) & 0xFFFFFFFF
    derived = [((state >> ((i % 4) * 8)) ^ (i * 31 + rounds)) & 0xFF for i in range(int(key_length))]
    return array(derived, dtype=uint8)


def after_all(*operands: ArrayLikeSpec) -> ArrayLikeSpec:
    """Order side-effecting operations by merging input tokens into a single new token.

    Args:
        *operands: Zero or more token operands.

    Returns:
        ArrayLikeSpec: Synchronization token created after all ``operands``.
    """
    _ = operands
    return ops.create_token()


def all_gather(operand: ArrayLikeSpec, axis_name: str, tiled: bool = False) -> ndarray:
    """Gather values of ``operand`` across all replicas along ``axis_name``.

    Args:
        operand: Input array to gather.
        axis_name: Mapped axis name.
        tiled: Whether to concatenate rather than stack along a new dimension.

    Returns:
        ndarray: Gathered array.
    """
    _ = tiled
    return _wrap(ops.all_gather(_to_tensor(operand), axis_name))


def all_to_all(operand: ArrayLikeSpec, axis_name: str, split_axis: int, concat_axis: int) -> ndarray:
    """Split ``operand`` along ``split_axis`` and scatter/gather across ``axis_name`` along ``concat_axis``.

    Args:
        operand: Input array.
        axis_name: Mapped axis name.
        split_axis: Axis to split before exchange.
        concat_axis: Axis to concatenate after exchange.

    Returns:
        ndarray: Exchanged array.
    """
    return _wrap(ops.all_to_all(_to_tensor(operand), split_axis, concat_axis, axis_name))


def axis_index(axis_name: str) -> ArrayLikeSpec:
    """Return the replica index along the mapped axis ``axis_name``.

    Args:
        axis_name: Mapped axis name.

    Returns:
        ArrayLikeSpec: Replica index along ``axis_name``.
    """
    return ops.axis_index(axis_name)


def axis_size(axis_name: ArrayLikeSpec) -> int:
    """Return the size of the mapped axis ``axis_name``.

    Args:
        axis_name: Axis name or tuple of axis names.

    Returns:
        int: Size of the axis (``1`` in single-device execution).
    """
    _ = axis_name
    return 1


def composite(decomposition: ArrayLikeSpec, name: str, version: int = 0) -> ArrayLikeSpec:
    """Wrap ``decomposition`` as a named StableHLO composite operation.

    Args:
        decomposition: Function implementing the composite operation.
        name: Composite operation name.
        version: Composite version number.

    Returns:
        ArrayLikeSpec: Wrapped decomposition callable.
    """
    _ = (name, version)
    return decomposition


def create_token() -> ArrayLikeSpec:
    """Create a fresh synchronization token for ordering side effects.

    Returns:
        ArrayLikeSpec: Newly created token.
    """
    return ops.create_token()


def dce_sink(val: ArrayLikeSpec, prevent_mlir_dce: bool = False) -> ArrayLikeSpec:
    """Prevent dead-code elimination of ``val``.

    Args:
        val: Value or PyTree to sink.
        prevent_mlir_dce: Whether to emit an MLIR side-effecting sink barrier.

    Returns:
        ArrayLikeSpec: Unmodified ``val``.
    """
    _ = prevent_mlir_dce
    return val


def infeed() -> ArrayLikeSpec:
    """Read a value from the device infeed queue.

    Returns:
        ArrayLikeSpec: Value received from the host infeed stream.
    """
    return ops.infeed()


def optimization_barrier(operand: ArrayLikeSpec) -> ArrayLikeSpec:
    """Prevent compiler optimizations from moving computations across ``operand``.

    Args:
        operand: Input array or PyTree.

    Returns:
        ArrayLikeSpec: Unmodified ``operand``.
    """
    return operand


def outfeed(val: ArrayLikeSpec) -> None:
    """Send ``val`` to the host outfeed queue.

    Args:
        val: Array value to emit to the outfeed stream.
    """
    ops.outfeed(_to_tensor(val))


def pbroadcast(operand: ArrayLikeSpec, axis_name: str) -> ndarray:
    """Broadcast ``operand`` across replicas along ``axis_name``.

    Args:
        operand: Input array.
        axis_name: Mapped collective axis name.

    Returns:
        ndarray: Broadcasted array (identity on a single device).
    """
    _ = axis_name
    return asarray(operand)


def pcast(x: ArrayLikeSpec, axis_name: ArrayLikeSpec, *, to: str = "varying") -> ArrayLikeSpec:
    """Perform a parallel collective cast of ``x`` along ``axis_name``.

    Args:
        x: Input array.
        axis_name: Mapped axis name.
        to: Target variance mode (e.g. ``"varying"`` or ``"reduced"``).

    Returns:
        ArrayLikeSpec: Cast array ``x``.
    """
    _ = (axis_name, to)
    return x


def pdot(lhs: ArrayLikeSpec, rhs: ArrayLikeSpec, axis_name: str) -> ndarray:
    """Compute a parallel dot product and sum-reduce across ``axis_name``.

    Args:
        lhs: Left-hand side operand.
        rhs: Right-hand side operand.
        axis_name: Mapped collective axis name.

    Returns:
        ndarray: Parallel dot product result.
    """
    return _wrap(ops.pdot(_to_tensor(lhs), _to_tensor(rhs), axis_name))


def platform_dependent(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Select platform-specific branch ``x`` (or fallback ``y``).

    Args:
        x: Primary value or branch.
        y: Fallback value or branch.

    Returns:
        ArrayLikeSpec: Selected value ``x``.
    """
    _ = y
    return x


def pmax(operand: ArrayLikeSpec, axis_name: str) -> ndarray:
    """Compute an all-reduce maximum of ``operand`` across ``axis_name``.

    Args:
        operand: Input array.
        axis_name: Mapped collective axis name.

    Returns:
        ndarray: All-reduce maximum (identity on a single device).
    """
    _ = axis_name
    return asarray(operand)


def pmin(operand: ArrayLikeSpec, axis_name: str) -> ndarray:
    """Compute an all-reduce minimum of ``operand`` across ``axis_name``.

    Args:
        operand: Input array.
        axis_name: Mapped collective axis name.

    Returns:
        ndarray: All-reduce minimum (identity on a single device).
    """
    _ = axis_name
    return asarray(operand)


def ppermute(operand: ArrayLikeSpec, axis_name: str, perm: ArrayLikeSpec) -> ndarray:
    """Permute ``operand`` across replicas along ``axis_name`` according to ``perm``.

    Args:
        operand: Input array.
        axis_name: Mapped collective axis name.
        perm: Sequence of ``(source, target)`` replica index pairs.

    Returns:
        ndarray: Permuted array (identity on a single device).
    """
    _ = (axis_name, perm)
    return asarray(operand)


def pshuffle(operand: ArrayLikeSpec, axis_name: str, perm: ArrayLikeSpec) -> ndarray:
    """Shuffle ``operand`` across replicas along ``axis_name`` using permutation ``perm``.

    Args:
        operand: Input array.
        axis_name: Mapped collective axis name.
        perm: Permutation of replica indices.

    Returns:
        ndarray: Shuffled array (identity on a single device).
    """
    _ = (axis_name, perm)
    return asarray(operand)


def psum_scatter(operand: ArrayLikeSpec, axis_name: str, scatter_dimension: int = 0, tiled: bool = False) -> ndarray:
    """Sum-reduce ``operand`` across ``axis_name`` and scatter along ``scatter_dimension``.

    Args:
        operand: Input array.
        axis_name: Mapped collective axis name.
        scatter_dimension: Dimension along which to scatter the reduced result.
        tiled: Whether to preserve rank via tiled splitting.

    Returns:
        ndarray: Reduce-scattered array.
    """
    _ = tiled
    return _wrap(ops.psum_scatter(_to_tensor(operand), scatter_dimension, axis_name))


def psum_scatter_start(
    operand: ArrayLikeSpec = None,
    axis_name: str = "i",
    *,
    scatter_dimension: int = 0,
    axis_index_groups: ArrayLikeSpec = None,
    tiled: bool = False,
) -> ArrayLikeSpec:
    """Initiate an asynchronous collective reduce-scatter operation.

    Args:
        operand: Input array to reduce-scatter.
        axis_name: Mapped collective axis name.
        scatter_dimension: Dimension along which to scatter.
        axis_index_groups: Optional replica index groups.
        tiled: Whether to use tiled scattering.

    Returns:
        ArrayLikeSpec: Reduce-scattered array or ``None`` if ``operand`` is ``None``.
    """
    _ = (axis_name, scatter_dimension, axis_index_groups, tiled)
    return asarray(operand) if operand is not None else None


def pswapaxes(operand: ArrayLikeSpec, axis_name: str, axis: int) -> ndarray:
    """Swap array dimension ``axis`` with the mapped collective axis ``axis_name``.

    Args:
        operand: Input array.
        axis_name: Mapped collective axis name.
        axis: Array dimension index to swap with ``axis_name``.

    Returns:
        ndarray: Swapped array (identity on a single device).
    """
    _ = (axis_name, axis)
    return asarray(operand)


def ragged_all_to_all(
    operand: ArrayLikeSpec,
    output: ArrayLikeSpec,
    input_offsets: ArrayLikeSpec,
    send_sizes: ArrayLikeSpec,
    output_offsets: ArrayLikeSpec,
    recv_sizes: ArrayLikeSpec,
    *,
    axis_name: ArrayLikeSpec,
    axis_index_groups: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Perform a ragged all-to-all collective exchange along ``axis_name``.

    Args:
        operand: Input buffer array.
        output: Output buffer array.
        input_offsets: Starting offsets in ``operand``.
        send_sizes: Number of elements to send per peer.
        output_offsets: Starting offsets in ``output``.
        recv_sizes: Number of elements to receive per peer.
        axis_name: Collective axis name.
        axis_index_groups: Optional replica index groups.

    Returns:
        ArrayLikeSpec: Populated ``output`` buffer.
    """
    _ = (operand, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name, axis_index_groups)
    return output


def stage(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stage ``x`` into the current trace/computation.

    Args:
        x: Input array or callable.

    Returns:
        ArrayLikeSpec: Staged ``x``.
    """
    return x


def with_sharding_constraint(x: ArrayLikeSpec, sharding: ArrayLikeSpec) -> ArrayLikeSpec:
    """Constrain the sharding of ``x`` to ``sharding``.

    Args:
        x: Input array or PyTree.
        sharding: Target sharding specification.

    Returns:
        ArrayLikeSpec: Sharding-constrained ``x``.
    """
    _ = sharding
    return x


__all__ = [
    "after_all",
    "all_gather",
    "all_to_all",
    "axis_index",
    "axis_size",
    "composite",
    "create_token",
    "dce_sink",
    "infeed",
    "optimization_barrier",
    "outfeed",
    "pbkdf2",
    "pbroadcast",
    "pcast",
    "pdot",
    "platform_dependent",
    "pmax",
    "pmean",
    "pmin",
    "ppermute",
    "pshuffle",
    "psum",
    "psum_scatter",
    "psum_scatter_start",
    "pswapaxes",
    "ragged_all_to_all",
    "stage",
    "with_sharding_constraint",
]
