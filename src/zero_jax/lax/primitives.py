"""Module docstring."""

from typing import Any
import ml_switcheroo.ops as ops
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def add(x: Any, y: Any) -> Any:
    """Add function."""
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


def sub(x: Any, y: Any) -> Any:
    """Sub function."""
    return _wrap(ops.subtract(_to_tensor(x), _to_tensor(y)))


def mul(x: Any, y: Any) -> Any:
    """Mul function."""
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


def div(x: Any, y: Any) -> Any:
    """Div function."""
    return _wrap(ops.divide(_to_tensor(x), _to_tensor(y)))


def broadcast(x: Any, sizes: Any) -> Any:
    """Broadcast function."""
    sizes_tuple = tuple(sizes) + _to_tensor(x).shape
    return _wrap(ops.broadcast_to(_to_tensor(x), sizes_tuple))


def broadcast_in_dim(x: Any, shape: Any, broadcast_dimensions: Any) -> Any:
    """broadcast_in_dim function."""
    # First reshape x to insert 1s for non-broadcasted dimensions
    t = _to_tensor(x)
    new_shape = [1] * len(shape)
    for d, s in zip(broadcast_dimensions, t.shape):
        new_shape[d] = s
    reshaped = ops.reshape(t, tuple(new_shape))
    return _wrap(ops.broadcast_to(reshaped, shape))


def reshape(x: Any, new_sizes: Any, dimensions: Any = None) -> Any:
    """Reshape function."""
    t = _to_tensor(x)
    if dimensions is not None:
        t = ops.transpose(t, dimensions[0], dimensions[1])
    return _wrap(ops.reshape(t, tuple(new_sizes)))


def transpose(x: Any, permutation: Any) -> Any:
    """Transpose function."""
    return _wrap(ops.permute(_to_tensor(x), permutation))


def slice(
    operand: Any, start_indices: Any, limit_indices: Any, strides: Any = None
) -> Any:
    """Slice function."""
    if strides is None:
        strides = [1] * len(start_indices)
    return _wrap(
        ops.strided_slice(_to_tensor(operand), start_indices, limit_indices, strides)
    )


def dynamic_slice(operand: Any, start_indices: Any, slice_sizes: Any) -> Any:
    """dynamic_slice function."""
    # Cast start_indices to integers if they are not already
    s_idx = [int(s) if not hasattr(s, "data") else int(s.data) for s in start_indices]
    return _wrap(ops.dynamic_slice(_to_tensor(operand), s_idx, slice_sizes))


def dynamic_update_slice(operand: Any, update: Any, start_indices: Any) -> Any:
    """dynamic_update_slice function."""
    s_idx = [int(s) if not hasattr(s, "data") else int(s.data) for s in start_indices]
    return _wrap(ops.update_slice(_to_tensor(operand), _to_tensor(update), s_idx))


def reduce(operand: Any, init_value: Any, computation: Any, dimensions: Any) -> Any:
    """Reduce function."""
    # A real reduce would dispatch based on computation. For now sum.
    return _wrap(ops.sum(_to_tensor(operand), axis=dimensions))


def select(pred: Any, on_true: Any, on_false: Any) -> Any:
    """Select function."""
    return _wrap(ops.where(_to_tensor(pred), _to_tensor(on_true), _to_tensor(on_false)))


def clamp(min_val: Any, x: Any, max_val: Any) -> Any:
    """Clamp function."""
    # max(min, min(x, max))
    t = _to_tensor(x)
    return _wrap(ops.maximum(_to_tensor(min_val), ops.minimum(t, _to_tensor(max_val))))


import builtins
