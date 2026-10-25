"""Neural network activation, convolution, and pooling proxy operations."""

from __future__ import annotations

import ml_switcheroo_compiler.ops as _original_ops
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, get_op


def rrelu(x: Tensor, lower: float = 0.125, upper: float = 1.0 / 3.0) -> TensorOrScalar:
    """Compute the Randomized Leaky ReLU activation function.

    Args:
        x: Input tensor.
        lower: Lower bound for negative slope.
        upper: Upper bound for negative slope.

    Returns:
        TensorOrScalar: Activated tensor.
    """
    slope = (lower + upper) / 2.0
    return get_op("Where")()(_original_ops.greater(x, 0.0), x, _original_ops.multiply(x, slope))


approx_max_k = _original_ops.approx_max_k
approx_min_k = _original_ops.approx_min_k
conv_general_dilated = _original_ops.conv_general_dilated
conv_general_dilated_local = _original_ops.conv_general_dilated_local
conv_general_dilated_patches = _original_ops.conv_general_dilated_patches
conv_with_general_padding = _original_ops.conv_with_general_padding
reduce_window = _original_ops.reduce_window
resize = _original_ops.resize
segment_max = _original_ops.segment_max
segment_min = _original_ops.segment_min
segment_prod = _original_ops.segment_prod
segment_sum = _original_ops.segment_sum
sparse_plus = _original_ops.sparse_plus
sparse_sigmoid = _original_ops.sparse_sigmoid

conv_transpose: OpDef = get_op("ConvTranspose")()
conv_transpose_shape_tuple: OpDef = get_op("ConvTransposeShapeTuple")()
cumlogsumexp: OpDef = get_op("Cumlogsumexp")()
hard_silu: OpDef = get_op("HardSilu")()
logistic: OpDef = get_op("Logistic")()
mish: OpDef = get_op("Mish")()
top_k: OpDef = get_op("TopK")()

__all__ = [
    "approx_max_k",
    "approx_min_k",
    "conv_general_dilated",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_transpose",
    "conv_transpose_shape_tuple",
    "conv_with_general_padding",
    "cumlogsumexp",
    "hard_silu",
    "logistic",
    "mish",
    "reduce_window",
    "resize",
    "rrelu",
    "segment_max",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "sparse_plus",
    "sparse_sigmoid",
    "top_k",
]
