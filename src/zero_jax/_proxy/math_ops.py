"""Mathematical, elementwise, and reduction proxy operations."""

from __future__ import annotations

from collections.abc import Sequence

import ml_switcheroo_compiler.ops as _original_ops
from ml_switcheroo_compiler.core.config import config as _core_config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, get_op
from ml_switcheroo_compiler.ops.shape import dynamic_slicing as _dynamic_slicing
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def broadcast_to(x: Tensor, shape: Sequence[int]) -> TensorOrScalar:
    """Broadcast a tensor to a new target shape.

    Args:
        x: Input tensor to broadcast.
        shape: Target shape sequence.

    Returns:
        TensorOrScalar: Broadcasted tensor.
    """
    target_shape = tuple(int(s) for s in shape)
    if _core_config.eager_mode:
        return get_op("BroadcastTo")()(x, target_shape)
    out_shape = get_op("BroadcastTo")().infer_shape(x.shape, target_shape)
    return _emit_shape_node("BroadcastTo", [x], {"shape": target_shape}, out_shape, x.dtype)


def dynamic_slice(
    operand: Tensor,
    start_indices: Sequence[Tensor],
    slice_sizes: Sequence[int],
) -> TensorOrScalar:
    """Extract a dynamic slice from a tensor.

    Args:
        operand: Input tensor to slice.
        start_indices: Starting indices per dimension.
        slice_sizes: Slice lengths per dimension.

    Returns:
        TensorOrScalar: Extracted slice tensor.
    """
    return _dynamic_slicing.dynamic_slice(operand, start_indices, slice_sizes)


def update_slice(
    operand: Tensor,
    update: Tensor,
    start_indices: Sequence[Tensor],
) -> TensorOrScalar:
    """Update a dynamic slice in a tensor.

    Args:
        operand: Input tensor to be updated.
        update: Replacement slice tensor.
        start_indices: Starting indices per dimension.

    Returns:
        TensorOrScalar: Updated tensor.
    """
    return _dynamic_slicing.update_slice(operand, update, start_indices)


abs = _original_ops.abs
acos = _original_ops.acos
acosh = _original_ops.acosh
add = _original_ops.add
allclose = _original_ops.allclose
angle = _original_ops.angle
append = _original_ops.append
arange = _original_ops.arange
array = _original_ops.array
asin = _original_ops.asin
asinh = _original_ops.asinh
atan = _original_ops.atan
atan2 = _original_ops.atan2
atanh = _original_ops.atanh
atleast_1d = _original_ops.atleast_1d
atleast_2d = _original_ops.atleast_2d
atleast_3d = _original_ops.atleast_3d
bitwise_and = _original_ops.bitwise_and
bitwise_count = _original_ops.bitwise_count
bitwise_not = _original_ops.bitwise_not
bitwise_or = _original_ops.bitwise_or
bitwise_xor = _original_ops.bitwise_xor
block = _original_ops.block
cast = _original_ops.cast
cbrt = _original_ops.cbrt
ceil = _original_ops.ceil
column_stack = _original_ops.column_stack
conj = _original_ops.conj
copysign = _original_ops.copysign
cos = _original_ops.cos
cosh = _original_ops.cosh
count_nonzero = _original_ops.count_nonzero
cummax = _original_ops.cummax
cummin = _original_ops.cummin
cumprod = _original_ops.cumprod
cumsum = _original_ops.cumsum
deg2rad = _original_ops.deg2rad
delete = _original_ops.delete
diag = _original_ops.diag
diag_indices = _original_ops.diag_indices
diag_indices_from = _original_ops.diag_indices_from
diagflat = _original_ops.diagflat
diagonal = _original_ops.diagonal
divide = _original_ops.divide
divmod = _original_ops.divmod
dsplit = _original_ops.dsplit
dstack = _original_ops.dstack
ediff1d = _original_ops.ediff1d
empty = _original_ops.empty
equal = _original_ops.equal
exp = _original_ops.exp
exp2 = _original_ops.exp2
expand_dims = _original_ops.expand_dims
expm1 = _original_ops.expm1
eye = _original_ops.eye
fill_diagonal = _original_ops.fill_diagonal
fix = _original_ops.fix
flip = _original_ops.flip
fliplr = _original_ops.fliplr
flipud = _original_ops.flipud
float_power = _original_ops.float_power
floor = _original_ops.floor
floor_divide = _original_ops.floor_divide
fmax = _original_ops.fmax
fmin = _original_ops.fmin
fmod = _original_ops.fmod
frexp = _original_ops.frexp
full = _original_ops.full
full_like = _original_ops.full_like
gcd = _original_ops.gcd
greater = _original_ops.greater
greater_equal = _original_ops.greater_equal
heaviside = _original_ops.heaviside
hsplit = _original_ops.hsplit
hstack = _original_ops.hstack
hypot = _original_ops.hypot
identity = _original_ops.identity
imag = _original_ops.imag
insert = _original_ops.insert
isclose = _original_ops.isclose
iscomplex = _original_ops.iscomplex
iscomplexobj = _original_ops.iscomplexobj
isfinite = _original_ops.isfinite
isin = _original_ops.isin
isinf = _original_ops.isinf
isnan = _original_ops.isnan
isreal = _original_ops.isreal
isrealobj = _original_ops.isrealobj
issubdtype = _original_ops.issubdtype
lcm = _original_ops.lcm
ldexp = _original_ops.ldexp
left_shift = _original_ops.left_shift
less = _original_ops.less
less_equal = _original_ops.less_equal
log = _original_ops.log
log1p = _original_ops.log1p
logaddexp = _original_ops.logaddexp
logaddexp2 = _original_ops.logaddexp2
logical_and = _original_ops.logical_and
logical_not = _original_ops.logical_not
logical_or = _original_ops.logical_or
logical_xor = _original_ops.logical_xor
maximum = _original_ops.maximum
minimum = _original_ops.minimum
mod = _original_ops.mod
moveaxis = _original_ops.moveaxis
multiply = _original_ops.multiply
nan_to_num = _original_ops.nan_to_num
nanargmax = _original_ops.nanargmax
nanargmin = _original_ops.nanargmin
nancumprod = _original_ops.nancumprod
nancumsum = _original_ops.nancumsum
nanmean = _original_ops.nanmean
nanmedian = _original_ops.nanmedian
nanpercentile = _original_ops.nanpercentile
nanquantile = _original_ops.nanquantile
nanstd = _original_ops.nanstd
nanvar = _original_ops.nanvar
negative = _original_ops.negative
nextafter = _original_ops.nextafter
not_equal = _original_ops.not_equal
ones = _original_ops.ones
ones_like = _original_ops.ones_like
positive = _original_ops.positive
power = _original_ops.power
rad2deg = _original_ops.rad2deg
real = _original_ops.real
reciprocal = _original_ops.reciprocal
rem = _original_ops.rem
remainder = _original_ops.remainder
right_shift = _original_ops.right_shift
roll = _original_ops.roll
round = _original_ops.round
rsqrt = _original_ops.rsqrt
shape = _original_ops.shape
sign = _original_ops.sign
signbit = _original_ops.signbit
sin = _original_ops.sin
sinc = _original_ops.sinc
sinh = _original_ops.sinh
split = _original_ops.split
sqrt = _original_ops.sqrt
square = _original_ops.square
stack = _original_ops.stack
subtract = _original_ops.subtract
swapaxes = _original_ops.swapaxes
tan = _original_ops.tan
tanh = _original_ops.tanh
trunc = _original_ops.trunc
unstack = _original_ops.unstack
vsplit = _original_ops.vsplit
vstack = _original_ops.vstack
zeros = _original_ops.zeros
zeros_like = _original_ops.zeros_like

all: OpDef = get_op("All")()
any: OpDef = get_op("Any")()
argmax: OpDef = get_op("Argmax")()
argmin: OpDef = get_op("Argmin")()
argsort: OpDef = get_op("Argsort")()
max: OpDef = get_op("Max")()
mean: OpDef = get_op("Mean")()
min: OpDef = get_op("Min")()
nonzero: OpDef = get_op("Nonzero")()
permute: OpDef = get_op("Transpose")()
prod: OpDef = get_op("Prod")()
repeat: OpDef = get_op("Repeat")()
reshape: OpDef = get_op("Reshape")()
searchsorted: OpDef = get_op("SearchSorted")()
size: OpDef = get_op("Size")()
sort: OpDef = get_op("Sort")()
squeeze: OpDef = get_op("Squeeze")()
sum: OpDef = get_op("Sum")()
take: OpDef = get_op("Take")()
take_along_axis: OpDef = get_op("TakeAlongAxis")()
tile: OpDef = get_op("Tile")()
transpose: OpDef = get_op("Transpose")()
tril: OpDef = get_op("Tril")()
triu: OpDef = get_op("Triu")()
unique: OpDef = get_op("Unique")()
where: OpDef = get_op("Where")()

__all__ = [
    "abs",
    "acos",
    "acosh",
    "add",
    "all",
    "allclose",
    "angle",
    "any",
    "append",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "array",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "bitwise_and",
    "bitwise_count",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "block",
    "broadcast_to",
    "cast",
    "cbrt",
    "ceil",
    "column_stack",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "count_nonzero",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "deg2rad",
    "delete",
    "diag",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "diagonal",
    "divide",
    "divmod",
    "dsplit",
    "dstack",
    "dynamic_slice",
    "ediff1d",
    "empty",
    "equal",
    "exp",
    "exp2",
    "expand_dims",
    "expm1",
    "eye",
    "fill_diagonal",
    "fix",
    "flip",
    "fliplr",
    "flipud",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "full",
    "full_like",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hsplit",
    "hstack",
    "hypot",
    "identity",
    "imag",
    "insert",
    "isclose",
    "iscomplex",
    "iscomplexobj",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isreal",
    "isrealobj",
    "issubdtype",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "log",
    "log1p",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mod",
    "moveaxis",
    "multiply",
    "nan_to_num",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmean",
    "nanmedian",
    "nanpercentile",
    "nanquantile",
    "nanstd",
    "nanvar",
    "negative",
    "nextafter",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "permute",
    "positive",
    "power",
    "prod",
    "rad2deg",
    "real",
    "reciprocal",
    "rem",
    "remainder",
    "repeat",
    "reshape",
    "right_shift",
    "roll",
    "round",
    "rsqrt",
    "searchsorted",
    "shape",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "size",
    "sort",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "subtract",
    "sum",
    "swapaxes",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tile",
    "transpose",
    "tril",
    "triu",
    "trunc",
    "unique",
    "unstack",
    "update_slice",
    "vsplit",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]
