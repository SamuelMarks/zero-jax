"""Static compiler proxy operations module for zero-jax."""

from __future__ import annotations

from collections.abc import Sequence

import ml_switcheroo_compiler.ops as _original_ops
from ml_switcheroo_compiler.core.config import config as _core_config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, get_op
from ml_switcheroo_compiler.ops.shape import dynamic_slicing as _dynamic_slicing
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

from zero_jax._types import VariadicCallable


def _make_unimplemented_op(name: str) -> VariadicCallable[TensorOrScalar]:
    """Create a typed callable stub that raises AttributeError when invoked.

    Args:
        name: Name of the unimplemented compiler operation.

    Returns:
        VariadicCallable[TensorOrScalar]: Stub callable raising AttributeError.
    """

    def _stub(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Raise AttributeError for unimplemented compiler operation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorOrScalar: Never returns.

        Raises:
            AttributeError: Always raised for unimplemented operations.
        """
        raise AttributeError(f"module 'ml_switcheroo_compiler.ops' has no attribute '{name}'")

    _stub.__name__ = name
    return _stub


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


abs = _original_ops.abs
acos = _original_ops.acos
acosh = _original_ops.acosh
add = _original_ops.add
allclose = _original_ops.allclose
angle = _original_ops.angle
append = _original_ops.append
approx_max_k = _original_ops.approx_max_k
approx_min_k = _original_ops.approx_min_k
arange = _original_ops.arange
array = _original_ops.array
asin = _original_ops.asin
asinh = _original_ops.asinh
associative_scan = _original_ops.associative_scan
atan = _original_ops.atan
atan2 = _original_ops.atan2
atanh = _original_ops.atanh
atleast_1d = _original_ops.atleast_1d
atleast_2d = _original_ops.atleast_2d
atleast_3d = _original_ops.atleast_3d
bessel_i0e = _original_ops.bessel_i0e
bessel_i1e = _original_ops.bessel_i1e
betainc = _original_ops.betainc
bitwise_and = _original_ops.bitwise_and
bitwise_count = _original_ops.bitwise_count
bitwise_not = _original_ops.bitwise_not
bitwise_or = _original_ops.bitwise_or
bitwise_xor = _original_ops.bitwise_xor
block = _original_ops.block
cast = _original_ops.cast
cbrt = _original_ops.cbrt
ceil = _original_ops.ceil
chebyshev_polynomial_t = _original_ops.chebyshev_polynomial_t
chebyshev_polynomial_u = _original_ops.chebyshev_polynomial_u
cholesky_solve = _original_ops.cholesky_solve
column_stack = _original_ops.column_stack
cond = _original_ops.cond
confusion_matrix = _original_ops.confusion_matrix
conj = _original_ops.conj
conv_general_dilated = _original_ops.conv_general_dilated
conv_general_dilated_local = _original_ops.conv_general_dilated_local
conv_general_dilated_patches = _original_ops.conv_general_dilated_patches
conv_with_general_padding = _original_ops.conv_with_general_padding
convolve = _original_ops.convolve
convolve2d = _original_ops.convolve2d
copysign = _original_ops.copysign
corrcoef = _original_ops.corrcoef
correlate = _original_ops.correlate
cos = _original_ops.cos
cosh = _original_ops.cosh
count_nonzero = _original_ops.count_nonzero
cov = _original_ops.cov
create_token = _original_ops.create_token
cross = _original_ops.cross
cummax = _original_ops.cummax
cummin = _original_ops.cummin
cumprod = _original_ops.cumprod
cumsum = _original_ops.cumsum
custom_gradient = _original_ops.custom_gradient
debug_infs = _original_ops.debug_infs
debug_nans = _original_ops.debug_nans
decode_csv = _original_ops.decode_csv
decode_image = _original_ops.decode_image
deg2rad = _original_ops.deg2rad
delete = _original_ops.delete
diag = _original_ops.diag
diag_indices = _original_ops.diag_indices
diag_indices_from = _original_ops.diag_indices_from
diagflat = _original_ops.diagflat
diagonal = _original_ops.diagonal
digamma = _original_ops.digamma
divide = _original_ops.divide
divmod = _original_ops.divmod
dot = _original_ops.dot
dot_general = _original_ops.dot_general
dsplit = _original_ops.dsplit
dstack = _original_ops.dstack
ediff1d = _original_ops.ediff1d
eigh_tridiagonal = _original_ops.eigh_tridiagonal
einsum = _original_ops.einsum
empty = _original_ops.empty
equal = _original_ops.equal
erf = _original_ops.erf
erf_inv = _original_ops.erf_inv
erfc = _original_ops.erfc
erfinv = _original_ops.erfinv
exp = _original_ops.exp
exp2 = _original_ops.exp2
expand_dims = _original_ops.expand_dims
expm = _original_ops.expm
expm1 = _original_ops.expm1
eye = _original_ops.eye
fftconvolve = _original_ops.fftconvolve
fftn = _original_ops.fftn
fftshift = _original_ops.fftshift
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
fori_loop = _original_ops.fori_loop
frexp = _original_ops.frexp
fromfile = _original_ops.fromfile
fromfunction = _original_ops.fromfunction
fromiter = _original_ops.fromiter
fromstring = _original_ops.fromstring
full = _original_ops.full
full_like = _original_ops.full_like
gcd = _original_ops.gcd
greater = _original_ops.greater
greater_equal = _original_ops.greater_equal
heaviside = _original_ops.heaviside
hermite_polynomial_h = _original_ops.hermite_polynomial_h
hermite_polynomial_he = _original_ops.hermite_polynomial_he
hfft = _original_ops.hfft
hsplit = _original_ops.hsplit
hstack = _original_ops.hstack
hypot = _original_ops.hypot
identity = _original_ops.identity
ifft = _original_ops.ifft
ifft2 = _original_ops.ifft2
ifftn = _original_ops.ifftn
ifftshift = _original_ops.ifftshift
igamma = _original_ops.igamma
igammac = _original_ops.igammac
imag = _original_ops.imag
inner = _original_ops.inner
insert = _original_ops.insert
irfft2 = _original_ops.irfft2
irfftn = _original_ops.irfftn
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
istft = _original_ops.istft
laguerre_polynomial_l = _original_ops.laguerre_polynomial_l
lcm = _original_ops.lcm
ldexp = _original_ops.ldexp
left_shift = _original_ops.left_shift
legendre_polynomial_p = _original_ops.legendre_polynomial_p
less = _original_ops.less
less_equal = _original_ops.less_equal
lgamma = _original_ops.lgamma
load = _original_ops.load
log = _original_ops.log
log1p = _original_ops.log1p
logaddexp = _original_ops.logaddexp
logaddexp2 = _original_ops.logaddexp2
logical_and = _original_ops.logical_and
logical_not = _original_ops.logical_not
logical_or = _original_ops.logical_or
logical_xor = _original_ops.logical_xor
logit = _original_ops.logit
logsumexp = _original_ops.logsumexp
lstsq = _original_ops.lstsq
lu = _original_ops.lu
lu_matrix_inverse = _original_ops.lu_matrix_inverse
lu_reconstruct = _original_ops.lu_reconstruct
lu_solve = _original_ops.lu_solve
map_coordinates = _original_ops.map_coordinates
matmul = _original_ops.matmul
matrix_norm = _original_ops.matrix_norm
matrix_rank = _original_ops.matrix_rank
matrix_transpose = _original_ops.matrix_transpose
maximum = _original_ops.maximum
minimum = _original_ops.minimum
mod = _original_ops.mod
modified_bessel_i0 = _original_ops.modified_bessel_i0
modified_bessel_i1 = _original_ops.modified_bessel_i1
modified_bessel_k0 = _original_ops.modified_bessel_k0
modified_bessel_k1 = _original_ops.modified_bessel_k1
moveaxis = _original_ops.moveaxis
multi_dot = _original_ops.multi_dot
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
ndtri = _original_ops.ndtri
negative = _original_ops.negative
nextafter = _original_ops.nextafter
norm = _original_ops.norm
not_equal = _original_ops.not_equal
ones = _original_ops.ones
ones_like = _original_ops.ones_like
outer = _original_ops.outer
parse_example = _original_ops.parse_example
parse_tensor = _original_ops.parse_tensor
pmean = _original_ops.pmean
polar = _original_ops.polar
polygamma = _original_ops.polygamma
positive = _original_ops.positive
power = _original_ops.power
psum = _original_ops.psum
rad2deg = _original_ops.rad2deg
read_file = _original_ops.read_file
real = _original_ops.real
reciprocal = _original_ops.reciprocal
reduce_window = _original_ops.reduce_window
rem = _original_ops.rem
remainder = _original_ops.remainder
resize = _original_ops.resize
rfft2 = _original_ops.rfft2
rfftfreq = _original_ops.rfftfreq
rfftn = _original_ops.rfftn
right_shift = _original_ops.right_shift
roll = _original_ops.roll
round = _original_ops.round
rsqrt = _original_ops.rsqrt
save = _original_ops.save
savez = _original_ops.savez
segment_max = _original_ops.segment_max
segment_min = _original_ops.segment_min
segment_prod = _original_ops.segment_prod
segment_sum = _original_ops.segment_sum
serialize_tensor = _original_ops.serialize_tensor
shape = _original_ops.shape
shifted_chebyshev_polynomial_t = _original_ops.shifted_chebyshev_polynomial_t
shifted_chebyshev_polynomial_u = _original_ops.shifted_chebyshev_polynomial_u
shifted_chebyshev_polynomial_v = _original_ops.shifted_chebyshev_polynomial_v
shifted_chebyshev_polynomial_w = _original_ops.shifted_chebyshev_polynomial_w
sign = _original_ops.sign
signbit = _original_ops.signbit
sin = _original_ops.sin
sinc = _original_ops.sinc
sinh = _original_ops.sinh
sparse_plus = _original_ops.sparse_plus
sparse_sigmoid = _original_ops.sparse_sigmoid
split = _original_ops.split
sqrt = _original_ops.sqrt
sqrtm = _original_ops.sqrtm
square = _original_ops.square
stack = _original_ops.stack
stft = _original_ops.stft
subtract = _original_ops.subtract
svdvals = _original_ops.svdvals
swapaxes = _original_ops.swapaxes
tan = _original_ops.tan
tanh = _original_ops.tanh
tensordot = _original_ops.tensordot
tensorinv = _original_ops.tensorinv
tensorsolve = _original_ops.tensorsolve
trace = _original_ops.trace
triangular_solve = _original_ops.triangular_solve
tridiagonal_solve = _original_ops.tridiagonal_solve
trunc = _original_ops.trunc
unstack = _original_ops.unstack
vdot = _original_ops.vdot
vecdot = _original_ops.vecdot
vector_norm = _original_ops.vector_norm
vsplit = _original_ops.vsplit
vstack = _original_ops.vstack
welch = _original_ops.welch
while_loop = _original_ops.while_loop
write_file = _original_ops.write_file
xlog1py = _original_ops.xlog1py
xlogy = _original_ops.xlogy
zeros = _original_ops.zeros
zeros_like = _original_ops.zeros_like
zeta = _original_ops.zeta
Atom: OpDef = get_op("Atom")()
Transpose: OpDef = get_op("Transpose")()
all: OpDef = get_op("All")()
all_gather: OpDef = get_op("AllGather")()
all_to_all: OpDef = get_op("AllToAll")()
any: OpDef = get_op("Any")()
apply_over_axes: OpDef = get_op("ApplyOverAxes")()
argmax: OpDef = get_op("Argmax")()
argmin: OpDef = get_op("Argmin")()
argpartition: OpDef = get_op("Argpartition")()
argsort: OpDef = get_op("Argsort")()
argwhere: OpDef = get_op("Argwhere")()
array_equiv: OpDef = get_op("ArrayEquiv")()
array_repr: OpDef = get_op("ArrayRepr")()
array_str: OpDef = get_op("ArrayStr")()
average: OpDef = get_op("Average")()
axis_index: OpDef = get_op("AxisIndex")()
ball: OpDef = get_op("Ball")()
bernoulli: OpDef = get_op("Bernoulli")()
bessel_jn: OpDef = get_op("BesselJn")()
beta: OpDef = get_op("Beta")()
bincount: OpDef = get_op("Bincount")()
binomial: OpDef = get_op("Binomial")()
bitcast_convert_type: OpDef = get_op("BitcastConvertType")()
bits: OpDef = get_op("Bits")()
block_diag: OpDef = get_op("BlockDiag")()
broadcast: OpDef = get_op("Broadcast")()
broadcast_arrays: OpDef = get_op("BroadcastArrays")()
broadcast_to_rank: OpDef = get_op("BroadcastToRank")()
broadcasted_iota: OpDef = get_op("BroadcastedIota")()
c_: OpDef = get_op("C")()
cache: OpDef = get_op("Cache")()
can_cast: OpDef = get_op("CanCast")()
cauchy: OpDef = get_op("Cauchy")()
chisquare: OpDef = get_op("Chisquare")()
cholesky: OpDef = get_op("Cholesky")()
choose: OpDef = get_op("Choose")()
clone: OpDef = get_op("Clone")()
clz: OpDef = get_op("Clz")()
collapse: OpDef = get_op("Collapse")()
compress: OpDef = get_op("Compress")()
concatenate: OpDef = get_op("Concatenate")()
conv_transpose: OpDef = get_op("ConvTranspose")()
conv_transpose_shape_tuple: OpDef = get_op("ConvTransposeShapeTuple")()
cumlogsumexp: OpDef = get_op("Cumlogsumexp")()
custom_linear_solve: OpDef = get_op("CustomLinearSolve")()
custom_root: OpDef = get_op("CustomRoot")()
descriptive: OpDef = get_op("Descriptive")()
det: OpDef = get_op("Det")()
diff: OpDef = get_op("Diff")()
digitize: OpDef = get_op("Digitize")()
dirichlet: OpDef = get_op("Dirichlet")()
distributions: OpDef = get_op("Distributions")()
double_sided_maxwell: OpDef = get_op("DoubleSidedMaxwell")()
dynamic_index_in_dim: OpDef = get_op("DynamicIndexInDim")()
dynamic_slice_in_dim: OpDef = get_op("DynamicSliceInDim")()
dynamic_update_index_in_dim: OpDef = get_op("DynamicUpdateIndexInDim")()
dynamic_update_slice_in_dim: OpDef = get_op("DynamicUpdateSliceInDim")()
eig: OpDef = get_op("Eig")()
eigh: OpDef = get_op("Eigh")()
eigvals: OpDef = get_op("Eigvals")()
eigvalsh: OpDef = get_op("Eigvalsh")()
einsum_path: OpDef = get_op("EinsumPath")()
exponential: OpDef = get_op("Exponential")()
extract: OpDef = get_op("Extract")()
f: OpDef = get_op("F")()
fabs: OpDef = get_op("Fabs")()
fft: OpDef = get_op("Fft")()
fft2: OpDef = get_op("Fft2")()
fftfreq: OpDef = get_op("Fftfreq")()
flatnonzero: OpDef = get_op("Flatnonzero")()
from_dlpack: OpDef = get_op("FromDlpack")()
frompyfunc: OpDef = get_op("Frompyfunc")()
gamma: OpDef = get_op("Gamma")()
gather: OpDef = get_op("Gather")()
gather_nd: OpDef = get_op("GatherNd")()
generalized_normal: OpDef = get_op("GeneralizedNormal")()
geometric: OpDef = get_op("Geometric")()
geomspace: OpDef = get_op("Geomspace")()
get_printoptions: OpDef = get_op("GetPrintoptions")()
gumbel: OpDef = get_op("Gumbel")()
hard_silu: OpDef = get_op("HardSilu")()
hessenberg: OpDef = get_op("Hessenberg")()
histogram_bin_edges: OpDef = get_op("HistogramBinEdges")()
householder_product: OpDef = get_op("HouseholderProduct")()
igamma_grad_a: OpDef = get_op("IgammaGradA")()
ihfft: OpDef = get_op("Ihfft")()
index_in_dim: OpDef = get_op("IndexInDim")()
infeed: OpDef = get_op("Infeed")()
inv: OpDef = get_op("Inv")()
irfft: OpDef = get_op("Irfft")()
iterable: OpDef = get_op("Iterable")()
key: OpDef = get_op("Key")()
key_data: OpDef = get_op("KeyData")()
key_impl: OpDef = get_op("KeyImpl")()
laplace: OpDef = get_op("Laplace")()
lexsort: OpDef = get_op("Lexsort")()
loggamma: OpDef = get_op("Loggamma")()
logistic: OpDef = get_op("Logistic")()
lognormal: OpDef = get_op("Lognormal")()
lu_factor: OpDef = get_op("LuFactor")()
lu_pivots_to_permutation: OpDef = get_op("LuPivotsToPermutation")()
mask_indices: OpDef = get_op("MaskIndices")()
matrix_power: OpDef = get_op("MatrixPower")()
max: OpDef = get_op("Max")()
maxwell: OpDef = get_op("Maxwell")()
mean: OpDef = get_op("Mean")()
mgrid: OpDef = get_op("Mgrid")()
min: OpDef = get_op("Min")()
mish: OpDef = get_op("Mish")()
multivariate_normal: OpDef = get_op("MultivariateNormal")()
nonzero: OpDef = get_op("Nonzero")()
orthogonal: OpDef = get_op("Orthogonal")()
outfeed: OpDef = get_op("Outfeed")()
packbits: OpDef = get_op("Packbits")()
pad: OpDef = get_op("Pad")()
pareto: OpDef = get_op("Pareto")()
pdot: OpDef = get_op("Pdot")()
percentile: OpDef = get_op("Percentile")()
permute: OpDef = get_op("Transpose")()
pinv: OpDef = get_op("Pinv")()
poisson: OpDef = get_op("Poisson")()
poly: OpDef = get_op("Poly")()
polyadd: OpDef = get_op("Polyadd")()
polyder: OpDef = get_op("Polyder")()
polydiv: OpDef = get_op("Polydiv")()
polyfit: OpDef = get_op("Polyfit")()
polyint: OpDef = get_op("Polyint")()
polymul: OpDef = get_op("Polymul")()
polysub: OpDef = get_op("Polysub")()
polyval: OpDef = get_op("Polyval")()
population_count: OpDef = get_op("PopulationCount")()
prod: OpDef = get_op("Prod")()
promote_types: OpDef = get_op("PromoteTypes")()
psum_scatter: OpDef = get_op("PsumScatter")()
qdwh: OpDef = get_op("Qdwh")()
qr: OpDef = get_op("Qr")()
quantile: OpDef = get_op("Quantile")()
ragged_dot: OpDef = get_op("RaggedDot")()
random_gamma_grad: OpDef = get_op("RandomGammaGrad")()
ravel_multi_index: OpDef = get_op("RavelMultiIndex")()
rayleigh: OpDef = get_op("Rayleigh")()
reduce_precision: OpDef = get_op("ReducePrecision")()
repeat: OpDef = get_op("Repeat")()
reshape: OpDef = get_op("Reshape")()
result_type: OpDef = get_op("ResultType")()
rfft: OpDef = get_op("Rfft")()
roots: OpDef = get_op("Roots")()
scatter: OpDef = get_op("Scatter")()
scatter_add: OpDef = get_op("ScatterAdd")()
scatter_apply: OpDef = get_op("ScatterApply")()
scatter_max: OpDef = get_op("ScatterMax")()
scatter_min: OpDef = get_op("ScatterMin")()
scatter_mul: OpDef = get_op("ScatterMul")()
scatter_nd: OpDef = get_op("ScatterNd")()
schur: OpDef = get_op("Schur")()
searchsorted: OpDef = get_op("SearchSorted")()
size: OpDef = get_op("Size")()
slice_in_dim: OpDef = get_op("SliceInDim")()
slogdet: OpDef = get_op("Slogdet")()
solve: OpDef = get_op("Solve")()
sort: OpDef = get_op("Sort")()
sort_complex: OpDef = get_op("SortComplex")()
sort_key_val: OpDef = get_op("SortKeyVal")()
spence: OpDef = get_op("Spence")()
squeeze: OpDef = get_op("Squeeze")()
strided_slice: OpDef = get_op("StridedSlice")()
sum: OpDef = get_op("Sum")()
svd: OpDef = get_op("Svd")()
t: OpDef = get_op("T")()
take: OpDef = get_op("Take")()
take_along_axis: OpDef = get_op("TakeAlongAxis")()
tile: OpDef = get_op("Tile")()
to_dlpack: OpDef = get_op("ToDlpack")()
top_k: OpDef = get_op("TopK")()
transpose: OpDef = get_op("Transpose")()
triangular: OpDef = get_op("Triangular")()
tridiagonal: OpDef = get_op("Tridiagonal")()
tril: OpDef = get_op("Tril")()
tril_indices: OpDef = get_op("TrilIndices")()
tril_indices_from: OpDef = get_op("TrilIndicesFrom")()
trim_zeros: OpDef = get_op("TrimZeros")()
triu: OpDef = get_op("Triu")()
triu_indices: OpDef = get_op("TriuIndices")()
triu_indices_from: OpDef = get_op("TriuIndicesFrom")()
unique: OpDef = get_op("Unique")()
unpackbits: OpDef = get_op("Unpackbits")()
wald: OpDef = get_op("Wald")()
weibull_min: OpDef = get_op("WeibullMin")()
where: OpDef = get_op("Where")()
wrap_key_data: OpDef = get_op("WrapKeyData")()
AbstractedAxesSpec: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("AbstractedAxesSpec")
ArrayLike: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ArrayLike")
AxisSize: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("AxisSize")
BatchingRule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("BatchingRule")
Const: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("Const")
ConstFoldRule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ConstFoldRule")
CurrentSourceInfoMetadata: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("CurrentSourceInfoMetadata")
DCERule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("DCERule")
DEFINE_bool: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("DEFINE_bool")
DEFINE_enum: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("DEFINE_enum")
DEFINE_float: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("DEFINE_float")
DEFINE_integer: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("DEFINE_integer")
DEFINE_string: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("DEFINE_string")
DTypeLike: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("DTypeLike")
Elt: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("Elt")
ForwardingRule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ForwardingRule")
FromEltHandler: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("FromEltHandler")
GetIdx: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("GetIdx")
Index: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("Index")
JaxprTracerRecipe: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("JaxprTracerRecipe")
LoadedExecutable_execute: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("LoadedExecutable_execute")
LoadedExecutable_execute_with_token: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("LoadedExecutable_execute_with_token")
MakeIotaHandler: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("MakeIotaHandler")
MapSpec: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("MapSpec")
MeshAxisName: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("MeshAxisName")
NamedTuple: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("NamedTuple")
NoReturn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("NoReturn")
ParamsUpdater: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ParamsUpdater")
PartialEvalCustomRule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("PartialEvalCustomRule")
ResAvalUpdater: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ResAvalUpdater")
ToEltHandler: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ToEltHandler")
Vmappable: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("Vmappable")
abstract_eval_fun: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("abstract_eval_fun")
add_jaxvals: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("add_jaxvals")
add_tangents: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("add_tangents")
annotate_function: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("annotate_function")
api_boundary: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("api_boundary")
apply_todos: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("apply_todos")
are_op_shardings_equal: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("are_op_shardings_equal")
arg_info_all: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("arg_info_all")
argnums_partial: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("argnums_partial")
array_mapping_to_axis_resources: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("array_mapping_to_axis_resources")
as_hashable_function: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("as_hashable_function")
as_named_shape: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("as_named_shape")
assert_unreachable: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("assert_unreachable")
axis_frame: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("axis_frame")
backward_pass: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("backward_pass")
backward_pass_internal: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("backward_pass_internal")
bartlett: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("bartlett")
batch: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("batch")
batch_custom_jvp_subtrace: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("batch_custom_jvp_subtrace")
batch_custom_vjp_bwd: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("batch_custom_vjp_bwd")
batch_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("batch_jaxpr")
batch_jaxpr2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("batch_jaxpr2")
batch_jaxpr_axes: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("batch_jaxpr_axes")
batch_subtrace: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("batch_subtrace")
bcoo_cusparse_lowering: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("bcoo_cusparse_lowering")
bdim_at_front: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("bdim_at_front")
betaln: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("betaln")
bicgstab: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("bicgstab")
bilinear_transpose: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("bilinear_transpose")
blackman: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("blackman")
bool_env: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("bool_env")
breakpoint: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("breakpoint")
broadcast_batcher: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("broadcast_batcher")
call: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("call")
call_bind_with_continuation: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("call_bind_with_continuation")
call_impl: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("call_impl")
call_padding_rule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("call_padding_rule")
call_partial_eval_custom_rule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("call_partial_eval_custom_rule")
call_transpose: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("call_transpose")
callback: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("callback")
canonicalize_axis: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("canonicalize_axis")
canonicalize_dtype: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("canonicalize_dtype")
cdf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("cdf")
ceil_of_ratio: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ceil_of_ratio")
cg: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("cg")
check_eqn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("check_eqn")
check_exists: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("check_exists")
check_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("check_jaxpr")
check_toposort: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("check_toposort")
check_tracer_leaks: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("check_tracer_leaks")
check_type: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("check_type")
check_valid_jaxtype: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("check_valid_jaxtype")
checking_leaks: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("checking_leaks")
cho_factor: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("cho_factor")
cho_solve: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("cho_solve")
clear_all_caches: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("clear_all_caches")
clear_all_weakref_lru_caches: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("clear_all_weakref_lru_caches")
clear_event_listeners: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("clear_event_listeners")
close_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("close_jaxpr")
closed_backward_pass: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("closed_backward_pass")
closed_call_partial_eval_custom_rule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("closed_call_partial_eval_custom_rule")
closure_convert: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("closure_convert")
compilation_cache_dir: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("compilation_cache_dir")
compilation_cache_include_metadata_in_key: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("compilation_cache_include_metadata_in_key")
compilation_cache_max_size: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("compilation_cache_max_size")
concrete_aval: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("concrete_aval")
concrete_or_error: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("concrete_or_error")
concretization_function_error: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("concretization_function_error")
contextmanager: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("contextmanager")
convert_constvars_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("convert_constvars_jaxpr")
convert_envvars_to_constvars: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("convert_envvars_to_constvars")
convert_invars_to_constvars: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("convert_invars_to_constvars")
correlate2d: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("correlate2d")
csd: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("csd")
cur_sublevel: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("cur_sublevel")
curry: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("curry")
custom_vjp_disable_shape_check: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("custom_vjp_disable_shape_check")
custom_vjp_primal_tree_values: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("custom_vjp_primal_tree_values")
dce_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("dce_jaxpr")
dce_jaxpr_call_rule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("dce_jaxpr_call_rule")
dce_jaxpr_closed_call_rule: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("dce_jaxpr_closed_call_rule")
dce_jaxpr_consts: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("dce_jaxpr_consts")
dct: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("dct")
dctn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("dctn")
debug_info: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("debug_info")
debug_info_final: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("debug_info_final")
debug_key_reuse: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("debug_key_reuse")
dedup_referents: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("dedup_referents")
def_trivial_padding: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("def_trivial_padding")
default_backend: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("default_backend")
default_device: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("default_device")
default_dtype_bits: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("default_dtype_bits")
default_matmul_precision: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("default_matmul_precision")
default_prng_impl: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("default_prng_impl")
defbilinear: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("defbilinear")
defbroadcasting: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("defbroadcasting")
define_bool_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("define_bool_state")
define_enum_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("define_enum_state")
define_float_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("define_float_state")
define_int_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("define_int_state")
define_optional_enum_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("define_optional_enum_state")
define_optional_string_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("define_optional_string_state")
define_string_or_object_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("define_string_or_object_state")
define_string_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("define_string_state")
defjvp: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("defjvp")
defjvp2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("defjvp2")
defjvp_zero: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("defjvp_zero")
deflinear: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("deflinear")
deflinear2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("deflinear2")
defreducer: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("defreducer")
defvectorized: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("defvectorized")
detrend: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("detrend")
device_memory_profile: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("device_memory_profile")
disable_all_debug_logging: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("disable_all_debug_logging")
disable_jit: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("disable_jit")
disable_x64: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("disable_x64")
distributed_debug: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("distributed_debug")
distributed_debug_log: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("distributed_debug_log")
do_subst_axis_names_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("do_subst_axis_names_jaxpr")
donation_vector: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("donation_vector")
dtype_to_etype: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("dtype_to_etype")
dynamic_shapes: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("dynamic_shapes")
eager_pmap: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("eager_pmap")
enable_checks: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("enable_checks")
enable_compilation_cache: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("enable_compilation_cache")
enable_custom_prng: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("enable_custom_prng")
enable_custom_vjp_by_custom_transpose: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("enable_custom_vjp_by_custom_transpose")
enable_debug_logging: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("enable_debug_logging")
enable_memories: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("enable_memories")
enable_pgle: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("enable_pgle")
enable_x64: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("enable_x64")
ensure_compile_time_eval: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ensure_compile_time_eval")
entr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("entr")
escaped_tracer_error: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("escaped_tracer_error")
eval_context: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("eval_context")
eval_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("eval_jaxpr")
execute_with_python_values: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("execute_with_python_values")
execute_with_python_values_replicated: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("execute_with_python_values_replicated")
exp1: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("exp1")
expi: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("expi")
expit: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("expit")
explain_cache_misses: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("explain_cache_misses")
explicit_device_get_scope: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("explicit_device_get_scope")
explicit_device_put_scope: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("explicit_device_put_scope")
expm_frechet: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("expm_frechet")
expn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("expn")
extend_axis_env: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("extend_axis_env")
extend_axis_env_nd: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("extend_axis_env_nd")
extend_jaxpr_stack: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("extend_jaxpr_stack")
f_jvp_traceable: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("f_jvp_traceable")
factorial: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("factorial")
filter_traceback: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("filter_traceback")
find_top_trace: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("find_top_trace")
finfo: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("finfo")
flatten: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("flatten")
flatten_axes: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("flatten_axes")
flatten_fun: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("flatten_fun")
flatten_fun_for_vmap: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("flatten_fun_for_vmap")
flatten_fun_nokwargs: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("flatten_fun_nokwargs")
format_exception_only: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("format_exception_only")
from_elt: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("from_elt")
full_lower: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("full_lower")
fun_name: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("fun_name")
funm: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("funm")
gammainc: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("gammainc")
gammaincc: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("gammaincc")
gammaln: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("gammaln")
gammasgn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("gammasgn")
generate_pjrt_gpu_plugin_options: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("generate_pjrt_gpu_plugin_options")
gensym: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("gensym")
get_aval: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("get_aval")
get_backend: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("get_backend")
get_compile_options: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("get_compile_options")
get_enable_x64: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("get_enable_x64")
get_primitive_transpose: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("get_primitive_transpose")
get_referent: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("get_referent")
global_aval_to_result_handler: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("global_aval_to_result_handler")
global_avals_to_results_handler: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("global_avals_to_results_handler")
global_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("global_state")
gmres: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("gmres")
gradient: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("gradient")
hamming: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("hamming")
hanning: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("hanning")
hard_swish: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("hard_swish")
heap_profile: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("heap_profile")
hilbert: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("hilbert")
histogram: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("histogram")
histogram2d: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("histogram2d")
histogramdd: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("histogramdd")
hlo_source_file_canonicalization_regex: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("hlo_source_file_canonicalization_regex")
hyp1f1: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("hyp1f1")
i0: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("i0")
i0e: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("i0e")
i1: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("i1")
i1e: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("i1e")
idct: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("idct")
idctn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("idctn")
iinfo: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("iinfo")
include_frame: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("include_frame")
include_full_tracebacks_in_locations: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("include_full_tracebacks_in_locations")
indices: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("indices")
infer_lambda_input_type: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("infer_lambda_input_type")
initialize: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("initialize")
initialize_cache: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("initialize_cache")
initialize_pjrt_plugin: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("initialize_pjrt_plugin")
inspect_array_sharding: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("inspect_array_sharding")
instantiate_const_at: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("instantiate_const_at")
instantiate_zeros: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("instantiate_zeros")
int_env: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("int_env")
interp: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("interp")
intersect1d: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("intersect1d")
io_callback: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("io_callback")
is_constant_dim: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("is_constant_dim")
is_constant_shape: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("is_constant_shape")
is_initialized: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("is_initialized")
is_op_sharding_replicated: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("is_op_sharding_replicated")
is_undefined_primal: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("is_undefined_primal")
is_vmappable: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("is_vmappable")
isf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("isf")
isscalar: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("isscalar")
ix_: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ix_")
jax2tf_associative_scan_reductions: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jax2tf_associative_scan_reductions")
jax2tf_default_native_serialization: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jax2tf_default_native_serialization")
jax_export_calling_convention_version: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jax_export_calling_convention_version")
jax_pjrt_client_create_options: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jax_pjrt_client_create_options")
jax_platforms: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jax_platforms")
jax_serialization_version: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jax_serialization_version")
jax_xla_profile_version: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jax_xla_profile_version")
jaxpr_as_fun: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jaxpr_as_fun")
jaxpr_uses_outfeed: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jaxpr_uses_outfeed")
jaxprs_in_params: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jaxprs_in_params")
join_effects: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("join_effects")
join_named_shapes: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("join_named_shapes")
jvp: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jvp")
jvp_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jvp_jaxpr")
jvp_subtrace: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jvp_subtrace")
jvp_subtrace_aux: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jvp_subtrace_aux")
jvpfun: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("jvpfun")
kaiser: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("kaiser")
kl_div: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("kl_div")
kron: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("kron")
lattice_join: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("lattice_join")
leaked_tracer_error: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("leaked_tracer_error")
legacy_prng_key: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("legacy_prng_key")
linear_call: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("linear_call")
linear_jvp: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("linear_jvp")
linear_transpose: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("linear_transpose")
linear_transpose2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("linear_transpose2")
linearize: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("linearize")
load_pjrt_plugin_dynamically: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("load_pjrt_plugin_dynamically")
load_pjrt_plugin_with_c_api: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("load_pjrt_plugin_with_c_api")
log_checkpoint_residuals: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("log_checkpoint_residuals")
log_compiles: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("log_compiles")
log_ndtr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("log_ndtr")
logcdf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("logcdf")
logpdf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("logpdf")
logpmf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("logpmf")
logsf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("logsf")
lpmn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("lpmn")
lpmn_values: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("lpmn_values")
make_c_api_client: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_c_api_client")
make_c_api_device_topology: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_c_api_device_topology")
make_convolution_dimension_numbers: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_convolution_dimension_numbers")
make_cpu_client: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_cpu_client")
make_dot_dimension_numbers: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_dot_dimension_numbers")
make_gpu_client: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_gpu_client")
make_iota: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_iota")
make_jaxpr_effects: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_jaxpr_effects")
make_padding_config: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_padding_config")
make_replica_groups: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_replica_groups")
make_tfrt_tpu_c_api_client: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_tfrt_tpu_c_api_client")
make_tfrt_tpu_c_api_device_topology: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_tfrt_tpu_c_api_device_topology")
make_tpu_client: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("make_tpu_client")
map_bind: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("map_bind")
map_bind_with_continuation: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("map_bind_with_continuation")
map_transpose: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("map_transpose")
mapped_aval: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("mapped_aval")
matchaxis: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("matchaxis")
max_dim: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("max_dim")
maybe_find_leaked_tracers: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("maybe_find_leaked_tracers")
maybe_named_axis: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("maybe_named_axis")
median: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("median")
memoize: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("memoize")
merge_lists: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("merge_lists")
min_dim: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("min_dim")
mode: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("mode")
modf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("modf")
move_binders_to_back: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("move_binders_to_back")
move_binders_to_front: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("move_binders_to_front")
multigammaln: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("multigammaln")
ndtr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ndtr")
new_base_main: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("new_base_main")
new_eqn_recipe: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("new_eqn_recipe")
new_jaxpr_eqn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("new_jaxpr_eqn")
new_main: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("new_main")
new_sublevel: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("new_sublevel")
nonzero_outputs: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("nonzero_outputs")
nonzero_tangent_outputs: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("nonzero_tangent_outputs")
numpy_dtype_promotion: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("numpy_dtype_promotion")
numpy_rank_promotion: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("numpy_rank_promotion")
ogrid: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ogrid")
op_sharding_to_indices: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("op_sharding_to_indices")
overload: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("overload")
pad_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pad_jaxpr")
parallel_callable: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("parallel_callable")
parse_flags_with_absl: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("parse_flags_with_absl")
partial_eval_jaxpr_custom: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("partial_eval_jaxpr_custom")
partial_eval_jaxpr_custom_rule_not_implemented: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("partial_eval_jaxpr_custom_rule_not_implemented")
partial_eval_jaxpr_nounits: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("partial_eval_jaxpr_nounits")
partial_eval_wrapper_nounits: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("partial_eval_wrapper_nounits")
partition_list: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("partition_list")
partition_pvals: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("partition_pvals")
pbroadcast: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pbroadcast")
pdf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pdf")
persistent_cache_min_compile_time_secs: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("persistent_cache_min_compile_time_secs")
persistent_cache_min_entry_size_bytes: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("persistent_cache_min_entry_size_bytes")
pgle_aggregation_percentile: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pgle_aggregation_percentile")
pgle_profiling_runs: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pgle_profiling_runs")
piecewise: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("piecewise")
pjrt_plugin_initialized: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pjrt_plugin_initialized")
pjrt_plugin_loaded: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pjrt_plugin_loaded")
pmap_no_rank_reduction: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pmap_no_rank_reduction")
pmap_shmap_merge: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pmap_shmap_merge")
pmax: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pmax")
pmf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pmf")
pmin: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pmin")
poch: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("poch")
ppermute: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ppermute")
ppf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("ppf")
primal_dtype_to_tangent_dtype: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("primal_dtype_to_tangent_dtype")
primitive_uses_outfeed: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("primitive_uses_outfeed")
print: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("print")
process_env_traces_call: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("process_env_traces_call")
process_env_traces_map: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("process_env_traces_map")
pshuffle: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pshuffle")
pswapaxes: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("pswapaxes")
r_: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("r_")
rademacher: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("rademacher")
raise_as_much_as_possible: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("raise_as_much_as_possible")
raise_persistent_cache_errors: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("raise_persistent_cache_errors")
raise_to_shaped: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("raise_to_shaped")
random_seed_offset: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("random_seed_offset")
rankdata: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("rankdata")
rearrange_binders: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("rearrange_binders")
rebase_donate_argnums: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("rebase_donate_argnums")
recast_to_float0: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("recast_to_float0")
recipe_to_eqn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("recipe_to_eqn")
record_event: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("record_event")
record_event_duration_secs: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("record_event_duration_secs")
reducer_batcher: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("reducer_batcher")
register_custom_call_handler: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("register_custom_call_handler")
register_custom_call_target: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("register_custom_call_target")
register_event_duration_secs_listener: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("register_event_duration_secs_listener")
register_event_listener: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("register_event_listener")
register_exclusion: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("register_exclusion")
register_vmappable: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("register_vmappable")
rel_entr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("rel_entr")
remat_opt_barrier: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("remat_opt_barrier")
replace_float0s: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("replace_float0s")
reset_cache: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("reset_cache")
reset_trace_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("reset_trace_state")
result_info: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("result_info")
rot90: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("rot90")
rsf2csf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("rsf2csf")
safe_map: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("safe_map")
safe_zip: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("safe_zip")
save_device_memory_profile: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("save_device_memory_profile")
scalar_type_of: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("scalar_type_of")
scale_and_translate: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("scale_and_translate")
sem: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("sem")
sequential_vmap: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("sequential_vmap")
set_cache_dir: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("set_cache_dir")
set_module: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("set_module")
set_thread_local_state_initialization_callback: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("set_thread_local_state_initialization_callback")
setdiff1d: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("setdiff1d")
setxor1d: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("setxor1d")
sf: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("sf")
shape_from_pyval: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("shape_from_pyval")
shaped_abstractify: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("shaped_abstractify")
shard_args: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("shard_args")
share_autotune_config_between_hosts: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("share_autotune_config_between_hosts")
share_binary_between_hosts: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("share_binary_between_hosts")
share_binary_between_hosts_timeout_ms: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("share_binary_between_hosts_timeout_ms")
shutdown: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("shutdown")
sig_info: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("sig_info")
softmax_custom_jvp: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("softmax_custom_jvp")
solve_triangular: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("solve_triangular")
spec_to_indices: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("spec_to_indices")
sph_harm: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("sph_harm")
split_dict: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("split_dict")
split_list: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("split_list")
split_list_checked: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("split_list_checked")
split_merge: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("split_merge")
spmd_mode: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("spmd_mode")
squareplus: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("squareplus")
stable_unique: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("stable_unique")
standard_jvp: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("standard_jvp")
standard_jvp2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("standard_jvp2")
start_server: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("start_server")
start_trace: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("start_trace")
stash_axis_env: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("stash_axis_env")
stop_server: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("stop_server")
stop_trace: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("stop_trace")
str_eqn_compact: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("str_eqn_compact")
subjaxprs: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("subjaxprs")
subs_list: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("subs_list")
subs_list2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("subs_list2")
subst_axis_names: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("subst_axis_names")
subst_axis_names_eqn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("subst_axis_names_eqn")
subst_axis_names_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("subst_axis_names_jaxpr")
subst_axis_names_var: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("subst_axis_names_var")
substitute_vars_in_output_ty: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("substitute_vars_in_output_ty")
subvals: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("subvals")
swap_thread_local_state_disable_jit: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("swap_thread_local_state_disable_jit")
thread_local_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("thread_local_state")
threefry_gpu_kernel_lowering: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("threefry_gpu_kernel_lowering")
threefry_partitionable: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("threefry_partitionable")
to_elt: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("to_elt")
toeplitz: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("toeplitz")
toposort: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("toposort")
trace_context: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_context")
trace_state_clean: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_state_clean")
trace_to_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_jaxpr")
trace_to_jaxpr_dynamic: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_jaxpr_dynamic")
trace_to_jaxpr_dynamic2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_jaxpr_dynamic2")
trace_to_jaxpr_final: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_jaxpr_final")
trace_to_jaxpr_final2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_jaxpr_final2")
trace_to_jaxpr_nounits: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_jaxpr_nounits")
trace_to_subjaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_subjaxpr")
trace_to_subjaxpr_dynamic: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_subjaxpr_dynamic")
trace_to_subjaxpr_dynamic2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_subjaxpr_dynamic2")
trace_to_subjaxpr_nounits: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_subjaxpr_nounits")
trace_to_subjaxpr_nounits_dyn: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_subjaxpr_nounits_dyn")
trace_to_subjaxpr_nounits_fwd: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trace_to_subjaxpr_nounits_fwd")
traceable: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("traceable")
traceback_filtering: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("traceback_filtering")
traceback_in_locations_limit: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("traceback_in_locations_limit")
tracebacks: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("tracebacks")
tracers_to_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("tracers_to_jaxpr")
transfer_guard: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("transfer_guard")
transfer_guard_device_to_device: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("transfer_guard_device_to_device")
transfer_guard_device_to_host: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("transfer_guard_device_to_host")
transfer_guard_host_to_device: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("transfer_guard_host_to_device")
trapezoid: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trapezoid")
traverse_jaxpr_params: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("traverse_jaxpr_params")
tri: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("tri")
trivial_ctx: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("trivial_ctx")
tuple_delete: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("tuple_delete")
tuple_insert: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("tuple_insert")
tuple_update: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("tuple_update")
typecheck: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("typecheck")
typecompat: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("typecompat")
typematch: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("typematch")
unflatten: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("unflatten")
union1d: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("union1d")
unmapped_aval: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("unmapped_aval")
unpair_pval: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("unpair_pval")
unregister_vmappable: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("unregister_vmappable")
unwrap: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("unwrap")
unzip2: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("unzip2")
unzip3: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("unzip3")
update: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("update")
update_thread_local_jit_state: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("update_thread_local_jit_state")
use_cpp_class: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("use_cpp_class")
use_cpp_method: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("use_cpp_method")
used_axis_names: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("used_axis_names")
used_axis_names_jaxpr: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("used_axis_names_jaxpr")
valid_jaxtype: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("valid_jaxtype")
vander: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("vander")
vectorize: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("vectorize")
vectorized_batcher: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("vectorized_batcher")
visualize_array_sharding: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("visualize_array_sharding")
visualize_sharding: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("visualize_sharding")
vjp: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("vjp")
vq: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("vq")
vtile: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("vtile")
weakref_lru_cache: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("weakref_lru_cache")
window_padding_type_to_pad_values: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("window_padding_type_to_pad_values")
wrap_name: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("wrap_name")
wraps: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("wraps")
xla_runtime_errors: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("xla_runtime_errors")
zero_jvp: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("zero_jvp")
zeros_like_aval: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("zeros_like_aval")
zeros_like_jaxval: VariadicCallable[TensorOrScalar] = _make_unimplemented_op("zeros_like_jaxval")

__all__ = [
    "AbstractedAxesSpec",
    "ArrayLike",
    "Atom",
    "AxisSize",
    "BatchingRule",
    "Const",
    "ConstFoldRule",
    "CurrentSourceInfoMetadata",
    "DCERule",
    "DEFINE_bool",
    "DEFINE_enum",
    "DEFINE_float",
    "DEFINE_integer",
    "DEFINE_string",
    "DTypeLike",
    "Elt",
    "ForwardingRule",
    "FromEltHandler",
    "GetIdx",
    "Index",
    "JaxprTracerRecipe",
    "LoadedExecutable_execute",
    "LoadedExecutable_execute_with_token",
    "MakeIotaHandler",
    "MapSpec",
    "MeshAxisName",
    "NamedTuple",
    "NoReturn",
    "ParamsUpdater",
    "PartialEvalCustomRule",
    "ResAvalUpdater",
    "ToEltHandler",
    "Transpose",
    "Vmappable",
    "abs",
    "abstract_eval_fun",
    "acos",
    "acosh",
    "add",
    "add_jaxvals",
    "add_tangents",
    "all",
    "all_gather",
    "all_to_all",
    "allclose",
    "angle",
    "annotate_function",
    "any",
    "api_boundary",
    "append",
    "apply_over_axes",
    "apply_todos",
    "approx_max_k",
    "approx_min_k",
    "arange",
    "are_op_shardings_equal",
    "arg_info_all",
    "argmax",
    "argmin",
    "argnums_partial",
    "argpartition",
    "argsort",
    "argwhere",
    "array",
    "array_equiv",
    "array_mapping_to_axis_resources",
    "array_repr",
    "array_str",
    "as_hashable_function",
    "as_named_shape",
    "asin",
    "asinh",
    "assert_unreachable",
    "associative_scan",
    "atan",
    "atan2",
    "atanh",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "average",
    "axis_frame",
    "axis_index",
    "backward_pass",
    "backward_pass_internal",
    "ball",
    "bartlett",
    "batch",
    "batch_custom_jvp_subtrace",
    "batch_custom_vjp_bwd",
    "batch_jaxpr",
    "batch_jaxpr2",
    "batch_jaxpr_axes",
    "batch_subtrace",
    "bcoo_cusparse_lowering",
    "bdim_at_front",
    "bernoulli",
    "bessel_i0e",
    "bessel_i1e",
    "bessel_jn",
    "beta",
    "betainc",
    "betaln",
    "bicgstab",
    "bilinear_transpose",
    "bincount",
    "binomial",
    "bitcast_convert_type",
    "bits",
    "bitwise_and",
    "bitwise_count",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "blackman",
    "block",
    "block_diag",
    "bool_env",
    "breakpoint",
    "broadcast",
    "broadcast_arrays",
    "broadcast_batcher",
    "broadcast_to",
    "broadcast_to_rank",
    "broadcasted_iota",
    "c_",
    "cache",
    "call",
    "call_bind_with_continuation",
    "call_impl",
    "call_padding_rule",
    "call_partial_eval_custom_rule",
    "call_transpose",
    "callback",
    "can_cast",
    "canonicalize_axis",
    "canonicalize_dtype",
    "cast",
    "cauchy",
    "cbrt",
    "cdf",
    "ceil",
    "ceil_of_ratio",
    "cg",
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "check_eqn",
    "check_exists",
    "check_jaxpr",
    "check_toposort",
    "check_tracer_leaks",
    "check_type",
    "check_valid_jaxtype",
    "checking_leaks",
    "chisquare",
    "cho_factor",
    "cho_solve",
    "cholesky",
    "cholesky_solve",
    "choose",
    "clear_all_caches",
    "clear_all_weakref_lru_caches",
    "clear_event_listeners",
    "clone",
    "close_jaxpr",
    "closed_backward_pass",
    "closed_call_partial_eval_custom_rule",
    "closure_convert",
    "clz",
    "collapse",
    "column_stack",
    "compilation_cache_dir",
    "compilation_cache_include_metadata_in_key",
    "compilation_cache_max_size",
    "compress",
    "concatenate",
    "concrete_aval",
    "concrete_or_error",
    "concretization_function_error",
    "cond",
    "confusion_matrix",
    "conj",
    "contextmanager",
    "conv_general_dilated",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_transpose",
    "conv_transpose_shape_tuple",
    "conv_with_general_padding",
    "convert_constvars_jaxpr",
    "convert_envvars_to_constvars",
    "convert_invars_to_constvars",
    "convolve",
    "convolve2d",
    "copysign",
    "corrcoef",
    "correlate",
    "correlate2d",
    "cos",
    "cosh",
    "count_nonzero",
    "cov",
    "create_token",
    "cross",
    "csd",
    "cumlogsumexp",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "cur_sublevel",
    "curry",
    "custom_gradient",
    "custom_linear_solve",
    "custom_root",
    "custom_vjp_disable_shape_check",
    "custom_vjp_primal_tree_values",
    "dce_jaxpr",
    "dce_jaxpr_call_rule",
    "dce_jaxpr_closed_call_rule",
    "dce_jaxpr_consts",
    "dct",
    "dctn",
    "debug_info",
    "debug_info_final",
    "debug_infs",
    "debug_key_reuse",
    "debug_nans",
    "decode_csv",
    "decode_image",
    "dedup_referents",
    "def_trivial_padding",
    "default_backend",
    "default_device",
    "default_dtype_bits",
    "default_matmul_precision",
    "default_prng_impl",
    "defbilinear",
    "defbroadcasting",
    "define_bool_state",
    "define_enum_state",
    "define_float_state",
    "define_int_state",
    "define_optional_enum_state",
    "define_optional_string_state",
    "define_string_or_object_state",
    "define_string_state",
    "defjvp",
    "defjvp2",
    "defjvp_zero",
    "deflinear",
    "deflinear2",
    "defreducer",
    "defvectorized",
    "deg2rad",
    "delete",
    "descriptive",
    "det",
    "detrend",
    "device_memory_profile",
    "diag",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "diagonal",
    "diff",
    "digamma",
    "digitize",
    "dirichlet",
    "disable_all_debug_logging",
    "disable_jit",
    "disable_x64",
    "distributed_debug",
    "distributed_debug_log",
    "distributions",
    "divide",
    "divmod",
    "do_subst_axis_names_jaxpr",
    "donation_vector",
    "dot",
    "dot_general",
    "double_sided_maxwell",
    "dsplit",
    "dstack",
    "dtype_to_etype",
    "dynamic_index_in_dim",
    "dynamic_shapes",
    "dynamic_slice",
    "dynamic_slice_in_dim",
    "dynamic_update_index_in_dim",
    "dynamic_update_slice_in_dim",
    "eager_pmap",
    "ediff1d",
    "eig",
    "eigh",
    "eigh_tridiagonal",
    "eigvals",
    "eigvalsh",
    "einsum",
    "einsum_path",
    "empty",
    "enable_checks",
    "enable_compilation_cache",
    "enable_custom_prng",
    "enable_custom_vjp_by_custom_transpose",
    "enable_debug_logging",
    "enable_memories",
    "enable_pgle",
    "enable_x64",
    "ensure_compile_time_eval",
    "entr",
    "equal",
    "erf",
    "erf_inv",
    "erfc",
    "erfinv",
    "escaped_tracer_error",
    "eval_context",
    "eval_jaxpr",
    "execute_with_python_values",
    "execute_with_python_values_replicated",
    "exp",
    "exp1",
    "exp2",
    "expand_dims",
    "expi",
    "expit",
    "explain_cache_misses",
    "explicit_device_get_scope",
    "explicit_device_put_scope",
    "expm",
    "expm1",
    "expm_frechet",
    "expn",
    "exponential",
    "extend_axis_env",
    "extend_axis_env_nd",
    "extend_jaxpr_stack",
    "extract",
    "eye",
    "f",
    "f_jvp_traceable",
    "fabs",
    "factorial",
    "fft",
    "fft2",
    "fftconvolve",
    "fftfreq",
    "fftn",
    "fftshift",
    "fill_diagonal",
    "filter_traceback",
    "find_top_trace",
    "finfo",
    "fix",
    "flatnonzero",
    "flatten",
    "flatten_axes",
    "flatten_fun",
    "flatten_fun_for_vmap",
    "flatten_fun_nokwargs",
    "flip",
    "fliplr",
    "flipud",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "fori_loop",
    "format_exception_only",
    "frexp",
    "from_dlpack",
    "from_elt",
    "fromfile",
    "fromfunction",
    "fromiter",
    "frompyfunc",
    "fromstring",
    "full",
    "full_like",
    "full_lower",
    "fun_name",
    "funm",
    "gamma",
    "gammainc",
    "gammaincc",
    "gammaln",
    "gammasgn",
    "gather",
    "gather_nd",
    "gcd",
    "generalized_normal",
    "generate_pjrt_gpu_plugin_options",
    "gensym",
    "geometric",
    "geomspace",
    "get_aval",
    "get_backend",
    "get_compile_options",
    "get_enable_x64",
    "get_primitive_transpose",
    "get_printoptions",
    "get_referent",
    "global_aval_to_result_handler",
    "global_avals_to_results_handler",
    "global_state",
    "gmres",
    "gradient",
    "greater",
    "greater_equal",
    "gumbel",
    "hamming",
    "hanning",
    "hard_silu",
    "hard_swish",
    "heap_profile",
    "heaviside",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "hessenberg",
    "hfft",
    "hilbert",
    "histogram",
    "histogram2d",
    "histogram_bin_edges",
    "histogramdd",
    "hlo_source_file_canonicalization_regex",
    "householder_product",
    "hsplit",
    "hstack",
    "hyp1f1",
    "hypot",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "idct",
    "idctn",
    "identity",
    "ifft",
    "ifft2",
    "ifftn",
    "ifftshift",
    "igamma",
    "igamma_grad_a",
    "igammac",
    "ihfft",
    "iinfo",
    "imag",
    "include_frame",
    "include_full_tracebacks_in_locations",
    "index_in_dim",
    "indices",
    "infeed",
    "infer_lambda_input_type",
    "initialize",
    "initialize_cache",
    "initialize_pjrt_plugin",
    "inner",
    "insert",
    "inspect_array_sharding",
    "instantiate_const_at",
    "instantiate_zeros",
    "int_env",
    "interp",
    "intersect1d",
    "inv",
    "io_callback",
    "irfft",
    "irfft2",
    "irfftn",
    "is_constant_dim",
    "is_constant_shape",
    "is_initialized",
    "is_op_sharding_replicated",
    "is_undefined_primal",
    "is_vmappable",
    "isclose",
    "iscomplex",
    "iscomplexobj",
    "isf",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isreal",
    "isrealobj",
    "isscalar",
    "issubdtype",
    "istft",
    "iterable",
    "ix_",
    "jax2tf_associative_scan_reductions",
    "jax2tf_default_native_serialization",
    "jax_export_calling_convention_version",
    "jax_pjrt_client_create_options",
    "jax_platforms",
    "jax_serialization_version",
    "jax_xla_profile_version",
    "jaxpr_as_fun",
    "jaxpr_uses_outfeed",
    "jaxprs_in_params",
    "join_effects",
    "join_named_shapes",
    "jvp",
    "jvp_jaxpr",
    "jvp_subtrace",
    "jvp_subtrace_aux",
    "jvpfun",
    "kaiser",
    "key",
    "key_data",
    "key_impl",
    "kl_div",
    "kron",
    "laguerre_polynomial_l",
    "laplace",
    "lattice_join",
    "lcm",
    "ldexp",
    "leaked_tracer_error",
    "left_shift",
    "legacy_prng_key",
    "legendre_polynomial_p",
    "less",
    "less_equal",
    "lexsort",
    "lgamma",
    "linear_call",
    "linear_jvp",
    "linear_transpose",
    "linear_transpose2",
    "linearize",
    "load",
    "load_pjrt_plugin_dynamically",
    "load_pjrt_plugin_with_c_api",
    "log",
    "log1p",
    "log_checkpoint_residuals",
    "log_compiles",
    "log_ndtr",
    "logaddexp",
    "logaddexp2",
    "logcdf",
    "loggamma",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logistic",
    "logit",
    "lognormal",
    "logpdf",
    "logpmf",
    "logsf",
    "logsumexp",
    "lpmn",
    "lpmn_values",
    "lstsq",
    "lu",
    "lu_factor",
    "lu_matrix_inverse",
    "lu_pivots_to_permutation",
    "lu_reconstruct",
    "lu_solve",
    "make_c_api_client",
    "make_c_api_device_topology",
    "make_convolution_dimension_numbers",
    "make_cpu_client",
    "make_dot_dimension_numbers",
    "make_gpu_client",
    "make_iota",
    "make_jaxpr_effects",
    "make_padding_config",
    "make_replica_groups",
    "make_tfrt_tpu_c_api_client",
    "make_tfrt_tpu_c_api_device_topology",
    "make_tpu_client",
    "map_bind",
    "map_bind_with_continuation",
    "map_coordinates",
    "map_transpose",
    "mapped_aval",
    "mask_indices",
    "matchaxis",
    "matmul",
    "matrix_norm",
    "matrix_power",
    "matrix_rank",
    "matrix_transpose",
    "max",
    "max_dim",
    "maximum",
    "maxwell",
    "maybe_find_leaked_tracers",
    "maybe_named_axis",
    "mean",
    "median",
    "memoize",
    "merge_lists",
    "mgrid",
    "min",
    "min_dim",
    "minimum",
    "mish",
    "mod",
    "mode",
    "modf",
    "modified_bessel_i0",
    "modified_bessel_i1",
    "modified_bessel_k0",
    "modified_bessel_k1",
    "move_binders_to_back",
    "move_binders_to_front",
    "moveaxis",
    "multi_dot",
    "multigammaln",
    "multiply",
    "multivariate_normal",
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
    "ndtr",
    "ndtri",
    "negative",
    "new_base_main",
    "new_eqn_recipe",
    "new_jaxpr_eqn",
    "new_main",
    "new_sublevel",
    "nextafter",
    "nonzero",
    "nonzero_outputs",
    "nonzero_tangent_outputs",
    "norm",
    "not_equal",
    "numpy_dtype_promotion",
    "numpy_rank_promotion",
    "ogrid",
    "ones",
    "ones_like",
    "op_sharding_to_indices",
    "orthogonal",
    "outer",
    "outfeed",
    "overload",
    "packbits",
    "pad",
    "pad_jaxpr",
    "parallel_callable",
    "pareto",
    "parse_example",
    "parse_flags_with_absl",
    "parse_tensor",
    "partial_eval_jaxpr_custom",
    "partial_eval_jaxpr_custom_rule_not_implemented",
    "partial_eval_jaxpr_nounits",
    "partial_eval_wrapper_nounits",
    "partition_list",
    "partition_pvals",
    "pbroadcast",
    "pdf",
    "pdot",
    "percentile",
    "permute",
    "persistent_cache_min_compile_time_secs",
    "persistent_cache_min_entry_size_bytes",
    "pgle_aggregation_percentile",
    "pgle_profiling_runs",
    "piecewise",
    "pinv",
    "pjrt_plugin_initialized",
    "pjrt_plugin_loaded",
    "pmap_no_rank_reduction",
    "pmap_shmap_merge",
    "pmax",
    "pmean",
    "pmf",
    "pmin",
    "poch",
    "poisson",
    "polar",
    "poly",
    "polyadd",
    "polyder",
    "polydiv",
    "polyfit",
    "polygamma",
    "polyint",
    "polymul",
    "polysub",
    "polyval",
    "population_count",
    "positive",
    "power",
    "ppermute",
    "ppf",
    "primal_dtype_to_tangent_dtype",
    "primitive_uses_outfeed",
    "print",
    "process_env_traces_call",
    "process_env_traces_map",
    "prod",
    "promote_types",
    "pshuffle",
    "psum",
    "psum_scatter",
    "pswapaxes",
    "qdwh",
    "qr",
    "quantile",
    "r_",
    "rad2deg",
    "rademacher",
    "ragged_dot",
    "raise_as_much_as_possible",
    "raise_persistent_cache_errors",
    "raise_to_shaped",
    "random_gamma_grad",
    "random_seed_offset",
    "rankdata",
    "ravel_multi_index",
    "rayleigh",
    "read_file",
    "real",
    "rearrange_binders",
    "rebase_donate_argnums",
    "recast_to_float0",
    "recipe_to_eqn",
    "reciprocal",
    "record_event",
    "record_event_duration_secs",
    "reduce_precision",
    "reduce_window",
    "reducer_batcher",
    "register_custom_call_handler",
    "register_custom_call_target",
    "register_event_duration_secs_listener",
    "register_event_listener",
    "register_exclusion",
    "register_vmappable",
    "rel_entr",
    "rem",
    "remainder",
    "remat_opt_barrier",
    "repeat",
    "replace_float0s",
    "reset_cache",
    "reset_trace_state",
    "reshape",
    "resize",
    "result_info",
    "result_type",
    "rfft",
    "rfft2",
    "rfftfreq",
    "rfftn",
    "right_shift",
    "roll",
    "roots",
    "rot90",
    "round",
    "rrelu",
    "rsf2csf",
    "rsqrt",
    "safe_map",
    "safe_zip",
    "save",
    "save_device_memory_profile",
    "savez",
    "scalar_type_of",
    "scale_and_translate",
    "scatter",
    "scatter_add",
    "scatter_apply",
    "scatter_max",
    "scatter_min",
    "scatter_mul",
    "scatter_nd",
    "schur",
    "searchsorted",
    "segment_max",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "sem",
    "sequential_vmap",
    "serialize_tensor",
    "set_cache_dir",
    "set_module",
    "set_thread_local_state_initialization_callback",
    "setdiff1d",
    "setxor1d",
    "sf",
    "shape",
    "shape_from_pyval",
    "shaped_abstractify",
    "shard_args",
    "share_autotune_config_between_hosts",
    "share_binary_between_hosts",
    "share_binary_between_hosts_timeout_ms",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
    "shutdown",
    "sig_info",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "size",
    "slice_in_dim",
    "slogdet",
    "softmax_custom_jvp",
    "solve",
    "solve_triangular",
    "sort",
    "sort_complex",
    "sort_key_val",
    "sparse_plus",
    "sparse_sigmoid",
    "spec_to_indices",
    "spence",
    "sph_harm",
    "split",
    "split_dict",
    "split_list",
    "split_list_checked",
    "split_merge",
    "spmd_mode",
    "sqrt",
    "sqrtm",
    "square",
    "squareplus",
    "squeeze",
    "stable_unique",
    "stack",
    "standard_jvp",
    "standard_jvp2",
    "start_server",
    "start_trace",
    "stash_axis_env",
    "stft",
    "stop_server",
    "stop_trace",
    "str_eqn_compact",
    "strided_slice",
    "subjaxprs",
    "subs_list",
    "subs_list2",
    "subst_axis_names",
    "subst_axis_names_eqn",
    "subst_axis_names_jaxpr",
    "subst_axis_names_var",
    "substitute_vars_in_output_ty",
    "subtract",
    "subvals",
    "sum",
    "svd",
    "svdvals",
    "swap_thread_local_state_disable_jit",
    "swapaxes",
    "t",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tensordot",
    "tensorinv",
    "tensorsolve",
    "thread_local_state",
    "threefry_gpu_kernel_lowering",
    "threefry_partitionable",
    "tile",
    "to_dlpack",
    "to_elt",
    "toeplitz",
    "top_k",
    "toposort",
    "trace",
    "trace_context",
    "trace_state_clean",
    "trace_to_jaxpr",
    "trace_to_jaxpr_dynamic",
    "trace_to_jaxpr_dynamic2",
    "trace_to_jaxpr_final",
    "trace_to_jaxpr_final2",
    "trace_to_jaxpr_nounits",
    "trace_to_subjaxpr",
    "trace_to_subjaxpr_dynamic",
    "trace_to_subjaxpr_dynamic2",
    "trace_to_subjaxpr_nounits",
    "trace_to_subjaxpr_nounits_dyn",
    "trace_to_subjaxpr_nounits_fwd",
    "traceable",
    "traceback_filtering",
    "traceback_in_locations_limit",
    "tracebacks",
    "tracers_to_jaxpr",
    "transfer_guard",
    "transfer_guard_device_to_device",
    "transfer_guard_device_to_host",
    "transfer_guard_host_to_device",
    "transpose",
    "trapezoid",
    "traverse_jaxpr_params",
    "tri",
    "triangular",
    "triangular_solve",
    "tridiagonal",
    "tridiagonal_solve",
    "tril",
    "tril_indices",
    "tril_indices_from",
    "trim_zeros",
    "triu",
    "triu_indices",
    "triu_indices_from",
    "trivial_ctx",
    "trunc",
    "tuple_delete",
    "tuple_insert",
    "tuple_update",
    "typecheck",
    "typecompat",
    "typematch",
    "unflatten",
    "union1d",
    "unique",
    "unmapped_aval",
    "unpackbits",
    "unpair_pval",
    "unregister_vmappable",
    "unstack",
    "unwrap",
    "unzip2",
    "unzip3",
    "update",
    "update_slice",
    "update_thread_local_jit_state",
    "use_cpp_class",
    "use_cpp_method",
    "used_axis_names",
    "used_axis_names_jaxpr",
    "valid_jaxtype",
    "vander",
    "vdot",
    "vecdot",
    "vector_norm",
    "vectorize",
    "vectorized_batcher",
    "visualize_array_sharding",
    "visualize_sharding",
    "vjp",
    "vq",
    "vsplit",
    "vstack",
    "vtile",
    "wald",
    "weakref_lru_cache",
    "weibull_min",
    "welch",
    "where",
    "while_loop",
    "window_padding_type_to_pad_values",
    "wrap_key_data",
    "wrap_name",
    "wraps",
    "write_file",
    "xla_runtime_errors",
    "xlog1py",
    "xlogy",
    "zero_jvp",
    "zeros",
    "zeros_like",
    "zeros_like_aval",
    "zeros_like_jaxval",
    "zeta",
]
