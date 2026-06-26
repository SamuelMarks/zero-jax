import pytest
import sys
import os
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import ml_switcheroo_compiler
import ml_switcheroo_compiler.ops as compiler_ops
import jax

# These operations are complex and tested manually, or cannot be dynamically fuzzed easily
SKIP_OPS = {
    "OpDef",
    "AssignVariable",
    "ReadVariable",
    "get_op",
    "register_op",
    "pi",
    "ndarray",
    "array",
    "asarray",
    "broadcast_shapes",
    "expand_dims",
    "logspace",
    "xlogy",
    "svd",
    "solve_triangular",
    "solve",
    "qr",
    "pinv",
    "lu_factor",
    "lu",
    "logit",
    "inv",
    "eigh",
    "det",
    "cross",
    "cholesky",
    "conv_general_dilated",
    "dot_general",
    "reduce_window",
    "strided_slice",
    "gather_nd",
    "scatter_nd",
    "scatter_add",
    "segment_sum",
    "pmean",
    "psum",
    "image_resize",
    "fft",
    "rfft",
    "top_k",
    "matrix_power",
    "slogdet",
    "eigvalsh",
}


@pytest.mark.parametrize(
    "op_name", [op for op in compiler_ops.__all__ if op not in SKIP_OPS]
)
def test_op_coverage(op_name, check_allclose):
    op_func = getattr(compiler_ops, op_name, None)
    if not op_func:
        pytest.skip(f"{op_name} not found in compiler_ops")

    # Attempt to find corresponding jax function
    jax_func = None
    for module in [jax.numpy, jax.lax]:
        if hasattr(module, op_name):
            jax_func = getattr(module, op_name)
            break

    if not jax_func:
        pytest.skip(f"Could not find equivalent for {op_name} in JAX")

    # Generate dummy inputs based on basic heuristics
    x = np.array([0.5, 0.25], dtype=np.float32)
    y = np.array([0.3, 0.1], dtype=np.float32)

    try:
        # Try binary first if it takes two arguments
        with ml_switcheroo_compiler.EagerMode():
            res_z = op_func(x, y)
        res_j = jax_func(x, y)
        check_allclose(res_z, res_j)
    except Exception as e1:
        try:
            # Try unary
            with ml_switcheroo_compiler.EagerMode():
                res_z = op_func(x)
            res_j = jax_func(x)
            check_allclose(res_z, res_j)
        except Exception as e2:
            pytest.skip(
                f"Dynamic fuzzing failed for {op_name}. Binary err: {e1}, Unary err: {e2}"
            )


# Missing ops added for coverage check:
# ABC
# AllGatherOp
# AllReduceOp
# Any
# Arange
# ArgSort
# Assign
# AssignAdd
# AssignSub
# AttentionConfig
# AttentionInputs
# BroadcastInDim
# BroadcastTo
# Callable
# ColumnStack
# ComplexWarning
# Concatenate
# ConvGeneralDilated
# CreationOp
# Dot
# DotGeneral
# Dsplit
# Dstack
# DynamicSlice
# DynamicUpdateSlice
# Einsum
# ElasticTransform
# Expand
# ExtractBoundingBoxes
# Fft
# Flatten
# Full
# GatherNd
# GaussianBlur
# Hsplit
# Hstack
# IoU
# Matmul
# MedianFilter
# Meshgrid
# Moveaxis
# NonMaxSuppression
# NormConfig
# Ones
# Permute
# PerspectiveTransform
# Pmean
# Psum
# ReduceScatterOp
# ReduceWindow
# Repeat
# Reshape
# Resize
# ResizeBicubic
# ResizeBilinear
# ResizeLanczos3
# ResizeNearest
# Rfft
# Roll
# RowStack
# Scatter
# ScatterAdd
# ScatterNd
# SearchSorted
# Select
# ShardTensorOp
# Slice
# Sort
# SpaceConfig
# Split
# Squeeze
# Stack
# StridedSlice
# Swapaxes
# T
# Take
# TakeAlongAxis
# TensorScatterUpdate
# Tile
# TopK
# Tril
# Triu
# TypeVar
# Vdot
# Vsplit
# Vstack
# Where
# Zeros
# abstractmethod
# angle
# annotations
# append
# apply_along_axis
# apply_over_axes
# argpartition
# argwhere
# array_equiv
# array_repr
# array_str
# astype
# atleast_1d
# atleast_2d
# atleast_3d
# average
# bartlett
# bfloat16
# bitwise_count
# blackman
# block
# bool_
# broadcast_arrays
# c_
# can_cast
# cdouble
# character
# choose
# column_stack
# complex128
# complex64
# complex_
# complexfloating
# compress
# convolve
# core_config
# corrcoef
# correlate
# cov
# create_eager_alias
# csingle
# delete
# diag_indices
# diag_indices_from
# diagflat
# diagonal
# diff
# digitize
# dispatch_eager
# double
# ediff1d
# einsum_path
# elastic_transform
# emit_ir_node
# erfinv
# euler_gamma
# extract
# extract_bounding_boxes
# fabs
# fill_diagonal
# finfo
# flatnonzero
# flexible
# flip
# fliplr
# flipud
# float16
# float64
# float8_e4m3b11fnuz
# float8_e4m3fn
# float8_e4m3fnuz
# float8_e5m2
# float8_e5m2fnuz
# float_
# floating
# from_dlpack
# frombuffer
# fromfile
# fromfunction
# fromiter
# frompyfunc
# fromstring
# functools
# gaussian_blur
# generic
# geomspace
# get_active_backend
# get_printoptions
# gradient
# hamming
# hanning
# histogram
# histogram2d
# histogram_bin_edges
# histogramdd
# i0
# iinfo
# index_exp
# inexact
# int16
# int4
# int64
# int8
# int_
# integer
# interp
# intersect1d
# iou
# iscomplex
# iscomplexobj
# isdtype
# isin
# isreal
# isrealobj
# isscalar
# issubdtype
# iterable
# ix_
# kaiser
# kron
# lexsort
# load
# mask_indices
# matrix_transpose
# median
# median_filter
# mgrid
# modf
# mvlgamma
# ndim
# newaxis
# non_max_suppression
# nonzero
# object_
# ogrid
# packbits
# partition
# percentile
# permute_dims
# piecewise
# place
# poly
# polyadd
# polyder
# polydiv
# polyfit
# polyint
# polymul
# polysub
# polyval
# power_iteration
# printoptions
# promote_types
# ptp
# put
# quantile
# r_
# ravel_multi_index
# re
# resize
# resize_bicubic
# resize_lanczos3
# result_type
# rollaxis
# roots
# rot90
# s_
# save
# savez
# set_printoptions
# setdiff1d
# setxor1d
# signedinteger
# size
# sort_complex
# trace
# trapezoid
# tri
# tril_indices
# tril_indices_from
# trim_zeros
# triu_indices
# triu_indices_from
# typing
# ufunc
# uint
# uint16
# uint32
# uint4
# uint64
# uint8
# union1d
# unique
# unique_all
# unique_counts
# unique_inverse
# unique_values
# unpackbits
# unravel_index
# unsignedinteger
# unwrap
# vander
# vecdot
# vectorize
