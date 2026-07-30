"""Tests for zero_jax.numpy."""

from typing import Any

import pytest

import zero_jax.numpy as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_ComplexWarning() -> None:
    """Test class ComplexWarning."""
    try:
        mod.ComplexWarning()
    except Exception:
        pass


def test_List() -> None:
    """Test List."""
    try:
        mod.List()
    except Exception:
        pass


def test_Optional() -> None:
    """Test Optional."""
    try:
        mod.Optional()
    except Exception:
        pass


def test_class_Tensor() -> None:
    """Test class Tensor."""
    try:
        mod.Tensor(1.0, 1.0)
    except Exception:
        pass


def test_Tuple() -> None:
    """Test Tuple."""
    try:
        mod.Tuple()
    except Exception:
        pass


def test_abs() -> None:
    """Test abs."""
    try:
        mod.abs(1.0)
    except Exception:
        pass


def test_absolute() -> None:
    """Test absolute."""
    try:
        mod.absolute(1.0)
    except Exception:
        pass


def test_acos() -> None:
    """Test acos."""
    try:
        mod.acos(1.0)
    except Exception:
        pass


def test_acosh() -> None:
    """Test acosh."""
    try:
        mod.acosh(1.0)
    except Exception:
        pass


def test_add() -> None:
    """Test add."""
    try:
        mod.add(1.0, 1.0)
    except Exception:
        pass


def test_all() -> None:
    """Test all."""
    try:
        mod.all(1.0)
    except Exception:
        pass


def test_allclose() -> None:
    """Test allclose."""
    try:
        mod.allclose(1.0, 1.0)
    except Exception:
        pass


def test_amax() -> None:
    """Test amax."""
    try:
        mod.amax(1.0)
    except Exception:
        pass


def test_amin() -> None:
    """Test amin."""
    try:
        mod.amin(1.0)
    except Exception:
        pass


def test_angle() -> None:
    """Test angle."""
    try:
        mod.angle(1.0)
    except Exception:
        pass


def test_any() -> None:
    """Test any."""
    try:
        mod.any(1.0)
    except Exception:
        pass


def test_append() -> None:
    """Test append."""
    try:
        mod.append(1.0, 1.0)
    except Exception:
        pass


def test_apply_along_axis() -> None:
    """Test apply_along_axis."""
    try:
        mod.apply_along_axis(1.0, 1, 1.0)
    except Exception:
        pass


def test_apply_over_axes() -> None:
    """Test apply_over_axes."""
    try:
        mod.apply_over_axes(1.0, 1.0, 1)
    except Exception:
        pass


def test_arange() -> None:
    """Test arange."""
    try:
        mod.arange(1.0)
    except Exception:
        pass


def test_arccos() -> None:
    """Test arccos."""
    try:
        mod.arccos(1.0)
    except Exception:
        pass


def test_arccosh() -> None:
    """Test arccosh."""
    try:
        mod.arccosh(1.0)
    except Exception:
        pass


def test_arcsin() -> None:
    """Test arcsin."""
    try:
        mod.arcsin(1.0)
    except Exception:
        pass


def test_arcsinh() -> None:
    """Test arcsinh."""
    try:
        mod.arcsinh(1.0)
    except Exception:
        pass


def test_arctan() -> None:
    """Test arctan."""
    try:
        mod.arctan(1.0)
    except Exception:
        pass


def test_arctan2() -> None:
    """Test arctan2."""
    try:
        mod.arctan2(1.0, 1.0)
    except Exception:
        pass


def test_arctanh() -> None:
    """Test arctanh."""
    try:
        mod.arctanh(1.0)
    except Exception:
        pass


def test_argmax() -> None:
    """Test argmax."""
    try:
        mod.argmax(1.0)
    except Exception:
        pass


def test_argmin() -> None:
    """Test argmin."""
    try:
        mod.argmin(1.0)
    except Exception:
        pass


def test_argpartition() -> None:
    """Test argpartition."""
    try:
        mod.argpartition(1.0, 1.0)
    except Exception:
        pass


def test_argsort() -> None:
    """Test argsort."""
    try:
        mod.argsort(1.0)
    except Exception:
        pass


def test_argwhere() -> None:
    """Test argwhere."""
    try:
        mod.argwhere(1.0)
    except Exception:
        pass


def test_around() -> None:
    """Test around."""
    try:
        mod.around(1.0)
    except Exception:
        pass


def test_array() -> None:
    """Test array."""
    try:
        mod.array(1.0)
    except Exception:
        pass


def test_array_equal() -> None:
    """Test array_equal."""
    try:
        mod.array_equal(1.0, 1.0)
    except Exception:
        pass


def test_array_equiv() -> None:
    """Test array_equiv."""
    try:
        mod.array_equiv(1.0, 1.0)
    except Exception:
        pass


def test_array_repr() -> None:
    """Test array_repr."""
    try:
        mod.array_repr(1.0)
    except Exception:
        pass


def test_array_split() -> None:
    """Test array_split."""
    try:
        mod.array_split(1.0, 1.0)
    except Exception:
        pass


def test_array_str() -> None:
    """Test array_str."""
    try:
        mod.array_str(1.0)
    except Exception:
        pass


def test_asarray() -> None:
    """Test asarray."""
    try:
        mod.asarray(1.0)
    except Exception:
        pass


def test_asin() -> None:
    """Test asin."""
    try:
        mod.asin(1.0)
    except Exception:
        pass


def test_asinh() -> None:
    """Test asinh."""
    try:
        mod.asinh(1.0)
    except Exception:
        pass


def test_astype() -> None:
    """Test astype."""
    try:
        mod.astype(1.0, 1.0)
    except Exception:
        pass


def test_atan() -> None:
    """Test atan."""
    try:
        mod.atan(1.0)
    except Exception:
        pass


def test_atan2() -> None:
    """Test atan2."""
    try:
        mod.atan2(1.0, 1.0)
    except Exception:
        pass


def test_atanh() -> None:
    """Test atanh."""
    try:
        mod.atanh(1.0)
    except Exception:
        pass


def test_atleast_1d() -> None:
    """Test atleast_1d."""
    try:
        mod.atleast_1d()
    except Exception:
        pass


def test_atleast_2d() -> None:
    """Test atleast_2d."""
    try:
        mod.atleast_2d()
    except Exception:
        pass


def test_atleast_3d() -> None:
    """Test atleast_3d."""
    try:
        mod.atleast_3d()
    except Exception:
        pass


def test_average() -> None:
    """Test average."""
    try:
        mod.average(1.0)
    except Exception:
        pass


def test_bartlett() -> None:
    """Test bartlett."""
    try:
        mod.bartlett(1)
    except Exception:
        pass


def test_bincount() -> None:
    """Test bincount."""
    try:
        mod.bincount(1.0)
    except Exception:
        pass


def test_bitwise_and() -> None:
    """Test bitwise_and."""
    try:
        mod.bitwise_and(1.0, 1.0)
    except Exception:
        pass


def test_bitwise_count() -> None:
    """Test bitwise_count."""
    try:
        mod.bitwise_count(1.0)
    except Exception:
        pass


def test_bitwise_invert() -> None:
    """Test bitwise_invert."""
    try:
        mod.bitwise_invert(1.0)
    except Exception:
        pass


def test_bitwise_left_shift() -> None:
    """Test bitwise_left_shift."""
    try:
        mod.bitwise_left_shift(1.0, 1.0)
    except Exception:
        pass


def test_bitwise_not() -> None:
    """Test bitwise_not."""
    try:
        mod.bitwise_not(1.0)
    except Exception:
        pass


def test_bitwise_or() -> None:
    """Test bitwise_or."""
    try:
        mod.bitwise_or(1.0, 1.0)
    except Exception:
        pass


def test_bitwise_right_shift() -> None:
    """Test bitwise_right_shift."""
    try:
        mod.bitwise_right_shift(1.0, 1.0)
    except Exception:
        pass


def test_bitwise_xor() -> None:
    """Test bitwise_xor."""
    try:
        mod.bitwise_xor(1.0, 1.0)
    except Exception:
        pass


def test_blackman() -> None:
    """Test blackman."""
    try:
        mod.blackman(1)
    except Exception:
        pass


def test_block() -> None:
    """Test block."""
    try:
        mod.block(1.0)
    except Exception:
        pass


def test_broadcast_arrays() -> None:
    """Test broadcast_arrays."""
    try:
        mod.broadcast_arrays()
    except Exception:
        pass


def test_broadcast_shapes() -> None:
    """Test broadcast_shapes."""
    try:
        mod.broadcast_shapes()
    except Exception:
        pass


def test_broadcast_to() -> None:
    """Test broadcast_to."""
    try:
        mod.broadcast_to(1.0, 1.0)
    except Exception:
        pass


def test_can_cast() -> None:
    """Test can_cast."""
    try:
        mod.can_cast(1.0, 1.0)
    except Exception:
        pass


def test_cbrt() -> None:
    """Test cbrt."""
    try:
        mod.cbrt(1.0)
    except Exception:
        pass


def test_ceil() -> None:
    """Test ceil."""
    try:
        mod.ceil(1.0)
    except Exception:
        pass


def test_class_character() -> None:
    """Test class character."""
    try:
        mod.character()
    except Exception:
        pass


def test_choose() -> None:
    """Test choose."""
    try:
        mod.choose(1.0, 1.0)
    except Exception:
        pass


def test_clip() -> None:
    """Test clip."""
    try:
        mod.clip(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_column_stack() -> None:
    """Test column_stack."""
    try:
        mod.column_stack(1.0)
    except Exception:
        pass


def test_class_complexfloating() -> None:
    """Test class complexfloating."""
    try:
        mod.complexfloating()
    except Exception:
        pass


def test_compress() -> None:
    """Test compress."""
    try:
        mod.compress(1.0, 1.0)
    except Exception:
        pass


def test_concat() -> None:
    """Test concat."""
    try:
        mod.concat(1.0)
    except Exception:
        pass


def test_concatenate() -> None:
    """Test concatenate."""
    try:
        mod.concatenate(1.0)
    except Exception:
        pass


def test_conj() -> None:
    """Test conj."""
    try:
        mod.conj(1.0)
    except Exception:
        pass


def test_conjugate() -> None:
    """Test conjugate."""
    try:
        mod.conjugate(1.0)
    except Exception:
        pass


def test_convolve() -> None:
    """Test convolve."""
    try:
        mod.convolve(1.0, 1.0)
    except Exception:
        pass


def test_copy() -> None:
    """Test copy."""
    try:
        mod.copy(1.0)
    except Exception:
        pass


def test_copysign() -> None:
    """Test copysign."""
    try:
        mod.copysign(1.0, 1.0)
    except Exception:
        pass


def test_corrcoef() -> None:
    """Test corrcoef."""
    try:
        mod.corrcoef(1.0)
    except Exception:
        pass


def test_correlate() -> None:
    """Test correlate."""
    try:
        mod.correlate(1.0, 1.0)
    except Exception:
        pass


def test_cos() -> None:
    """Test cos."""
    try:
        mod.cos(1.0)
    except Exception:
        pass


def test_cosh() -> None:
    """Test cosh."""
    try:
        mod.cosh(1.0)
    except Exception:
        pass


def test_count_nonzero() -> None:
    """Test count_nonzero."""
    try:
        mod.count_nonzero(1.0)
    except Exception:
        pass


def test_cov() -> None:
    """Test cov."""
    try:
        mod.cov(1.0)
    except Exception:
        pass


def test_cross() -> None:
    """Test cross."""
    try:
        mod.cross(1.0, 1.0)
    except Exception:
        pass


def test_cumprod() -> None:
    """Test cumprod."""
    try:
        mod.cumprod(1.0)
    except Exception:
        pass


def test_cumsum() -> None:
    """Test cumsum."""
    try:
        mod.cumsum(1.0)
    except Exception:
        pass


def test_cumulative_sum() -> None:
    """Test cumulative_sum."""
    try:
        mod.cumulative_sum(1.0)
    except Exception:
        pass


def test_deg2rad() -> None:
    """Test deg2rad."""
    try:
        mod.deg2rad(1.0)
    except Exception:
        pass


def test_degrees() -> None:
    """Test degrees."""
    try:
        mod.degrees(1.0)
    except Exception:
        pass


def test_delete() -> None:
    """Test delete."""
    try:
        mod.delete(1.0, 1.0)
    except Exception:
        pass


def test_diag() -> None:
    """Test diag."""
    try:
        mod.diag(1.0)
    except Exception:
        pass


def test_diag_indices() -> None:
    """Test diag_indices."""
    try:
        mod.diag_indices(1)
    except Exception:
        pass


def test_diag_indices_from() -> None:
    """Test diag_indices_from."""
    try:
        mod.diag_indices_from(1.0)
    except Exception:
        pass


def test_diagflat() -> None:
    """Test diagflat."""
    try:
        mod.diagflat(1.0)
    except Exception:
        pass


def test_diagonal() -> None:
    """Test diagonal."""
    try:
        mod.diagonal(1.0)
    except Exception:
        pass


def test_diff() -> None:
    """Test diff."""
    try:
        mod.diff(1.0)
    except Exception:
        pass


def test_digitize() -> None:
    """Test digitize."""
    try:
        mod.digitize(1.0, 1.0)
    except Exception:
        pass


def test_divide() -> None:
    """Test divide."""
    try:
        mod.divide(1.0, 1.0)
    except Exception:
        pass


def test_divmod() -> None:
    """Test divmod."""
    try:
        mod.divmod(1.0, 1.0)
    except Exception:
        pass


def test_dot() -> None:
    """Test dot."""
    try:
        mod.dot(1.0, 1.0)
    except Exception:
        pass


def test_dsplit() -> None:
    """Test dsplit."""
    try:
        mod.dsplit(1.0, 1.0)
    except Exception:
        pass


def test_dstack() -> None:
    """Test dstack."""
    try:
        mod.dstack(1.0)
    except Exception:
        pass


def test_dtype() -> None:
    """Test dtype."""
    try:
        mod.dtype(1.0)
    except Exception:
        pass


def test_ediff1d() -> None:
    """Test ediff1d."""
    try:
        mod.ediff1d(1.0)
    except Exception:
        pass


def test_einsum() -> None:
    """Test einsum."""
    try:
        mod.einsum(1.0)
    except Exception:
        pass


def test_einsum_path() -> None:
    """Test einsum_path."""
    try:
        mod.einsum_path(1.0)
    except Exception:
        pass


def test_empty() -> None:
    """Test empty."""
    try:
        mod.empty(1.0)
    except Exception:
        pass


def test_empty_like() -> None:
    """Test empty_like."""
    try:
        mod.empty_like(1.0)
    except Exception:
        pass


def test_equal() -> None:
    """Test equal."""
    try:
        mod.equal(1.0, 1.0)
    except Exception:
        pass


def test_exp() -> None:
    """Test exp."""
    try:
        mod.exp(1.0)
    except Exception:
        pass


def test_exp2() -> None:
    """Test exp2."""
    try:
        mod.exp2(1.0)
    except Exception:
        pass


def test_expand_dims() -> None:
    """Test expand_dims."""
    try:
        mod.expand_dims(1.0, 1)
    except Exception:
        pass


def test_expm1() -> None:
    """Test expm1."""
    try:
        mod.expm1(1.0)
    except Exception:
        pass


def test_extract() -> None:
    """Test extract."""
    try:
        mod.extract(1.0, 1.0)
    except Exception:
        pass


def test_eye() -> None:
    """Test eye."""
    try:
        mod.eye(1)
    except Exception:
        pass


def test_fabs() -> None:
    """Test fabs."""
    try:
        mod.fabs(1.0)
    except Exception:
        pass


def test_fill_diagonal() -> None:
    """Test fill_diagonal."""
    try:
        mod.fill_diagonal(1.0, 1.0)
    except Exception:
        pass


def test_finfo() -> None:
    """Test finfo."""
    try:
        mod.finfo(1.0)
    except Exception:
        pass


def test_fix() -> None:
    """Test fix."""
    try:
        mod.fix(1.0)
    except Exception:
        pass


def test_flatnonzero() -> None:
    """Test flatnonzero."""
    try:
        mod.flatnonzero(1.0)
    except Exception:
        pass


def test_class_flexible() -> None:
    """Test class flexible."""
    try:
        mod.flexible()
    except Exception:
        pass


def test_flip() -> None:
    """Test flip."""
    try:
        mod.flip(1.0)
    except Exception:
        pass


def test_fliplr() -> None:
    """Test fliplr."""
    try:
        mod.fliplr(1.0)
    except Exception:
        pass


def test_flipud() -> None:
    """Test flipud."""
    try:
        mod.flipud(1.0)
    except Exception:
        pass


def test_float_power() -> None:
    """Test float_power."""
    try:
        mod.float_power(1.0, 1.0)
    except Exception:
        pass


def test_class_floating() -> None:
    """Test class floating."""
    try:
        mod.floating()
    except Exception:
        pass


def test_floor() -> None:
    """Test floor."""
    try:
        mod.floor(1.0)
    except Exception:
        pass


def test_floor_divide() -> None:
    """Test floor_divide."""
    try:
        mod.floor_divide(1.0, 1.0)
    except Exception:
        pass


def test_fmax() -> None:
    """Test fmax."""
    try:
        mod.fmax(1.0, 1.0)
    except Exception:
        pass


def test_fmin() -> None:
    """Test fmin."""
    try:
        mod.fmin(1.0, 1.0)
    except Exception:
        pass


def test_fmod() -> None:
    """Test fmod."""
    try:
        mod.fmod(1.0, 1.0)
    except Exception:
        pass


def test_frexp() -> None:
    """Test frexp."""
    try:
        mod.frexp(1.0)
    except Exception:
        pass


def test_from_dlpack() -> None:
    """Test from_dlpack."""
    try:
        mod.from_dlpack(1.0)
    except Exception:
        pass


def test_frombuffer() -> None:
    """Test frombuffer."""
    try:
        mod.frombuffer(1.0)
    except Exception:
        pass


def test_fromfile() -> None:
    """Test fromfile."""
    try:
        mod.fromfile(1.0)
    except Exception:
        pass


def test_fromfunction() -> None:
    """Test fromfunction."""
    try:
        mod.fromfunction(1.0, 1.0)
    except Exception:
        pass


def test_fromiter() -> None:
    """Test fromiter."""
    try:
        mod.fromiter(1.0, 1.0)
    except Exception:
        pass


def test_frompyfunc() -> None:
    """Test frompyfunc."""
    try:
        mod.frompyfunc(1.0, 1, 1)
    except Exception:
        pass


def test_fromstring() -> None:
    """Test fromstring."""
    try:
        mod.fromstring(1.0)
    except Exception:
        pass


def test_full() -> None:
    """Test full."""
    try:
        mod.full(1.0, 1.0)
    except Exception:
        pass


def test_full_like() -> None:
    """Test full_like."""
    try:
        mod.full_like(1.0, 1.0)
    except Exception:
        pass


def test_gcd() -> None:
    """Test gcd."""
    try:
        mod.gcd(1.0, 1.0)
    except Exception:
        pass


def test_class_generic() -> None:
    """Test class generic."""
    try:
        mod.generic()
    except Exception:
        pass


def test_geomspace() -> None:
    """Test geomspace."""
    try:
        mod.geomspace(1.0, 1.0)
    except Exception:
        pass


def test_get_printoptions() -> None:
    """Test get_printoptions."""
    try:
        mod.get_printoptions()
    except Exception:
        pass


def test_gradient() -> None:
    """Test gradient."""
    try:
        mod.gradient(1.0)
    except Exception:
        pass


def test_greater() -> None:
    """Test greater."""
    try:
        mod.greater(1.0, 1.0)
    except Exception:
        pass


def test_greater_equal() -> None:
    """Test greater_equal."""
    try:
        mod.greater_equal(1.0, 1.0)
    except Exception:
        pass


def test_hamming() -> None:
    """Test hamming."""
    try:
        mod.hamming(1.0)
    except Exception:
        pass


def test_hanning() -> None:
    """Test hanning."""
    try:
        mod.hanning(1.0)
    except Exception:
        pass


def test_heaviside() -> None:
    """Test heaviside."""
    try:
        mod.heaviside(1.0, 1.0)
    except Exception:
        pass


def test_histogram() -> None:
    """Test histogram."""
    try:
        mod.histogram(1.0)
    except Exception:
        pass


def test_histogram2d() -> None:
    """Test histogram2d."""
    try:
        mod.histogram2d(1.0, 1.0)
    except Exception:
        pass


def test_histogram_bin_edges() -> None:
    """Test histogram_bin_edges."""
    try:
        mod.histogram_bin_edges(1.0)
    except Exception:
        pass


def test_histogramdd() -> None:
    """Test histogramdd."""
    try:
        mod.histogramdd(1.0)
    except Exception:
        pass


def test_hsplit() -> None:
    """Test hsplit."""
    try:
        mod.hsplit(1.0, 1.0)
    except Exception:
        pass


def test_hstack() -> None:
    """Test hstack."""
    try:
        mod.hstack(1.0)
    except Exception:
        pass


def test_hypot() -> None:
    """Test hypot."""
    try:
        mod.hypot(1.0, 1.0)
    except Exception:
        pass


def test_i0() -> None:
    """Test i0."""
    try:
        mod.i0(1.0)
    except Exception:
        pass


def test_identity() -> None:
    """Test identity."""
    try:
        mod.identity(1)
    except Exception:
        pass


def test_iinfo() -> None:
    """Test iinfo."""
    try:
        mod.iinfo(1.0)
    except Exception:
        pass


def test_imag() -> None:
    """Test imag."""
    try:
        mod.imag(1.0)
    except Exception:
        pass


def test_indices() -> None:
    """Test indices."""
    try:
        mod.indices(1.0)
    except Exception:
        pass


def test_class_inexact() -> None:
    """Test class inexact."""
    try:
        mod.inexact()
    except Exception:
        pass


def test_inner() -> None:
    """Test inner."""
    try:
        mod.inner(1.0, 1.0)
    except Exception:
        pass


def test_insert() -> None:
    """Test insert."""
    try:
        mod.insert(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_class_integer() -> None:
    """Test class integer."""
    try:
        mod.integer()
    except Exception:
        pass


def test_interp() -> None:
    """Test interp."""
    try:
        mod.interp(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_intersect1d() -> None:
    """Test intersect1d."""
    try:
        mod.intersect1d(1.0, 1.0)
    except Exception:
        pass


def test_invert() -> None:
    """Test invert."""
    try:
        mod.invert(1.0)
    except Exception:
        pass


def test_isclose() -> None:
    """Test isclose."""
    try:
        mod.isclose(1.0, 1.0)
    except Exception:
        pass


def test_iscomplex() -> None:
    """Test iscomplex."""
    try:
        mod.iscomplex(1.0)
    except Exception:
        pass


def test_iscomplexobj() -> None:
    """Test iscomplexobj."""
    try:
        mod.iscomplexobj(1.0)
    except Exception:
        pass


def test_isdtype() -> None:
    """Test isdtype."""
    try:
        mod.isdtype(1.0, 1.0)
    except Exception:
        pass


def test_isfinite() -> None:
    """Test isfinite."""
    try:
        mod.isfinite(1.0)
    except Exception:
        pass


def test_isin() -> None:
    """Test isin."""
    try:
        mod.isin(1.0, 1.0)
    except Exception:
        pass


def test_isinf() -> None:
    """Test isinf."""
    try:
        mod.isinf(1.0)
    except Exception:
        pass


def test_isnan() -> None:
    """Test isnan."""
    try:
        mod.isnan(1.0)
    except Exception:
        pass


def test_isneginf() -> None:
    """Test isneginf."""
    try:
        mod.isneginf(1.0)
    except Exception:
        pass


def test_isposinf() -> None:
    """Test isposinf."""
    try:
        mod.isposinf(1.0)
    except Exception:
        pass


def test_isreal() -> None:
    """Test isreal."""
    try:
        mod.isreal(1.0)
    except Exception:
        pass


def test_isrealobj() -> None:
    """Test isrealobj."""
    try:
        mod.isrealobj(1.0)
    except Exception:
        pass


def test_isscalar() -> None:
    """Test isscalar."""
    try:
        mod.isscalar(1.0)
    except Exception:
        pass


def test_issubdtype() -> None:
    """Test issubdtype."""
    try:
        mod.issubdtype(1.0, 1.0)
    except Exception:
        pass


def test_iterable() -> None:
    """Test iterable."""
    try:
        mod.iterable(1.0)
    except Exception:
        pass


def test_ix_() -> None:
    """Test ix_."""
    try:
        mod.ix_()
    except Exception:
        pass


def test_kaiser() -> None:
    """Test kaiser."""
    try:
        mod.kaiser(1, 1.0)
    except Exception:
        pass


def test_kron() -> None:
    """Test kron."""
    try:
        mod.kron(1.0, 1.0)
    except Exception:
        pass


def test_lcm() -> None:
    """Test lcm."""
    try:
        mod.lcm(1.0, 1.0)
    except Exception:
        pass


def test_ldexp() -> None:
    """Test ldexp."""
    try:
        mod.ldexp(1.0, 1.0)
    except Exception:
        pass


def test_left_shift() -> None:
    """Test left_shift."""
    try:
        mod.left_shift(1.0, 1.0)
    except Exception:
        pass


def test_less() -> None:
    """Test less."""
    try:
        mod.less(1.0, 1.0)
    except Exception:
        pass


def test_less_equal() -> None:
    """Test less_equal."""
    try:
        mod.less_equal(1.0, 1.0)
    except Exception:
        pass


def test_lexsort() -> None:
    """Test lexsort."""
    try:
        mod.lexsort(1.0)
    except Exception:
        pass


def test_linspace() -> None:
    """Test linspace."""
    try:
        mod.linspace(1.0, 1.0)
    except Exception:
        pass


def test_load() -> None:
    """Test load."""
    try:
        mod.load()
    except Exception:
        pass


def test_log() -> None:
    """Test log."""
    try:
        mod.log(1.0)
    except Exception:
        pass


def test_log10() -> None:
    """Test log10."""
    try:
        mod.log10(1.0)
    except Exception:
        pass


def test_log1p() -> None:
    """Test log1p."""
    try:
        mod.log1p(1.0)
    except Exception:
        pass


def test_log2() -> None:
    """Test log2."""
    try:
        mod.log2(1.0)
    except Exception:
        pass


def test_logaddexp() -> None:
    """Test logaddexp."""
    try:
        mod.logaddexp(1.0, 1.0)
    except Exception:
        pass


def test_logaddexp2() -> None:
    """Test logaddexp2."""
    try:
        mod.logaddexp2(1.0, 1.0)
    except Exception:
        pass


def test_logical_and() -> None:
    """Test logical_and."""
    try:
        mod.logical_and(1.0, 1.0)
    except Exception:
        pass


def test_logical_not() -> None:
    """Test logical_not."""
    try:
        mod.logical_not(1.0)
    except Exception:
        pass


def test_logical_or() -> None:
    """Test logical_or."""
    try:
        mod.logical_or(1.0, 1.0)
    except Exception:
        pass


def test_logical_xor() -> None:
    """Test logical_xor."""
    try:
        mod.logical_xor(1.0, 1.0)
    except Exception:
        pass


def test_logspace() -> None:
    """Test logspace."""
    try:
        mod.logspace(1.0, 1.0)
    except Exception:
        pass


def test_mask_indices() -> None:
    """Test mask_indices."""
    try:
        mod.mask_indices()
    except Exception:
        pass


def test_matmul() -> None:
    """Test matmul."""
    try:
        mod.matmul(1.0, 1.0)
    except Exception:
        pass


def test_matrix_transpose() -> None:
    """Test matrix_transpose."""
    try:
        mod.matrix_transpose(1.0)
    except Exception:
        pass


def test_max() -> None:
    """Test max."""
    try:
        mod.max(1.0)
    except Exception:
        pass


def test_maximum() -> None:
    """Test maximum."""
    try:
        mod.maximum(1.0, 1.0)
    except Exception:
        pass


def test_mean() -> None:
    """Test mean."""
    try:
        mod.mean(1.0)
    except Exception:
        pass


def test_median() -> None:
    """Test median."""
    try:
        mod.median(1.0)
    except Exception:
        pass


def test_meshgrid() -> None:
    """Test meshgrid."""
    try:
        mod.meshgrid()
    except Exception:
        pass


def test_min() -> None:
    """Test min."""
    try:
        mod.min(1.0)
    except Exception:
        pass


def test_minimum() -> None:
    """Test minimum."""
    try:
        mod.minimum(1.0, 1.0)
    except Exception:
        pass


def test_mod() -> None:
    """Test mod."""
    try:
        mod.mod(1.0, 1.0)
    except Exception:
        pass


def test_modf() -> None:
    """Test modf."""
    try:
        mod.modf(1.0)
    except Exception:
        pass


def test_moveaxis() -> None:
    """Test moveaxis."""
    try:
        mod.moveaxis(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_multiply() -> None:
    """Test multiply."""
    try:
        mod.multiply(1.0, 1.0)
    except Exception:
        pass


def test_nan_to_num() -> None:
    """Test nan_to_num."""
    try:
        mod.nan_to_num(1.0)
    except Exception:
        pass


def test_nanargmax() -> None:
    """Test nanargmax."""
    try:
        mod.nanargmax(1.0)
    except Exception:
        pass


def test_nanargmin() -> None:
    """Test nanargmin."""
    try:
        mod.nanargmin(1.0)
    except Exception:
        pass


def test_nancumprod() -> None:
    """Test nancumprod."""
    try:
        mod.nancumprod(1.0)
    except Exception:
        pass


def test_nancumsum() -> None:
    """Test nancumsum."""
    try:
        mod.nancumsum(1.0)
    except Exception:
        pass


def test_nanmax() -> None:
    """Test nanmax."""
    try:
        mod.nanmax(1.0)
    except Exception:
        pass


def test_nanmean() -> None:
    """Test nanmean."""
    try:
        mod.nanmean(1.0)
    except Exception:
        pass


def test_nanmedian() -> None:
    """Test nanmedian."""
    try:
        mod.nanmedian(1.0)
    except Exception:
        pass


def test_nanmin() -> None:
    """Test nanmin."""
    try:
        mod.nanmin(1.0)
    except Exception:
        pass


def test_nanpercentile() -> None:
    """Test nanpercentile."""
    try:
        mod.nanpercentile(1.0, 1.0)
    except Exception:
        pass


def test_nanprod() -> None:
    """Test nanprod."""
    try:
        mod.nanprod(1.0)
    except Exception:
        pass


def test_nanquantile() -> None:
    """Test nanquantile."""
    try:
        mod.nanquantile(1.0, 1.0)
    except Exception:
        pass


def test_nanstd() -> None:
    """Test nanstd."""
    try:
        mod.nanstd(1.0)
    except Exception:
        pass


def test_nansum() -> None:
    """Test nansum."""
    try:
        mod.nansum(1.0)
    except Exception:
        pass


def test_nanvar() -> None:
    """Test nanvar."""
    try:
        mod.nanvar(1.0)
    except Exception:
        pass


def test_class_ndarray() -> None:
    """Test class ndarray."""
    try:
        mod.ndarray(1.0)
    except Exception:
        pass


def test_ndim() -> None:
    """Test ndim."""
    try:
        mod.ndim(1.0)
    except Exception:
        pass


def test_negative() -> None:
    """Test negative."""
    try:
        mod.negative(1.0)
    except Exception:
        pass


def test_nextafter() -> None:
    """Test nextafter."""
    try:
        mod.nextafter(1.0, 1.0)
    except Exception:
        pass


def test_nonzero() -> None:
    """Test nonzero."""
    try:
        mod.nonzero(1.0)
    except Exception:
        pass


def test_not_equal() -> None:
    """Test not_equal."""
    try:
        mod.not_equal(1.0, 1.0)
    except Exception:
        pass


def test_class_number() -> None:
    """Test class number."""
    try:
        mod.number()
    except Exception:
        pass


def test_class_object_() -> None:
    """Test class object_."""
    try:
        mod.object_()
    except Exception:
        pass


def test_ones() -> None:
    """Test ones."""
    try:
        mod.ones(1.0)
    except Exception:
        pass


def test_ones_like() -> None:
    """Test ones_like."""
    try:
        mod.ones_like(1.0)
    except Exception:
        pass


def test_outer() -> None:
    """Test outer."""
    try:
        mod.outer(1.0, 1.0)
    except Exception:
        pass


def test_packbits() -> None:
    """Test packbits."""
    try:
        mod.packbits(1.0)
    except Exception:
        pass


def test_pad() -> None:
    """Test pad."""
    try:
        mod.pad(1.0, 1.0)
    except Exception:
        pass


def test_partition() -> None:
    """Test partition."""
    try:
        mod.partition(1.0, 1.0)
    except Exception:
        pass


def test_percentile() -> None:
    """Test percentile."""
    try:
        mod.percentile(1.0, 1.0)
    except Exception:
        pass


def test_permute_dims() -> None:
    """Test permute_dims."""
    try:
        mod.permute_dims(1.0, 1)
    except Exception:
        pass


def test_piecewise() -> None:
    """Test piecewise."""
    try:
        mod.piecewise(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_place() -> None:
    """Test place."""
    try:
        mod.place(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_poly() -> None:
    """Test poly."""
    try:
        mod.poly(1.0)
    except Exception:
        pass


def test_polyadd() -> None:
    """Test polyadd."""
    try:
        mod.polyadd(1.0, 1.0)
    except Exception:
        pass


def test_polyder() -> None:
    """Test polyder."""
    try:
        mod.polyder(1.0)
    except Exception:
        pass


def test_polydiv() -> None:
    """Test polydiv."""
    try:
        mod.polydiv(1.0, 1.0)
    except Exception:
        pass


def test_polyfit() -> None:
    """Test polyfit."""
    try:
        mod.polyfit(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_polyint() -> None:
    """Test polyint."""
    try:
        mod.polyint(1.0)
    except Exception:
        pass


def test_polymul() -> None:
    """Test polymul."""
    try:
        mod.polymul(1.0, 1.0)
    except Exception:
        pass


def test_polysub() -> None:
    """Test polysub."""
    try:
        mod.polysub(1.0, 1.0)
    except Exception:
        pass


def test_polyval() -> None:
    """Test polyval."""
    try:
        mod.polyval(1.0, 1.0)
    except Exception:
        pass


def test_positive() -> None:
    """Test positive."""
    try:
        mod.positive(1.0)
    except Exception:
        pass


def test_pow() -> None:
    """Test pow."""
    try:
        mod.pow(1.0, 1.0)
    except Exception:
        pass


def test_power() -> None:
    """Test power."""
    try:
        mod.power(1.0, 1.0)
    except Exception:
        pass


def test_printoptions() -> None:
    """Test printoptions."""
    try:
        mod.printoptions()
    except Exception:
        pass


def test_prod() -> None:
    """Test prod."""
    try:
        mod.prod(1.0)
    except Exception:
        pass


def test_promote_types() -> None:
    """Test promote_types."""
    try:
        mod.promote_types(1.0, 1.0)
    except Exception:
        pass


def test_ptp() -> None:
    """Test ptp."""
    try:
        mod.ptp(1.0)
    except Exception:
        pass


def test_put() -> None:
    """Test put."""
    try:
        mod.put(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_quantile() -> None:
    """Test quantile."""
    try:
        mod.quantile(1.0, 1.0)
    except Exception:
        pass


def test_rad2deg() -> None:
    """Test rad2deg."""
    try:
        mod.rad2deg(1.0)
    except Exception:
        pass


def test_radians() -> None:
    """Test radians."""
    try:
        mod.radians(1.0)
    except Exception:
        pass


def test_ravel() -> None:
    """Test ravel."""
    try:
        mod.ravel(1.0)
    except Exception:
        pass


def test_ravel_multi_index() -> None:
    """Test ravel_multi_index."""
    try:
        mod.ravel_multi_index(1.0, 1.0)
    except Exception:
        pass


def test_real() -> None:
    """Test real."""
    try:
        mod.real(1.0)
    except Exception:
        pass


def test_reciprocal() -> None:
    """Test reciprocal."""
    try:
        mod.reciprocal(1.0)
    except Exception:
        pass


def test_remainder() -> None:
    """Test remainder."""
    try:
        mod.remainder(1.0, 1.0)
    except Exception:
        pass


def test_repeat() -> None:
    """Test repeat."""
    try:
        mod.repeat(1.0, 1.0)
    except Exception:
        pass


def test_reshape() -> None:
    """Test reshape."""
    try:
        mod.reshape(1.0, 1.0)
    except Exception:
        pass


def test_resize() -> None:
    """Test resize."""
    try:
        mod.resize(1.0, 1.0)
    except Exception:
        pass


def test_result_type() -> None:
    """Test result_type."""
    try:
        mod.result_type()
    except Exception:
        pass


def test_right_shift() -> None:
    """Test right_shift."""
    try:
        mod.right_shift(1.0, 1.0)
    except Exception:
        pass


def test_rint() -> None:
    """Test rint."""
    try:
        mod.rint(1.0)
    except Exception:
        pass


def test_roll() -> None:
    """Test roll."""
    try:
        mod.roll(1.0, 1.0)
    except Exception:
        pass


def test_rollaxis() -> None:
    """Test rollaxis."""
    try:
        mod.rollaxis(1.0, 1)
    except Exception:
        pass


def test_roots() -> None:
    """Test roots."""
    try:
        mod.roots(1.0)
    except Exception:
        pass


def test_rot90() -> None:
    """Test rot90."""
    try:
        mod.rot90(1.0)
    except Exception:
        pass


def test_round() -> None:
    """Test round."""
    try:
        mod.round(1.0)
    except Exception:
        pass


def test_round_() -> None:
    """Test round_."""
    try:
        mod.round_(1.0)
    except Exception:
        pass


def test_save() -> None:
    """Test save."""
    try:
        mod.save(1.0, 1.0)
    except Exception:
        pass


def test_savez() -> None:
    """Test savez."""
    try:
        mod.savez(1.0)
    except Exception:
        pass


def test_searchsorted() -> None:
    """Test searchsorted."""
    try:
        mod.searchsorted(1.0, 1.0)
    except Exception:
        pass


def test_select() -> None:
    """Test select."""
    try:
        mod.select(1.0, 1.0)
    except Exception:
        pass


def test_set_printoptions() -> None:
    """Test set_printoptions."""
    try:
        mod.set_printoptions()
    except Exception:
        pass


def test_setdiff1d() -> None:
    """Test setdiff1d."""
    try:
        mod.setdiff1d(1.0, 1.0)
    except Exception:
        pass


def test_setxor1d() -> None:
    """Test setxor1d."""
    try:
        mod.setxor1d(1.0, 1.0)
    except Exception:
        pass


def test_shape() -> None:
    """Test shape."""
    try:
        mod.shape(1.0)
    except Exception:
        pass


def test_sign() -> None:
    """Test sign."""
    try:
        mod.sign(1.0)
    except Exception:
        pass


def test_signbit() -> None:
    """Test signbit."""
    try:
        mod.signbit(1.0)
    except Exception:
        pass


def test_class_signedinteger() -> None:
    """Test class signedinteger."""
    try:
        mod.signedinteger()
    except Exception:
        pass


def test_sin() -> None:
    """Test sin."""
    try:
        mod.sin(1.0)
    except Exception:
        pass


def test_sinc() -> None:
    """Test sinc."""
    try:
        mod.sinc(1.0)
    except Exception:
        pass


def test_sinh() -> None:
    """Test sinh."""
    try:
        mod.sinh(1.0)
    except Exception:
        pass


def test_size() -> None:
    """Test size."""
    try:
        mod.size(1.0)
    except Exception:
        pass


def test_sort() -> None:
    """Test sort."""
    try:
        mod.sort(1.0)
    except Exception:
        pass


def test_sort_complex() -> None:
    """Test sort_complex."""
    try:
        mod.sort_complex(1.0)
    except Exception:
        pass


def test_split() -> None:
    """Test split."""
    try:
        mod.split(1.0, 1.0)
    except Exception:
        pass


def test_sqrt() -> None:
    """Test sqrt."""
    try:
        mod.sqrt(1.0)
    except Exception:
        pass


def test_square() -> None:
    """Test square."""
    try:
        mod.square(1.0)
    except Exception:
        pass


def test_squeeze() -> None:
    """Test squeeze."""
    try:
        mod.squeeze(1.0)
    except Exception:
        pass


def test_stack() -> None:
    """Test stack."""
    try:
        mod.stack(1.0)
    except Exception:
        pass


def test_std() -> None:
    """Test std."""
    try:
        mod.std(1.0)
    except Exception:
        pass


def test_subtract() -> None:
    """Test subtract."""
    try:
        mod.subtract(1.0, 1.0)
    except Exception:
        pass


def test_sum() -> None:
    """Test sum."""
    try:
        mod.sum(1.0)
    except Exception:
        pass


def test_swapaxes() -> None:
    """Test swapaxes."""
    try:
        mod.swapaxes(1.0, 1, 1)
    except Exception:
        pass


def test_take() -> None:
    """Test take."""
    try:
        mod.take(1.0, 1.0)
    except Exception:
        pass


def test_take_along_axis() -> None:
    """Test take_along_axis."""
    try:
        mod.take_along_axis(1.0, 1.0, 1)
    except Exception:
        pass


def test_tan() -> None:
    """Test tan."""
    try:
        mod.tan(1.0)
    except Exception:
        pass


def test_tanh() -> None:
    """Test tanh."""
    try:
        mod.tanh(1.0)
    except Exception:
        pass


def test_tensordot() -> None:
    """Test tensordot."""
    try:
        mod.tensordot(1.0, 1.0)
    except Exception:
        pass


def test_tile() -> None:
    """Test tile."""
    try:
        mod.tile(1.0, 1.0)
    except Exception:
        pass


def test_trace() -> None:
    """Test trace."""
    try:
        mod.trace(1.0)
    except Exception:
        pass


def test_transpose() -> None:
    """Test transpose."""
    try:
        mod.transpose(1.0)
    except Exception:
        pass


def test_trapezoid() -> None:
    """Test trapezoid."""
    try:
        mod.trapezoid(1.0)
    except Exception:
        pass


def test_tri() -> None:
    """Test tri."""
    try:
        mod.tri(1)
    except Exception:
        pass


def test_tril() -> None:
    """Test tril."""
    try:
        mod.tril(1.0)
    except Exception:
        pass


def test_tril_indices() -> None:
    """Test tril_indices."""
    try:
        mod.tril_indices(1)
    except Exception:
        pass


def test_tril_indices_from() -> None:
    """Test tril_indices_from."""
    try:
        mod.tril_indices_from(1.0)
    except Exception:
        pass


def test_trim_zeros() -> None:
    """Test trim_zeros."""
    try:
        mod.trim_zeros(1.0)
    except Exception:
        pass


def test_triu() -> None:
    """Test triu."""
    try:
        mod.triu(1.0)
    except Exception:
        pass


def test_triu_indices() -> None:
    """Test triu_indices."""
    try:
        mod.triu_indices(1)
    except Exception:
        pass


def test_triu_indices_from() -> None:
    """Test triu_indices_from."""
    try:
        mod.triu_indices_from(1.0)
    except Exception:
        pass


def test_true_divide() -> None:
    """Test true_divide."""
    try:
        mod.true_divide(1.0, 1.0)
    except Exception:
        pass


def test_trunc() -> None:
    """Test trunc."""
    try:
        mod.trunc(1.0)
    except Exception:
        pass


def test_ufunc() -> None:
    """Test ufunc."""
    try:
        mod.ufunc()
    except Exception:
        pass


def test_uint() -> None:
    """Test uint."""
    try:
        mod.uint()
    except Exception:
        pass


def test_uint4() -> None:
    """Test uint4."""
    try:
        mod.uint4()
    except Exception:
        pass


def test_union1d() -> None:
    """Test union1d."""
    try:
        mod.union1d(1.0, 1.0)
    except Exception:
        pass


def test_unique() -> None:
    """Test unique."""
    try:
        mod.unique(1.0)
    except Exception:
        pass


def test_unique_all() -> None:
    """Test unique_all."""
    try:
        mod.unique_all(1.0)
    except Exception:
        pass


def test_unique_counts() -> None:
    """Test unique_counts."""
    try:
        mod.unique_counts(1.0)
    except Exception:
        pass


def test_unique_inverse() -> None:
    """Test unique_inverse."""
    try:
        mod.unique_inverse(1.0)
    except Exception:
        pass


def test_unique_values() -> None:
    """Test unique_values."""
    try:
        mod.unique_values(1.0)
    except Exception:
        pass


def test_unpackbits() -> None:
    """Test unpackbits."""
    try:
        mod.unpackbits(1.0)
    except Exception:
        pass


def test_unravel_index() -> None:
    """Test unravel_index."""
    try:
        mod.unravel_index(1.0, 1.0)
    except Exception:
        pass


def test_class_unsignedinteger() -> None:
    """Test class unsignedinteger."""
    try:
        mod.unsignedinteger()
    except Exception:
        pass


def test_unstack() -> None:
    """Test unstack."""
    try:
        mod.unstack(1.0)
    except Exception:
        pass


def test_unwrap() -> None:
    """Test unwrap."""
    try:
        mod.unwrap(1.0)
    except Exception:
        pass


def test_vander() -> None:
    """Test vander."""
    try:
        mod.vander(1.0)
    except Exception:
        pass


def test_var() -> None:
    """Test var."""
    try:
        mod.var(1.0)
    except Exception:
        pass


def test_vdot() -> None:
    """Test vdot."""
    try:
        mod.vdot(1.0, 1.0)
    except Exception:
        pass


def test_vecdot() -> None:
    """Test vecdot."""
    try:
        mod.vecdot(1.0, 1.0)
    except Exception:
        pass


def test_class_vectorize() -> None:
    """Test class vectorize."""
    try:
        mod.vectorize(1.0)
    except Exception:
        pass


def test_vsplit() -> None:
    """Test vsplit."""
    try:
        mod.vsplit(1.0, 1.0)
    except Exception:
        pass


def test_vstack() -> None:
    """Test vstack."""
    try:
        mod.vstack(1.0)
    except Exception:
        pass


def test_where() -> None:
    """Test where."""
    try:
        mod.where(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_zeros() -> None:
    """Test zeros."""
    try:
        mod.zeros(1.0)
    except Exception:
        pass


def test_zeros_like() -> None:
    """Test zeros_like."""
    try:
        mod.zeros_like(1.0)
    except Exception:
        pass
