"""Tests for zero_jax.numpy.lax_numpy."""

from __future__ import annotations

import pytest

import zero_jax.numpy.lax_numpy as mod


def test_class_ComplexWarning() -> None:
    """Test ComplexWarning."""
    w = mod.ComplexWarning("warn")
    assert isinstance(w, Warning)


def test_List() -> None:
    """Test List."""
    assert "List" not in dir(mod)


def test_Optional() -> None:
    """Test Optional."""
    assert "Optional" not in dir(mod)


def test_class_Tensor() -> None:
    """Test Tensor."""
    assert "Tensor" not in dir(mod)


def test_Tuple() -> None:
    """Test Tuple."""
    assert "Tuple" not in dir(mod)


def test_abs() -> None:
    """Test abs."""
    res = mod.abs(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_absolute() -> None:
    """Test absolute."""
    res = mod.absolute(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_acos() -> None:
    """Test acos."""
    res = mod.acos(mod.array([0.0, 0.5]))
    assert res.shape == (2,)


def test_acosh() -> None:
    """Test acosh."""
    res = mod.acosh(mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_add() -> None:
    """Test add."""
    res = mod.add(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_all() -> None:
    """Test all."""
    res = mod.all(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_allclose() -> None:
    """Test allclose."""
    res = mod.allclose(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == ()


def test_amax() -> None:
    """Test amax."""
    res = mod.amax(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_amin() -> None:
    """Test amin."""
    res = mod.amin(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_angle() -> None:
    """Test angle."""
    res = mod.angle(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_any() -> None:
    """Test any."""
    res = mod.any(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_append() -> None:
    """Test append."""
    res = mod.append(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (4,)


def test_apply_along_axis() -> None:
    """Test apply_along_axis."""
    res = mod.apply_along_axis(mod.sum, 0, mod.array([[1.0, 2.0], [3.0, 4.0]]))
    assert res.shape == ()


def test_apply_over_axes() -> None:
    """Test apply_over_axes."""
    res = mod.apply_over_axes(mod.sum, mod.array([[1.0, 2.0], [3.0, 4.0]]), [0])
    assert res.shape == (1, 2)


def test_arange() -> None:
    """Test arange."""
    res = mod.arange(4)
    assert res.shape == (4,)


def test_arccos() -> None:
    """Test arccos."""
    res = mod.arccos(mod.array([0.0, 0.5]))
    assert res.shape == (2,)


def test_arccosh() -> None:
    """Test arccosh."""
    res = mod.arccosh(mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_arcsin() -> None:
    """Test arcsin."""
    res = mod.arcsin(mod.array([0.0, 0.5]))
    assert res.shape == (2,)


def test_arcsinh() -> None:
    """Test arcsinh."""
    res = mod.arcsinh(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_arctan() -> None:
    """Test arctan."""
    res = mod.arctan(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_arctan2() -> None:
    """Test arctan2."""
    res = mod.arctan2(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_arctanh() -> None:
    """Test arctanh."""
    res = mod.arctanh(mod.array([0.0, 0.5]))
    assert res.shape == (2,)


def test_argmax() -> None:
    """Test argmax."""
    res = mod.argmax(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_argmin() -> None:
    """Test argmin."""
    res = mod.argmin(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_argpartition() -> None:
    """Test argpartition."""
    res = mod.argpartition(mod.array([3.0, 1.0, 2.0]), 1)
    assert res.shape == (3,)


def test_argsort() -> None:
    """Test argsort."""
    res = mod.argsort(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_argwhere() -> None:
    """Test argwhere."""
    res = mod.argwhere(mod.array([0.5, 1.5]))
    assert res.shape == (2, 1)


def test_around() -> None:
    """Test around."""
    res = mod.around(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_array() -> None:
    """Test array."""
    res = mod.array(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_array_equal() -> None:
    """Test array_equal."""
    res = mod.array_equal(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res is False


def test_array_equiv() -> None:
    """Test array_equiv."""
    res = mod.array_equiv(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == ()


def test_array_repr() -> None:
    """Test array_repr."""
    res = mod.array_repr(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_array_split() -> None:
    """Test array_split."""
    res = mod.array_split(mod.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert len(res) == 2


def test_array_str() -> None:
    """Test array_str."""
    res = mod.array_str(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_asarray() -> None:
    """Test asarray."""
    res = mod.asarray(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_asin() -> None:
    """Test asin."""
    res = mod.asin(mod.array([0.0, 0.5]))
    assert res.shape == (2,)


def test_asinh() -> None:
    """Test asinh."""
    res = mod.asinh(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_astype() -> None:
    """Test astype."""
    res = mod.astype(mod.array([1.0, 2.0]), mod.int32)
    assert res.shape == (2,)


def test_atan() -> None:
    """Test atan."""
    res = mod.atan(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_atan2() -> None:
    """Test atan2."""
    res = mod.atan2(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_atanh() -> None:
    """Test atanh."""
    res = mod.atanh(mod.array([0.0, 0.5]))
    assert res.shape == (2,)


def test_atleast_1d() -> None:
    """Test atleast_1d."""
    res = mod.atleast_1d(mod.array(1.0))
    assert res.shape == (1,)


def test_atleast_2d() -> None:
    """Test atleast_2d."""
    res = mod.atleast_2d(mod.array([1.0, 2.0]))
    assert res.shape == (1, 2)


def test_atleast_3d() -> None:
    """Test atleast_3d."""
    res = mod.atleast_3d(mod.array([1.0, 2.0]))
    assert res.shape == (1, 2, 1)


def test_average() -> None:
    """Test average."""
    res = mod.average(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_bartlett() -> None:
    """Test bartlett."""
    res = mod.bartlett(4)
    assert res.shape == (4,)


def test_bincount() -> None:
    """Test bincount."""
    res = mod.bincount(mod.array([0, 1, 1, 2], dtype=mod.int32))
    assert res.shape == (3,)


def test_bitwise_and() -> None:
    """Test bitwise_and."""
    res = mod.bitwise_and(mod.array([1, 3], dtype=mod.int32), mod.array([1, 2], dtype=mod.int32))
    assert res.shape == (2,)


def test_bitwise_count() -> None:
    """Test bitwise_count."""
    res = mod.bitwise_count(mod.array([1, 3], dtype=mod.int32))
    assert res.shape == (2,)


def test_bitwise_invert() -> None:
    """Test bitwise_invert."""
    res = mod.bitwise_invert(mod.array([1, 3], dtype=mod.int32))
    assert res.shape == (2,)


def test_bitwise_left_shift() -> None:
    """Test bitwise_left_shift."""
    res = mod.bitwise_left_shift(mod.array([1, 2], dtype=mod.int32), mod.array([1, 1], dtype=mod.int32))
    assert res.shape == (2,)


def test_bitwise_not() -> None:
    """Test bitwise_not."""
    res = mod.bitwise_not(mod.array([1, 3], dtype=mod.int32))
    assert res.shape == (2,)


def test_bitwise_or() -> None:
    """Test bitwise_or."""
    res = mod.bitwise_or(mod.array([1, 2], dtype=mod.int32), mod.array([2, 1], dtype=mod.int32))
    assert res.shape == (2,)


def test_bitwise_right_shift() -> None:
    """Test bitwise_right_shift."""
    res = mod.bitwise_right_shift(mod.array([2, 4], dtype=mod.int32), mod.array([1, 1], dtype=mod.int32))
    assert res.shape == (2,)


def test_bitwise_xor() -> None:
    """Test bitwise_xor."""
    res = mod.bitwise_xor(mod.array([1, 3], dtype=mod.int32), mod.array([1, 2], dtype=mod.int32))
    assert res.shape == (2,)


def test_blackman() -> None:
    """Test blackman."""
    res = mod.blackman(4)
    assert res.shape == (4,)


def test_block() -> None:
    """Test block."""
    res = mod.block([mod.array([1.0, 2.0]), mod.array([3.0, 4.0])])
    assert res.shape == (4,)


def test_broadcast_arrays() -> None:
    """Test broadcast_arrays."""
    res = mod.broadcast_arrays(mod.array([1.0, 2.0]), mod.array([[1.0], [2.0]]))
    assert len(res) == 2


def test_broadcast_shapes() -> None:
    """Test broadcast_shapes."""
    res = mod.broadcast_shapes((2, 1), (1, 3))
    assert res == (2, 3)


def test_broadcast_to() -> None:
    """Test broadcast_to."""
    res = mod.broadcast_to(mod.array([1.0, 2.0]), (3, 2))
    assert res.shape == (3, 2)


def test_can_cast() -> None:
    """Test can_cast."""
    assert isinstance(mod.can_cast(mod.float32, mod.float64), bool)


def test_cbrt() -> None:
    """Test cbrt."""
    res = mod.cbrt(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_ceil() -> None:
    """Test ceil."""
    res = mod.ceil(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_class_character() -> None:
    """Test character."""
    obj = mod.character()
    assert isinstance(obj, mod.flexible)


def test_choose() -> None:
    """Test choose."""
    res = mod.choose(mod.array([0, 1], dtype=mod.int32), [mod.array([10.0, 20.0]), mod.array([30.0, 40.0])])
    assert res.shape == (2,)


def test_clip() -> None:
    """Test clip."""
    res = mod.clip(mod.array([1.0, 5.0]), 2.0, 4.0)
    assert res.shape == (2,)


def test_column_stack() -> None:
    """Test column_stack."""
    res = mod.column_stack([mod.array([1.0, 2.0]), mod.array([3.0, 4.0])])
    assert res.shape == (2, 2)


def test_class_complexfloating() -> None:
    """Test complexfloating."""
    obj = mod.complexfloating()
    assert isinstance(obj, mod.inexact)


def test_compress() -> None:
    """Test compress."""
    res = mod.compress(mod.array([True, False, True]), mod.array([1.0, 2.0, 3.0]))
    assert res.shape == (2,)


def test_concat() -> None:
    """Test concat."""
    res = mod.concat([mod.array([1.0, 2.0]), mod.array([3.0, 4.0])])
    assert res.shape == (4,)


def test_concatenate() -> None:
    """Test concatenate."""
    res = mod.concatenate([mod.array([1.0, 2.0]), mod.array([3.0, 4.0])])
    assert res.shape == (4,)


def test_conj() -> None:
    """Test conj."""
    res = mod.conj(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_conjugate() -> None:
    """Test conjugate."""
    res = mod.conjugate(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_convolve() -> None:
    """Test convolve."""
    res = mod.convolve(mod.array([1.0, 2.0, 3.0]), mod.array([1.0, 1.0]))
    assert res.shape == (4,)


def test_copy() -> None:
    """Test copy."""
    res = mod.copy(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_copysign() -> None:
    """Test copysign."""
    res = mod.copysign(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_corrcoef() -> None:
    """Test corrcoef."""
    res = mod.corrcoef(mod.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))
    assert res.shape == (2, 2)


def test_correlate() -> None:
    """Test correlate."""
    res = mod.correlate(mod.array([1.0, 2.0, 3.0]), mod.array([1.0, 1.0]))
    assert res.shape == (2,)


def test_cos() -> None:
    """Test cos."""
    res = mod.cos(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_cosh() -> None:
    """Test cosh."""
    res = mod.cosh(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_count_nonzero() -> None:
    """Test count_nonzero."""
    res = mod.count_nonzero(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_cov() -> None:
    """Test cov."""
    res = mod.cov(mod.array([1.0, 2.0, 3.0]))
    assert res.shape == ()


def test_cross() -> None:
    """Test cross."""
    res = mod.cross(mod.array([1.0, 0.0, 0.0]), mod.array([0.0, 1.0, 0.0]))
    assert res.shape == (3,)


def test_cumprod() -> None:
    """Test cumprod."""
    res = mod.cumprod(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_cumsum() -> None:
    """Test cumsum."""
    res = mod.cumsum(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_cumulative_sum() -> None:
    """Test cumulative_sum."""
    res = mod.cumulative_sum(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_deg2rad() -> None:
    """Test deg2rad."""
    res = mod.deg2rad(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_degrees() -> None:
    """Test degrees."""
    res = mod.degrees(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_delete() -> None:
    """Test delete."""
    res = mod.delete(mod.array([1.0, 2.0, 3.0]), 1)
    assert res.shape == (2,)


def test_diag() -> None:
    """Test diag."""
    res = mod.diag(mod.array([0.5, 1.5]))
    assert res.shape == (2, 2)


def test_diag_indices() -> None:
    """Test diag_indices."""
    res = mod.diag_indices(3)
    assert len(res) == 2


def test_diag_indices_from() -> None:
    """Test diag_indices_from."""
    res = mod.diag_indices_from(mod.eye(3))
    assert len(res) == 2


def test_diagflat() -> None:
    """Test diagflat."""
    res = mod.diagflat(mod.array([0.5, 1.5]))
    assert res.shape == (2, 2)


def test_diagonal() -> None:
    """Test diagonal."""
    res = mod.diagonal(mod.eye(3))
    assert res.shape == (3,)


def test_diff() -> None:
    """Test diff."""
    res = mod.diff(mod.array([0.5, 1.5]))
    assert res.shape == (1,)


def test_digitize() -> None:
    """Test digitize."""
    res = mod.digitize(mod.array([0.5, 1.5]), mod.array([0.0, 1.0, 2.0]))
    assert res.shape == (2,)


def test_divide() -> None:
    """Test divide."""
    res = mod.divide(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_divmod() -> None:
    """Test divmod."""
    res = mod.divmod(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert len(res) == 2


def test_dot() -> None:
    """Test dot."""
    res = mod.dot(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == ()


def test_dsplit() -> None:
    """Test dsplit."""
    res = mod.dsplit(mod.ones((2, 2, 4)), 2)
    assert len(res) == 2


def test_dstack() -> None:
    """Test dstack."""
    res = mod.dstack([mod.ones((2, 2)), mod.ones((2, 2))])
    assert res.shape == (2, 2, 2)


def test_dtype() -> None:
    """Test dtype."""
    dt = mod.dtype("float32")
    assert dt == mod.float32


def test_ediff1d() -> None:
    """Test ediff1d."""
    res = mod.ediff1d(mod.array([0.5, 1.5]))
    assert res.shape == (1,)


def test_einsum() -> None:
    """Test einsum."""
    res = mod.einsum("i,i->", mod.array([1.0, 2.0]), mod.array([3.0, 4.0]))
    assert res.shape == ()


def test_einsum_path() -> None:
    """Test einsum_path."""
    path, info = mod.einsum_path("i,i->", mod.array([1.0, 2.0]), mod.array([3.0, 4.0]))
    assert path.shape == ()
    assert info.shape == ()


def test_empty() -> None:
    """Test empty."""
    res = mod.empty((2, 3))
    assert res.shape == (2, 3)


def test_empty_like() -> None:
    """Test empty_like."""
    res = mod.empty_like(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_equal() -> None:
    """Test equal."""
    res = mod.equal(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_exp() -> None:
    """Test exp."""
    res = mod.exp(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_exp2() -> None:
    """Test exp2."""
    res = mod.exp2(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_expand_dims() -> None:
    """Test expand_dims."""
    res = mod.expand_dims(mod.array([1.0, 2.0]), axis=0)
    assert res.shape == (1, 2)


def test_expm1() -> None:
    """Test expm1."""
    res = mod.expm1(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_extract() -> None:
    """Test extract."""
    res = mod.extract(mod.array([True, False, True]), mod.array([1.0, 2.0, 3.0]))
    assert res.shape == (2,)


def test_eye() -> None:
    """Test eye."""
    res = mod.eye(3)
    assert res.shape == (3, 3)


def test_fabs() -> None:
    """Test fabs."""
    res = mod.fabs(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_fill_diagonal() -> None:
    """Test fill_diagonal."""
    res = mod.fill_diagonal(mod.zeros((2, 2)), 5.0, inplace=False)
    assert res.shape == (2, 2)


def test_finfo() -> None:
    """Test finfo."""
    assert mod.finfo(mod.float32).bits == 32


def test_fix() -> None:
    """Test fix."""
    res = mod.fix(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_flatnonzero() -> None:
    """Test flatnonzero."""
    res = mod.flatnonzero(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_class_flexible() -> None:
    """Test flexible."""
    obj = mod.flexible()
    assert isinstance(obj, mod.generic)


def test_flip() -> None:
    """Test flip."""
    res = mod.flip(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_fliplr() -> None:
    """Test fliplr."""
    res = mod.fliplr(mod.eye(2))
    assert res.shape == (2, 2)


def test_flipud() -> None:
    """Test flipud."""
    res = mod.flipud(mod.eye(2))
    assert res.shape == (2, 2)


def test_float_power() -> None:
    """Test float_power."""
    res = mod.float_power(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_class_floating() -> None:
    """Test floating."""
    obj = mod.floating()
    assert isinstance(obj, mod.inexact)


def test_floor() -> None:
    """Test floor."""
    res = mod.floor(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_floor_divide() -> None:
    """Test floor_divide."""
    res = mod.floor_divide(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_fmax() -> None:
    """Test fmax."""
    res = mod.fmax(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_fmin() -> None:
    """Test fmin."""
    res = mod.fmin(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_fmod() -> None:
    """Test fmod."""
    res = mod.fmod(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_frexp() -> None:
    """Test frexp."""
    res = mod.frexp(mod.array([0.5, 1.5]))
    assert len(res) == 2


def test_from_dlpack() -> None:
    """Test from_dlpack."""
    res = mod.from_dlpack(mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_frombuffer() -> None:
    """Test frombuffer."""
    res = mod.frombuffer(b"\x00\x00\x80?", dtype=mod.float32)
    assert res.shape == (4,)


def test_fromfile() -> None:
    """Test fromfile."""
    import tempfile

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(b"1.0 2.0")
        fp.flush()
        res = mod.fromfile(fp.name, sep=" ")
        assert res.shape == (2,)


def test_fromfunction() -> None:
    """Test fromfunction."""
    res = mod.fromfunction(lambda i, j: i + j, (2, 2))
    assert res.shape == (2, 2)


def test_fromiter() -> None:
    """Test fromiter."""
    res = mod.fromiter([1.0, 2.0, 3.0], dtype=mod.float32)
    assert res.shape == (3,)


def test_frompyfunc() -> None:
    """Test frompyfunc."""
    uf = mod.frompyfunc(lambda x: x + 1.0, 1, 1)
    res = uf(mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_fromstring() -> None:
    """Test fromstring."""
    res = mod.fromstring("1.0 2.0", sep=" ")
    assert res.shape == (2,)


def test_full() -> None:
    """Test full."""
    res = mod.full((2, 2), 3.0)
    assert res.shape == (2, 2)


def test_full_like() -> None:
    """Test full_like."""
    res = mod.full_like(mod.array([1.0, 2.0]), 3.0)
    assert res.shape == (2,)


def test_gcd() -> None:
    """Test gcd."""
    res = mod.gcd(mod.array([4, 6], dtype=mod.int32), mod.array([6, 9], dtype=mod.int32))
    assert res.shape == (2,)


def test_class_generic() -> None:
    """Test generic."""
    obj = mod.generic()
    assert isinstance(obj, mod.generic)


def test_geomspace() -> None:
    """Test geomspace."""
    res = mod.geomspace(1.0, 100.0, num=3)
    assert res.shape == (3,)


def test_get_printoptions() -> None:
    """Test get_printoptions."""
    opts = mod.get_printoptions()
    assert isinstance(opts, dict)


def test_gradient() -> None:
    """Test gradient."""
    res = mod.gradient(mod.array([1.0, 2.0, 4.0]))
    assert res.shape == (3,)


def test_greater() -> None:
    """Test greater."""
    res = mod.greater(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_greater_equal() -> None:
    """Test greater_equal."""
    res = mod.greater_equal(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_hamming() -> None:
    """Test hamming."""
    res = mod.hamming(4)
    assert res.shape == (4,)


def test_hanning() -> None:
    """Test hanning."""
    res = mod.hanning(4)
    assert res.shape == (4,)


def test_heaviside() -> None:
    """Test heaviside."""
    res = mod.heaviside(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_histogram() -> None:
    """Test histogram."""
    hist, edges = mod.histogram(mod.array([1.0, 2.0, 3.0]), bins=2)
    assert hist.shape == (2,)
    assert edges.shape == (3,)


def test_histogram2d() -> None:
    """Test histogram2d."""
    h, xe, ye = mod.histogram2d(mod.array([0.0, 1.0]), mod.array([0.0, 1.0]), bins=2)
    assert h.shape == (2, 2)
    assert xe.shape == (3,)
    assert ye.shape == (3,)


def test_histogram_bin_edges() -> None:
    """Test histogram_bin_edges."""
    edges = mod.histogram_bin_edges(mod.array([1.0, 2.0, 3.0]), bins=2)
    assert edges.shape == (3,)


def test_histogramdd() -> None:
    """Test histogramdd."""
    h, edges = mod.histogramdd(mod.array([[0.0, 0.0], [1.0, 1.0]]), bins=2)
    assert h.shape == (2, 2)
    assert len(edges) == 2


def test_hsplit() -> None:
    """Test hsplit."""
    res = mod.hsplit(mod.ones((2, 4)), 2)
    assert len(res) == 2


def test_hstack() -> None:
    """Test hstack."""
    res = mod.hstack([mod.array([1.0, 2.0]), mod.array([3.0, 4.0])])
    assert res.shape == (4,)


def test_hypot() -> None:
    """Test hypot."""
    res = mod.hypot(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_i0() -> None:
    """Test i0."""
    res = mod.i0(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_identity() -> None:
    """Test identity."""
    res = mod.identity(3)
    assert res.shape == (3, 3)


def test_iinfo() -> None:
    """Test iinfo."""
    assert mod.iinfo(mod.int32).bits == 32


def test_imag() -> None:
    """Test imag."""
    res = mod.imag(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_indices() -> None:
    """Test indices."""
    res = mod.indices((2, 3))
    assert res.shape == (2, 2, 3)


def test_class_inexact() -> None:
    """Test inexact."""
    obj = mod.inexact()
    assert isinstance(obj, mod.number)


def test_inner() -> None:
    """Test inner."""
    res = mod.inner(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == ()


def test_insert() -> None:
    """Test insert."""
    res = mod.insert(mod.array([1.0, 2.0]), 1, 9.0)
    assert res.shape == (3,)


def test_class_integer() -> None:
    """Test integer."""
    obj = mod.integer()
    assert isinstance(obj, mod.number)


def test_interp() -> None:
    """Test interp."""
    res = mod.interp(mod.array([1.5]), mod.array([1.0, 2.0]), mod.array([10.0, 20.0]))
    assert res.shape == (1,)


def test_intersect1d() -> None:
    """Test intersect1d."""
    res = mod.intersect1d(mod.array([1, 2]), mod.array([2, 3]))
    assert res.shape == (1,)


def test_invert() -> None:
    """Test invert."""
    res = mod.invert(mod.array([1, 3], dtype=mod.int32))
    assert res.shape == (2,)


def test_isclose() -> None:
    """Test isclose."""
    res = mod.isclose(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_iscomplex() -> None:
    """Test iscomplex."""
    res = mod.iscomplex(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_iscomplexobj() -> None:
    """Test iscomplexobj."""
    res = mod.iscomplexobj(mod.array([0.5, 1.5]))
    assert res is False


def test_isdtype() -> None:
    """Test isdtype."""
    assert isinstance(mod.isdtype(mod.float32, mod.floating), bool)


def test_isfinite() -> None:
    """Test isfinite."""
    res = mod.isfinite(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_isin() -> None:
    """Test isin."""
    res = mod.isin(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_isinf() -> None:
    """Test isinf."""
    res = mod.isinf(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_isnan() -> None:
    """Test isnan."""
    res = mod.isnan(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_isneginf() -> None:
    """Test isneginf."""
    res = mod.isneginf(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_isposinf() -> None:
    """Test isposinf."""
    res = mod.isposinf(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_isreal() -> None:
    """Test isreal."""
    res = mod.isreal(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_isrealobj() -> None:
    """Test isrealobj."""
    res = mod.isrealobj(mod.array([0.5, 1.5]))
    assert res is True


def test_isscalar() -> None:
    """Test isscalar."""
    res = mod.isscalar(mod.array([0.5, 1.5]))
    assert res is False


def test_issubdtype() -> None:
    """Test issubdtype."""
    assert isinstance(mod.issubdtype(mod.float32, mod.floating), bool)


def test_iterable() -> None:
    """Test iterable."""
    res = mod.iterable(mod.array([0.5, 1.5]))
    assert res is True


def test_ix_() -> None:
    """Test ix_."""
    res = mod.ix_(mod.array([0, 1]), mod.array([0, 1]))
    assert len(res) == 2


def test_kaiser() -> None:
    """Test kaiser."""
    res = mod.kaiser(4, 1.0)
    assert res.shape == (4,)


def test_kron() -> None:
    """Test kron."""
    res = mod.kron(mod.array([1.0, 2.0]), mod.array([3.0, 4.0]))
    assert res.shape == (4,)


def test_lcm() -> None:
    """Test lcm."""
    res = mod.lcm(mod.array([4, 6], dtype=mod.int32), mod.array([6, 9], dtype=mod.int32))
    assert res.shape == (2,)


def test_ldexp() -> None:
    """Test ldexp."""
    res = mod.ldexp(mod.array([1.0, 2.0]), mod.array([1, 2], dtype=mod.int32))
    assert res.shape == (2,)


def test_left_shift() -> None:
    """Test left_shift."""
    res = mod.left_shift(mod.array([1, 2], dtype=mod.int32), mod.array([1, 1], dtype=mod.int32))
    assert res.shape == (2,)


def test_less() -> None:
    """Test less."""
    res = mod.less(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_less_equal() -> None:
    """Test less_equal."""
    res = mod.less_equal(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_lexsort() -> None:
    """Test lexsort."""
    res = mod.lexsort((mod.array([1.0, 2.0]), mod.array([2.0, 1.0])))
    assert res.shape == (2,)


def test_linspace() -> None:
    """Test linspace."""
    res = mod.linspace(0.0, 1.0, 5)
    assert res.shape == (5,)


def test_load() -> None:
    """Test load."""
    with pytest.raises(FileNotFoundError):
        mod.load("nonexistent.npy")


def test_log() -> None:
    """Test log."""
    res = mod.log(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_log10() -> None:
    """Test log10."""
    res = mod.log10(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_log1p() -> None:
    """Test log1p."""
    res = mod.log1p(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_log2() -> None:
    """Test log2."""
    res = mod.log2(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_logaddexp() -> None:
    """Test logaddexp."""
    res = mod.logaddexp(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_logaddexp2() -> None:
    """Test logaddexp2."""
    res = mod.logaddexp2(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_logical_and() -> None:
    """Test logical_and."""
    res = mod.logical_and(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_logical_not() -> None:
    """Test logical_not."""
    res = mod.logical_not(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_logical_or() -> None:
    """Test logical_or."""
    res = mod.logical_or(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_logical_xor() -> None:
    """Test logical_xor."""
    res = mod.logical_xor(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_logspace() -> None:
    """Test logspace."""
    res = mod.logspace(0.0, 2.0, 3)
    assert res.shape == (3,)


def test_mask_indices() -> None:
    """Test mask_indices."""
    res = mod.mask_indices(3, lambda x, k=0: mod.triu(x, k)._tensor.data)
    assert len(res) == 2


def test_matmul() -> None:
    """Test matmul."""
    res = mod.matmul(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == ()


def test_matrix_transpose() -> None:
    """Test matrix_transpose."""
    res = mod.matrix_transpose(mod.ones((2, 3)))
    assert res.shape == (3, 2)


def test_max() -> None:
    """Test max."""
    res = mod.max(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_maximum() -> None:
    """Test maximum."""
    res = mod.maximum(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_mean() -> None:
    """Test mean."""
    res = mod.mean(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_median() -> None:
    """Test median."""
    assert mod.median(mod.array([1.0, 2.0, 3.0])) == 2.0


def test_meshgrid() -> None:
    """Test meshgrid."""
    x, y = mod.meshgrid(mod.array([1.0, 2.0]), mod.array([3.0, 4.0]))
    assert x.shape == (2, 2)
    assert y.shape == (2, 2)


def test_min() -> None:
    """Test min."""
    res = mod.min(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_minimum() -> None:
    """Test minimum."""
    res = mod.minimum(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_mod() -> None:
    """Test mod."""
    res = mod.mod(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_modf() -> None:
    """Test modf."""
    res = mod.modf(mod.array([0.5, 1.5]))
    assert len(res) == 2


def test_moveaxis() -> None:
    """Test moveaxis."""
    res = mod.moveaxis(mod.ones((2, 3, 4)), 0, -1)
    assert res.shape == (3, 4, 2)


def test_multiply() -> None:
    """Test multiply."""
    res = mod.multiply(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_nan_to_num() -> None:
    """Test nan_to_num."""
    res = mod.nan_to_num(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_nanargmax() -> None:
    """Test nanargmax."""
    res = mod.nanargmax(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_nanargmin() -> None:
    """Test nanargmin."""
    res = mod.nanargmin(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_nancumprod() -> None:
    """Test nancumprod."""
    res = mod.nancumprod(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_nancumsum() -> None:
    """Test nancumsum."""
    res = mod.nancumsum(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_nanmax() -> None:
    """Test nanmax."""
    res = mod.nanmax(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_nanmean() -> None:
    """Test nanmean."""
    res = mod.nanmean(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_nanmedian() -> None:
    """Test nanmedian."""
    res = mod.nanmedian(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_nanmin() -> None:
    """Test nanmin."""
    res = mod.nanmin(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_nanpercentile() -> None:
    """Test nanpercentile."""
    res = mod.nanpercentile(mod.array([1.0, 2.0, 3.0]), 50.0)
    assert res.shape == ()


def test_nanprod() -> None:
    """Test nanprod."""
    res = mod.nanprod(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_nanquantile() -> None:
    """Test nanquantile."""
    res = mod.nanquantile(mod.array([1.0, 2.0, 3.0]), 0.5)
    assert res.shape == ()


def test_nanstd() -> None:
    """Test nanstd."""
    res = mod.nanstd(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_nansum() -> None:
    """Test nansum."""
    res = mod.nansum(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_nanvar() -> None:
    """Test nanvar."""
    res = mod.nanvar(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_class_ndarray() -> None:
    """Test ndarray."""
    arr = mod.array([1.0, 2.0])
    assert isinstance(arr, mod.ndarray)
    assert arr.shape == (2,)


def test_ndim() -> None:
    """Test ndim."""
    res = mod.ndim(mod.array([0.5, 1.5]))
    assert res == 1


def test_negative() -> None:
    """Test negative."""
    res = mod.negative(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_nextafter() -> None:
    """Test nextafter."""
    res = mod.nextafter(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_nonzero() -> None:
    """Test nonzero."""
    res = mod.nonzero(mod.array([0.5, 1.5]))
    assert len(res) == 1


def test_not_equal() -> None:
    """Test not_equal."""
    res = mod.not_equal(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_class_number() -> None:
    """Test number."""
    obj = mod.number()
    assert isinstance(obj, mod.generic)


def test_class_object_() -> None:
    """Test object_."""
    obj = mod.object_()
    assert isinstance(obj, mod.generic)


def test_ones() -> None:
    """Test ones."""
    res = mod.ones((2, 3))
    assert res.shape == (2, 3)


def test_ones_like() -> None:
    """Test ones_like."""
    res = mod.ones_like(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_outer() -> None:
    """Test outer."""
    res = mod.outer(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2, 2)


def test_packbits() -> None:
    """Test packbits."""
    res = mod.packbits(mod.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=mod.uint8))
    assert res.shape == (1,)


def test_pad() -> None:
    """Test pad."""
    res = mod.pad(mod.array([1.0, 2.0]), (1, 1))
    assert res.shape == (4,)


def test_partition() -> None:
    """Test partition."""
    res = mod.partition(mod.array([3.0, 1.0, 2.0]), 1)
    assert res.shape == (3,)


def test_percentile() -> None:
    """Test percentile."""
    res = mod.percentile(mod.array([1.0, 2.0, 3.0]), 50.0)
    assert res.shape == ()


def test_permute_dims() -> None:
    """Test permute_dims."""
    res = mod.permute_dims(mod.ones((2, 3)), (1, 0))
    assert res.shape == (3, 2)


def test_piecewise() -> None:
    """Test piecewise."""
    res = mod.piecewise(mod.array([1.0, -1.0]), [mod.array([True, False])], [1.0])
    assert res.shape == (2,)


def test_place() -> None:
    """Test place."""
    arr = mod.array([1.0, 2.0])
    mod.place(arr, mod.array([True, False]), mod.array([9.0]))
    assert arr.shape == (2,)


def test_positive() -> None:
    """Test positive."""
    res = mod.positive(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_pow() -> None:
    """Test pow."""
    res = mod.pow(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_power() -> None:
    """Test power."""
    res = mod.power(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_prod() -> None:
    """Test prod."""
    res = mod.prod(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_promote_types() -> None:
    """Test promote_types."""
    res = mod.promote_types(mod.float32, mod.float64)
    assert res.shape == ()


def test_ptp() -> None:
    """Test ptp."""
    res = mod.ptp(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_put() -> None:
    """Test put."""
    arr = mod.array([1.0, 2.0, 3.0])
    mod.put(arr, 1, 9.0)
    assert arr.shape == (3,)


def test_quantile() -> None:
    """Test quantile."""
    res = mod.quantile(mod.array([1.0, 2.0, 3.0]), 0.5)
    assert res.shape == ()


def test_rad2deg() -> None:
    """Test rad2deg."""
    res = mod.rad2deg(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_radians() -> None:
    """Test radians."""
    res = mod.radians(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_ravel() -> None:
    """Test ravel."""
    res = mod.ravel(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_ravel_multi_index() -> None:
    """Test ravel_multi_index."""
    res = mod.ravel_multi_index((mod.array([0, 1], dtype=mod.int32), mod.array([1, 2], dtype=mod.int32)), (2, 3))
    assert res.shape == (2,)


def test_real() -> None:
    """Test real."""
    res = mod.real(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_reciprocal() -> None:
    """Test reciprocal."""
    res = mod.reciprocal(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_remainder() -> None:
    """Test remainder."""
    res = mod.remainder(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_repeat() -> None:
    """Test repeat."""
    res = mod.repeat(mod.array([1.0, 2.0]), 2)
    assert res.shape == (4,)


def test_reshape() -> None:
    """Test reshape."""
    res = mod.reshape(mod.array([1.0, 2.0, 3.0, 4.0]), (2, 2))
    assert res.shape == (2, 2)


def test_resize() -> None:
    """Test resize."""
    with pytest.raises(TypeError):
        mod.resize(mod.array([1.0, 2.0]), (4,))


def test_result_type() -> None:
    """Test result_type."""
    res = mod.result_type(mod.float32, mod.int32)
    assert res.shape == ()


def test_right_shift() -> None:
    """Test right_shift."""
    res = mod.right_shift(mod.array([2, 4], dtype=mod.int32), mod.array([1, 1], dtype=mod.int32))
    assert res.shape == (2,)


def test_rint() -> None:
    """Test rint."""
    res = mod.rint(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_roll() -> None:
    """Test roll."""
    res = mod.roll(mod.array([1.0, 2.0, 3.0]), 1)
    assert res.shape == (3,)


def test_rollaxis() -> None:
    """Test rollaxis."""
    res = mod.rollaxis(mod.ones((2, 3, 4)), 2, 0)
    assert res.shape == (4, 2, 3)


def test_rot90() -> None:
    """Test rot90."""
    assert mod.rot90(mod.eye(2)).shape == (2, 2)


def test_round() -> None:
    """Test round."""
    res = mod.round(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_round_() -> None:
    """Test round_."""
    res = mod.round_(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_save() -> None:
    """Test save."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npy") as fp:
        mod.save(fp.name, mod.array([1.0])._tensor.data)
        assert os.path.exists(fp.name)


def test_savez() -> None:
    """Test savez."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npz") as fp:
        mod.savez(fp.name, mod.array([1.0])._tensor.data)
        assert os.path.exists(fp.name)


def test_searchsorted() -> None:
    """Test searchsorted."""
    res = mod.searchsorted(mod.array([1.0, 2.0, 3.0]), mod.array([1.5]))
    assert res.shape == (1,)


def test_select() -> None:
    """Test select."""
    res = mod.select([mod.array([True, False])], [mod.array([1.0, 2.0])], default=0.0)
    assert res.shape == (2,)


def test_setdiff1d() -> None:
    """Test setdiff1d."""
    res = mod.setdiff1d(mod.array([1, 2]), mod.array([2, 3]))
    assert res.shape == (1,)


def test_setxor1d() -> None:
    """Test setxor1d."""
    res = mod.setxor1d(mod.array([1, 2]), mod.array([2, 3]))
    assert res.shape == (2,)


def test_shape() -> None:
    """Test shape."""
    res = mod.shape(mod.array([0.5, 1.5]))
    assert len(res) == 1


def test_sign() -> None:
    """Test sign."""
    res = mod.sign(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_signbit() -> None:
    """Test signbit."""
    res = mod.signbit(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_class_signedinteger() -> None:
    """Test signedinteger."""
    obj = mod.signedinteger()
    assert isinstance(obj, mod.integer)


def test_sin() -> None:
    """Test sin."""
    res = mod.sin(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_sinc() -> None:
    """Test sinc."""
    res = mod.sinc(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_sinh() -> None:
    """Test sinh."""
    res = mod.sinh(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_size() -> None:
    """Test size."""
    res = mod.size(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_sort() -> None:
    """Test sort."""
    res = mod.sort(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_sort_complex() -> None:
    """Test sort_complex."""
    res = mod.sort_complex(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_split() -> None:
    """Test split."""
    res = mod.split(mod.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert len(res) == 2


def test_sqrt() -> None:
    """Test sqrt."""
    res = mod.sqrt(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_square() -> None:
    """Test square."""
    res = mod.square(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_squeeze() -> None:
    """Test squeeze."""
    res = mod.squeeze(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_stack() -> None:
    """Test stack."""
    res = mod.stack([mod.array([1.0, 2.0]), mod.array([3.0, 4.0])])
    assert res.shape == (2, 2)


def test_std() -> None:
    """Test std."""
    res = mod.std(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_subtract() -> None:
    """Test subtract."""
    res = mod.subtract(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_sum() -> None:
    """Test sum."""
    res = mod.sum(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_swapaxes() -> None:
    """Test swapaxes."""
    res = mod.swapaxes(mod.ones((2, 3)), 0, 1)
    assert res.shape == (3, 2)


def test_take() -> None:
    """Test take."""
    res = mod.take(mod.array([10.0, 20.0, 30.0]), mod.array([0, 2], dtype=mod.int32))
    assert res.shape == (2,)


def test_take_along_axis() -> None:
    """Test take_along_axis."""
    res = mod.take_along_axis(mod.array([10.0, 20.0, 30.0]), mod.array([0, 2], dtype=mod.int32), axis=0)
    assert res.shape == (2,)


def test_tan() -> None:
    """Test tan."""
    res = mod.tan(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_tanh() -> None:
    """Test tanh."""
    res = mod.tanh(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_tensordot() -> None:
    """Test tensordot."""
    res = mod.tensordot(mod.eye(2), mod.eye(2), axes=1)
    assert res.shape == (2, 2)


def test_tile() -> None:
    """Test tile."""
    res = mod.tile(mod.array([1.0, 2.0]), 2)
    assert res.shape == (4,)


def test_trace() -> None:
    """Test trace."""
    res = mod.trace(mod.eye(3))
    assert res.shape == ()


def test_transpose() -> None:
    """Test transpose."""
    res = mod.transpose(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_trapezoid() -> None:
    """Test trapezoid."""
    assert mod.trapezoid(mod.array([1.0, 2.0, 3.0])) == 4.0


def test_tri() -> None:
    """Test tri."""
    assert mod.tri(3).shape == (3, 3)


def test_tril() -> None:
    """Test tril."""
    res = mod.tril(mod.ones((3, 3)))
    assert res.shape == (3, 3)


def test_tril_indices() -> None:
    """Test tril_indices."""
    res = mod.tril_indices(3)
    assert len(res) == 2


def test_tril_indices_from() -> None:
    """Test tril_indices_from."""
    res = mod.tril_indices_from(mod.ones((3, 3)))
    assert len(res) == 2


def test_trim_zeros() -> None:
    """Test trim_zeros."""
    res = mod.trim_zeros(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_triu() -> None:
    """Test triu."""
    res = mod.triu(mod.ones((3, 3)))
    assert res.shape == (3, 3)


def test_triu_indices() -> None:
    """Test triu_indices."""
    res = mod.triu_indices(3)
    assert len(res) == 2


def test_triu_indices_from() -> None:
    """Test triu_indices_from."""
    res = mod.triu_indices_from(mod.ones((3, 3)))
    assert len(res) == 2


def test_true_divide() -> None:
    """Test true_divide."""
    res = mod.true_divide(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_trunc() -> None:
    """Test trunc."""
    res = mod.trunc(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_ufunc() -> None:
    """Test ufunc."""
    mod.ufunc()
    assert callable(mod.ufunc)


def test_uint() -> None:
    """Test uint."""
    res = mod.uint(3)
    assert res.shape == ()


def test_uint4() -> None:
    """Test uint4."""
    res = mod.uint4(3)
    assert res.shape == ()


def test_union1d() -> None:
    """Test union1d."""
    res = mod.union1d(mod.array([1, 2]), mod.array([2, 3]))
    assert res.shape == (3,)


def test_unique() -> None:
    """Test unique."""
    res = mod.unique(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_unique_all() -> None:
    """Test unique_all."""
    res = mod.unique_all(mod.array([0.5, 1.5]))
    assert len(res) == 4


def test_unique_counts() -> None:
    """Test unique_counts."""
    res = mod.unique_counts(mod.array([0.5, 1.5]))
    assert len(res) == 2


def test_unique_inverse() -> None:
    """Test unique_inverse."""
    res = mod.unique_inverse(mod.array([0.5, 1.5]))
    assert len(res) == 2


def test_unique_values() -> None:
    """Test unique_values."""
    res = mod.unique_values(mod.array([0.5, 1.5]))
    assert res.shape == (2,)


def test_unpackbits() -> None:
    """Test unpackbits."""
    res = mod.unpackbits(mod.array([160], dtype=mod.uint8))
    assert res.shape == (8,)


def test_unravel_index() -> None:
    """Test unravel_index."""
    res = mod.unravel_index(mod.array([1, 2], dtype=mod.int32), (2, 2))
    assert len(res) == 2


def test_class_unsignedinteger() -> None:
    """Test unsignedinteger."""
    obj = mod.unsignedinteger()
    assert isinstance(obj, mod.integer)


def test_unstack() -> None:
    """Test unstack."""
    res = mod.unstack(mod.ones((2, 3)))
    assert len(res) == 2


def test_unwrap() -> None:
    """Test unwrap."""
    assert mod.unwrap(mod.array([0.0, 3.14])).shape == (2,)


def test_vander() -> None:
    """Test vander."""
    assert mod.vander(mod.array([1.0, 2.0, 3.0]), N=2).shape == (3, 2)


def test_var() -> None:
    """Test var."""
    res = mod.var(mod.array([0.5, 1.5]))
    assert res.shape == ()


def test_vdot() -> None:
    """Test vdot."""
    res = mod.vdot(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == ()


def test_vecdot() -> None:
    """Test vecdot."""
    res = mod.vecdot(mod.array([0.5, 1.5]), mod.array([1.0, 2.0]))
    assert res.shape == ()


def test_class_vectorize() -> None:
    """Test vectorize."""
    vfn = mod.vectorize(lambda x: x + 1.0)
    res = vfn(mod.array([1.0, 2.0]))
    assert res.shape == (2,)


def test_vsplit() -> None:
    """Test vsplit."""
    res = mod.vsplit(mod.ones((4, 2)), 2)
    assert len(res) == 2


def test_vstack() -> None:
    """Test vstack."""
    res = mod.vstack([mod.array([1.0, 2.0]), mod.array([3.0, 4.0])])
    assert res.shape == (2, 2)


def test_where() -> None:
    """Test where."""
    res = mod.where(mod.array([True, False]), mod.array([1.0, 2.0]), mod.array([3.0, 4.0]))
    assert res.shape == (2,)


def test_zeros() -> None:
    """Test zeros."""
    res = mod.zeros((2, 3))
    assert res.shape == (2, 3)


def test_zeros_like() -> None:
    """Test zeros_like."""
    res = mod.zeros_like(mod.array([0.5, 1.5]))
    assert res.shape == (2,)
