"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.ad as mod


def test_CustomJVPException() -> None:
    """Test CustomJVPException."""
    obj = mod.CustomJVPException()
    assert obj is not None


def test_CustomVJPException() -> None:
    """Test CustomVJPException."""
    obj = mod.CustomVJPException()
    assert obj is not None


def test_JVPTrace() -> None:
    """Test JVPTrace."""
    obj = mod.JVPTrace()
    assert obj is not None


def test_JVPTracer() -> None:
    """Test JVPTracer."""
    obj = mod.JVPTracer()
    assert obj is not None


def test_UndefinedPrimal() -> None:
    """Test UndefinedPrimal."""
    obj = mod.UndefinedPrimal()
    assert obj is not None


def test_Zero() -> None:
    """Test Zero."""
    obj = mod.Zero()
    assert obj is not None


def test_add_jaxvals() -> None:
    """Test add_jaxvals."""
    with pytest.raises(NotImplementedError):
        mod.add_jaxvals()


def test_add_tangents() -> None:
    """Test add_tangents."""
    with pytest.raises(NotImplementedError):
        mod.add_tangents()


def test_backward_pass() -> None:
    """Test backward_pass."""
    with pytest.raises(NotImplementedError):
        mod.backward_pass()


def test_backward_pass_internal() -> None:
    """Test backward_pass_internal."""
    with pytest.raises(NotImplementedError):
        mod.backward_pass_internal()


def test_bilinear_transpose() -> None:
    """Test bilinear_transpose."""
    with pytest.raises(NotImplementedError):
        mod.bilinear_transpose()


def test_call_transpose() -> None:
    """Test call_transpose."""
    with pytest.raises(NotImplementedError):
        mod.call_transpose()


def test_closed_backward_pass() -> None:
    """Test closed_backward_pass."""
    with pytest.raises(NotImplementedError):
        mod.closed_backward_pass()


def test_defbilinear() -> None:
    """Test defbilinear."""
    with pytest.raises(NotImplementedError):
        mod.defbilinear()


def test_defjvp() -> None:
    """Test defjvp."""
    with pytest.raises(NotImplementedError):
        mod.defjvp()


def test_defjvp2() -> None:
    """Test defjvp2."""
    with pytest.raises(NotImplementedError):
        mod.defjvp2()


def test_defjvp_zero() -> None:
    """Test defjvp_zero."""
    with pytest.raises(NotImplementedError):
        mod.defjvp_zero()


def test_deflinear() -> None:
    """Test deflinear."""
    with pytest.raises(NotImplementedError):
        mod.deflinear()


def test_deflinear2() -> None:
    """Test deflinear2."""
    with pytest.raises(NotImplementedError):
        mod.deflinear2()


def test_f_jvp_traceable() -> None:
    """Test f_jvp_traceable."""
    with pytest.raises(NotImplementedError):
        mod.f_jvp_traceable()


def test_get_primitive_transpose() -> None:
    """Test get_primitive_transpose."""
    with pytest.raises(NotImplementedError):
        mod.get_primitive_transpose()


def test_instantiate_zeros() -> None:
    """Test instantiate_zeros."""
    with pytest.raises(NotImplementedError):
        mod.instantiate_zeros()


def test_is_undefined_primal() -> None:
    """Test is_undefined_primal."""
    with pytest.raises(NotImplementedError):
        mod.is_undefined_primal()


def test_jvp() -> None:
    """Test jvp."""
    with pytest.raises(NotImplementedError):
        mod.jvp()


def test_jvp_jaxpr() -> None:
    """Test jvp_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.jvp_jaxpr()


def test_jvp_subtrace() -> None:
    """Test jvp_subtrace."""
    with pytest.raises(NotImplementedError):
        mod.jvp_subtrace()


def test_jvp_subtrace_aux() -> None:
    """Test jvp_subtrace_aux."""
    with pytest.raises(NotImplementedError):
        mod.jvp_subtrace_aux()


def test_jvpfun() -> None:
    """Test jvpfun."""
    with pytest.raises(NotImplementedError):
        mod.jvpfun()


def test_linear_jvp() -> None:
    """Test linear_jvp."""
    with pytest.raises(NotImplementedError):
        mod.linear_jvp()


def test_linear_transpose() -> None:
    """Test linear_transpose."""
    with pytest.raises(NotImplementedError):
        mod.linear_transpose()


def test_linear_transpose2() -> None:
    """Test linear_transpose2."""
    with pytest.raises(NotImplementedError):
        mod.linear_transpose2()


def test_linearize() -> None:
    """Test linearize."""
    with pytest.raises(NotImplementedError):
        mod.linearize()


def test_map_transpose() -> None:
    """Test map_transpose."""
    with pytest.raises(NotImplementedError):
        mod.map_transpose()


def test_nonzero_outputs() -> None:
    """Test nonzero_outputs."""
    with pytest.raises(NotImplementedError):
        mod.nonzero_outputs()


def test_nonzero_tangent_outputs() -> None:
    """Test nonzero_tangent_outputs."""
    with pytest.raises(NotImplementedError):
        mod.nonzero_tangent_outputs()


def test_rearrange_binders() -> None:
    """Test rearrange_binders."""
    with pytest.raises(NotImplementedError):
        mod.rearrange_binders()


def test_recast_to_float0() -> None:
    """Test recast_to_float0."""
    with pytest.raises(NotImplementedError):
        mod.recast_to_float0()


def test_replace_float0s() -> None:
    """Test replace_float0s."""
    with pytest.raises(NotImplementedError):
        mod.replace_float0s()


def test_standard_jvp() -> None:
    """Test standard_jvp."""
    with pytest.raises(NotImplementedError):
        mod.standard_jvp()


def test_standard_jvp2() -> None:
    """Test standard_jvp2."""
    with pytest.raises(NotImplementedError):
        mod.standard_jvp2()


def test_traceable() -> None:
    """Test traceable."""
    with pytest.raises(NotImplementedError):
        mod.traceable()


def test_unpair_pval() -> None:
    """Test unpair_pval."""
    with pytest.raises(NotImplementedError):
        mod.unpair_pval()


def test_vjp() -> None:
    """Test vjp."""
    with pytest.raises(NotImplementedError):
        mod.vjp()


def test_zero_jvp() -> None:
    """Test zero_jvp."""
    with pytest.raises(NotImplementedError):
        mod.zero_jvp()


def test_zeros_like_aval() -> None:
    """Test zeros_like_aval."""
    with pytest.raises(NotImplementedError):
        mod.zeros_like_aval()


def test_zeros_like_jaxval() -> None:
    """Test zeros_like_jaxval."""
    with pytest.raises(NotImplementedError):
        mod.zeros_like_jaxval()
