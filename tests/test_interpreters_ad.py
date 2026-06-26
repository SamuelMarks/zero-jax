"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
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
    with patch("ml_switcheroo_compiler.ops.add_jaxvals") as mock_op:
        mod.add_jaxvals()
        mock_op.assert_called_once_with()


def test_add_tangents() -> None:
    """Test add_tangents."""
    with patch("ml_switcheroo_compiler.ops.add_tangents") as mock_op:
        mod.add_tangents()
        mock_op.assert_called_once_with()


def test_backward_pass() -> None:
    """Test backward_pass."""
    with patch("ml_switcheroo_compiler.ops.backward_pass") as mock_op:
        mod.backward_pass()
        mock_op.assert_called_once_with()


def test_backward_pass_internal() -> None:
    """Test backward_pass_internal."""
    with patch("ml_switcheroo_compiler.ops.backward_pass_internal") as mock_op:
        mod.backward_pass_internal()
        mock_op.assert_called_once_with()


def test_bilinear_transpose() -> None:
    """Test bilinear_transpose."""
    with patch("ml_switcheroo_compiler.ops.bilinear_transpose") as mock_op:
        mod.bilinear_transpose()
        mock_op.assert_called_once_with()


def test_call_transpose() -> None:
    """Test call_transpose."""
    with patch("ml_switcheroo_compiler.ops.call_transpose") as mock_op:
        mod.call_transpose()
        mock_op.assert_called_once_with()


def test_closed_backward_pass() -> None:
    """Test closed_backward_pass."""
    with patch("ml_switcheroo_compiler.ops.closed_backward_pass") as mock_op:
        mod.closed_backward_pass()
        mock_op.assert_called_once_with()


def test_defbilinear() -> None:
    """Test defbilinear."""
    with patch("ml_switcheroo_compiler.ops.defbilinear") as mock_op:
        mod.defbilinear()
        mock_op.assert_called_once_with()


def test_defjvp() -> None:
    """Test defjvp."""
    with patch("ml_switcheroo_compiler.ops.defjvp") as mock_op:
        mod.defjvp()
        mock_op.assert_called_once_with()


def test_defjvp2() -> None:
    """Test defjvp2."""
    with patch("ml_switcheroo_compiler.ops.defjvp2") as mock_op:
        mod.defjvp2()
        mock_op.assert_called_once_with()


def test_defjvp_zero() -> None:
    """Test defjvp_zero."""
    with patch("ml_switcheroo_compiler.ops.defjvp_zero") as mock_op:
        mod.defjvp_zero()
        mock_op.assert_called_once_with()


def test_deflinear() -> None:
    """Test deflinear."""
    with patch("ml_switcheroo_compiler.ops.deflinear") as mock_op:
        mod.deflinear()
        mock_op.assert_called_once_with()


def test_deflinear2() -> None:
    """Test deflinear2."""
    with patch("ml_switcheroo_compiler.ops.deflinear2") as mock_op:
        mod.deflinear2()
        mock_op.assert_called_once_with()


def test_f_jvp_traceable() -> None:
    """Test f_jvp_traceable."""
    with patch("ml_switcheroo_compiler.ops.f_jvp_traceable") as mock_op:
        mod.f_jvp_traceable()
        mock_op.assert_called_once_with()


def test_get_primitive_transpose() -> None:
    """Test get_primitive_transpose."""
    with patch("ml_switcheroo_compiler.ops.get_primitive_transpose") as mock_op:
        mod.get_primitive_transpose()
        mock_op.assert_called_once_with()


def test_instantiate_zeros() -> None:
    """Test instantiate_zeros."""
    with patch("ml_switcheroo_compiler.ops.instantiate_zeros") as mock_op:
        mod.instantiate_zeros()
        mock_op.assert_called_once_with()


def test_is_undefined_primal() -> None:
    """Test is_undefined_primal."""
    with patch("ml_switcheroo_compiler.ops.is_undefined_primal") as mock_op:
        mod.is_undefined_primal()
        mock_op.assert_called_once_with()


def test_jvp() -> None:
    """Test jvp."""
    with patch("ml_switcheroo_compiler.ops.jvp") as mock_op:
        mod.jvp()
        mock_op.assert_called_once_with()


def test_jvp_jaxpr() -> None:
    """Test jvp_jaxpr."""
    with patch("ml_switcheroo_compiler.ops.jvp_jaxpr") as mock_op:
        mod.jvp_jaxpr()
        mock_op.assert_called_once_with()


def test_jvp_subtrace() -> None:
    """Test jvp_subtrace."""
    with patch("ml_switcheroo_compiler.ops.jvp_subtrace") as mock_op:
        mod.jvp_subtrace()
        mock_op.assert_called_once_with()


def test_jvp_subtrace_aux() -> None:
    """Test jvp_subtrace_aux."""
    with patch("ml_switcheroo_compiler.ops.jvp_subtrace_aux") as mock_op:
        mod.jvp_subtrace_aux()
        mock_op.assert_called_once_with()


def test_jvpfun() -> None:
    """Test jvpfun."""
    with patch("ml_switcheroo_compiler.ops.jvpfun") as mock_op:
        mod.jvpfun()
        mock_op.assert_called_once_with()


def test_linear_jvp() -> None:
    """Test linear_jvp."""
    with patch("ml_switcheroo_compiler.ops.linear_jvp") as mock_op:
        mod.linear_jvp()
        mock_op.assert_called_once_with()


def test_linear_transpose() -> None:
    """Test linear_transpose."""
    with patch("ml_switcheroo_compiler.ops.linear_transpose") as mock_op:
        mod.linear_transpose()
        mock_op.assert_called_once_with()


def test_linear_transpose2() -> None:
    """Test linear_transpose2."""
    with patch("ml_switcheroo_compiler.ops.linear_transpose2") as mock_op:
        mod.linear_transpose2()
        mock_op.assert_called_once_with()


def test_linearize() -> None:
    """Test linearize."""
    with patch("ml_switcheroo_compiler.ops.linearize") as mock_op:
        mod.linearize()
        mock_op.assert_called_once_with()


def test_map_transpose() -> None:
    """Test map_transpose."""
    with patch("ml_switcheroo_compiler.ops.map_transpose") as mock_op:
        mod.map_transpose()
        mock_op.assert_called_once_with()


def test_nonzero_outputs() -> None:
    """Test nonzero_outputs."""
    with patch("ml_switcheroo_compiler.ops.nonzero_outputs") as mock_op:
        mod.nonzero_outputs()
        mock_op.assert_called_once_with()


def test_nonzero_tangent_outputs() -> None:
    """Test nonzero_tangent_outputs."""
    with patch("ml_switcheroo_compiler.ops.nonzero_tangent_outputs") as mock_op:
        mod.nonzero_tangent_outputs()
        mock_op.assert_called_once_with()


def test_rearrange_binders() -> None:
    """Test rearrange_binders."""
    with patch("ml_switcheroo_compiler.ops.rearrange_binders") as mock_op:
        mod.rearrange_binders()
        mock_op.assert_called_once_with()


def test_recast_to_float0() -> None:
    """Test recast_to_float0."""
    with patch("ml_switcheroo_compiler.ops.recast_to_float0") as mock_op:
        mod.recast_to_float0()
        mock_op.assert_called_once_with()


def test_replace_float0s() -> None:
    """Test replace_float0s."""
    with patch("ml_switcheroo_compiler.ops.replace_float0s") as mock_op:
        mod.replace_float0s()
        mock_op.assert_called_once_with()


def test_standard_jvp() -> None:
    """Test standard_jvp."""
    with patch("ml_switcheroo_compiler.ops.standard_jvp") as mock_op:
        mod.standard_jvp()
        mock_op.assert_called_once_with()


def test_standard_jvp2() -> None:
    """Test standard_jvp2."""
    with patch("ml_switcheroo_compiler.ops.standard_jvp2") as mock_op:
        mod.standard_jvp2()
        mock_op.assert_called_once_with()


def test_traceable() -> None:
    """Test traceable."""
    with patch("ml_switcheroo_compiler.ops.traceable") as mock_op:
        mod.traceable()
        mock_op.assert_called_once_with()


def test_unpair_pval() -> None:
    """Test unpair_pval."""
    with patch("ml_switcheroo_compiler.ops.unpair_pval") as mock_op:
        mod.unpair_pval()
        mock_op.assert_called_once_with()


def test_vjp() -> None:
    """Test vjp."""
    with patch("ml_switcheroo_compiler.ops.vjp") as mock_op:
        mod.vjp()
        mock_op.assert_called_once_with()


def test_zero_jvp() -> None:
    """Test zero_jvp."""
    with patch("ml_switcheroo_compiler.ops.zero_jvp") as mock_op:
        mod.zero_jvp()
        mock_op.assert_called_once_with()


def test_zeros_like_aval() -> None:
    """Test zeros_like_aval."""
    with patch("ml_switcheroo_compiler.ops.zeros_like_aval") as mock_op:
        mod.zeros_like_aval()
        mock_op.assert_called_once_with()


def test_zeros_like_jaxval() -> None:
    """Test zeros_like_jaxval."""
    with patch("ml_switcheroo_compiler.ops.zeros_like_jaxval") as mock_op:
        mod.zeros_like_jaxval()
        mock_op.assert_called_once_with()
