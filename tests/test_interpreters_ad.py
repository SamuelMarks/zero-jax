"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.interpreters.ad as mod


def test_CustomJVPException() -> None:
    """Test CustomJVPException."""
    obj = mod.CustomJVPException("test message")
    assert obj is not None
    assert obj.msg == "test message"


def test_CustomVJPException() -> None:
    """Test CustomVJPException."""
    obj = mod.CustomVJPException("test message")
    assert obj is not None
    assert obj.msg == "test message"


def test_JVPTrace() -> None:
    """Test JVPTrace."""
    obj = mod.JVPTrace(id=1, name="test_JVPTrace", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_JVPTrace"
    assert obj.value == "test_value"


def test_JVPTracer() -> None:
    """Test JVPTracer."""
    obj = mod.JVPTracer(id=1, name="test_JVPTracer", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_JVPTracer"
    assert obj.value == "test_value"


def test_UndefinedPrimal() -> None:
    """Test UndefinedPrimal."""
    obj = mod.UndefinedPrimal(id=1, name="test_UndefinedPrimal", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_UndefinedPrimal"
    assert obj.value == "test_value"


def test_Zero() -> None:
    """Test Zero."""
    obj = mod.Zero(id=1, name="test_Zero", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Zero"
    assert obj.value == "test_value"


def test_add_jaxvals() -> None:
    """Test add_jaxvals."""
    with patch("zero_jax._compiler_proxy_ops.add_jaxvals", create=True) as mock_op:
        mod.add_jaxvals()
        mock_op.assert_called_once_with()


def test_add_tangents() -> None:
    """Test add_tangents."""
    with patch("zero_jax._compiler_proxy_ops.add_tangents", create=True) as mock_op:
        mod.add_tangents()
        mock_op.assert_called_once_with()


def test_backward_pass() -> None:
    """Test backward_pass."""
    with patch("zero_jax._compiler_proxy_ops.backward_pass", create=True) as mock_op:
        mod.backward_pass()
        mock_op.assert_called_once_with()


def test_backward_pass_internal() -> None:
    """Test backward_pass_internal."""
    with patch(
        "zero_jax._compiler_proxy_ops.backward_pass_internal", create=True
    ) as mock_op:
        mod.backward_pass_internal()
        mock_op.assert_called_once_with()


def test_bilinear_transpose() -> None:
    """Test bilinear_transpose."""
    with patch(
        "zero_jax._compiler_proxy_ops.bilinear_transpose", create=True
    ) as mock_op:
        mod.bilinear_transpose()
        mock_op.assert_called_once_with()


def test_call_transpose() -> None:
    """Test call_transpose."""
    with patch("zero_jax._compiler_proxy_ops.call_transpose", create=True) as mock_op:
        mod.call_transpose()
        mock_op.assert_called_once_with()


def test_closed_backward_pass() -> None:
    """Test closed_backward_pass."""
    with patch(
        "zero_jax._compiler_proxy_ops.closed_backward_pass", create=True
    ) as mock_op:
        mod.closed_backward_pass()
        mock_op.assert_called_once_with()


def test_defbilinear() -> None:
    """Test defbilinear."""
    with patch("zero_jax._compiler_proxy_ops.defbilinear", create=True) as mock_op:
        mod.defbilinear()
        mock_op.assert_called_once_with()


def test_defjvp() -> None:
    """Test defjvp."""
    with patch("zero_jax._compiler_proxy_ops.defjvp", create=True) as mock_op:
        mod.defjvp()
        mock_op.assert_called_once_with()


def test_defjvp2() -> None:
    """Test defjvp2."""
    with patch("zero_jax._compiler_proxy_ops.defjvp2", create=True) as mock_op:
        mod.defjvp2()
        mock_op.assert_called_once_with()


def test_defjvp_zero() -> None:
    """Test defjvp_zero."""
    with patch("zero_jax._compiler_proxy_ops.defjvp_zero", create=True) as mock_op:
        mod.defjvp_zero()
        mock_op.assert_called_once_with()


def test_deflinear() -> None:
    """Test deflinear."""
    with patch("zero_jax._compiler_proxy_ops.deflinear", create=True) as mock_op:
        mod.deflinear()
        mock_op.assert_called_once_with()


def test_deflinear2() -> None:
    """Test deflinear2."""
    with patch("zero_jax._compiler_proxy_ops.deflinear2", create=True) as mock_op:
        mod.deflinear2()
        mock_op.assert_called_once_with()


def test_f_jvp_traceable() -> None:
    """Test f_jvp_traceable."""
    with patch("zero_jax._compiler_proxy_ops.f_jvp_traceable", create=True) as mock_op:
        mod.f_jvp_traceable()
        mock_op.assert_called_once_with()


def test_get_primitive_transpose() -> None:
    """Test get_primitive_transpose."""
    with patch(
        "zero_jax._compiler_proxy_ops.get_primitive_transpose", create=True
    ) as mock_op:
        mod.get_primitive_transpose()
        mock_op.assert_called_once_with()


def test_instantiate_zeros() -> None:
    """Test instantiate_zeros."""
    with patch(
        "zero_jax._compiler_proxy_ops.instantiate_zeros", create=True
    ) as mock_op:
        mod.instantiate_zeros()
        mock_op.assert_called_once_with()


def test_is_undefined_primal() -> None:
    """Test is_undefined_primal."""
    with patch(
        "zero_jax._compiler_proxy_ops.is_undefined_primal", create=True
    ) as mock_op:
        mod.is_undefined_primal()
        mock_op.assert_called_once_with()


def test_jvp() -> None:
    """Test jvp."""
    with patch("zero_jax._compiler_proxy_ops.jvp", create=True) as mock_op:
        mod.jvp()
        mock_op.assert_called_once_with()


def test_jvp_jaxpr() -> None:
    """Test jvp_jaxpr."""
    with patch("zero_jax._compiler_proxy_ops.jvp_jaxpr", create=True) as mock_op:
        mod.jvp_jaxpr()
        mock_op.assert_called_once_with()


def test_jvp_subtrace() -> None:
    """Test jvp_subtrace."""
    with patch("zero_jax._compiler_proxy_ops.jvp_subtrace", create=True) as mock_op:
        mod.jvp_subtrace()
        mock_op.assert_called_once_with()


def test_jvp_subtrace_aux() -> None:
    """Test jvp_subtrace_aux."""
    with patch("zero_jax._compiler_proxy_ops.jvp_subtrace_aux", create=True) as mock_op:
        mod.jvp_subtrace_aux()
        mock_op.assert_called_once_with()


def test_jvpfun() -> None:
    """Test jvpfun."""
    with patch("zero_jax._compiler_proxy_ops.jvpfun", create=True) as mock_op:
        mod.jvpfun()
        mock_op.assert_called_once_with()


def test_linear_jvp() -> None:
    """Test linear_jvp."""
    with patch("zero_jax._compiler_proxy_ops.linear_jvp", create=True) as mock_op:
        mod.linear_jvp()
        mock_op.assert_called_once_with()


def test_linear_transpose() -> None:
    """Test linear_transpose."""
    with patch("zero_jax._compiler_proxy_ops.linear_transpose", create=True) as mock_op:
        mod.linear_transpose()
        mock_op.assert_called_once_with()


def test_linear_transpose2() -> None:
    """Test linear_transpose2."""
    with patch(
        "zero_jax._compiler_proxy_ops.linear_transpose2", create=True
    ) as mock_op:
        mod.linear_transpose2()
        mock_op.assert_called_once_with()


def test_linearize() -> None:
    """Test linearize."""
    with patch("zero_jax._compiler_proxy_ops.linearize", create=True) as mock_op:
        mod.linearize()
        mock_op.assert_called_once_with()


def test_map_transpose() -> None:
    """Test map_transpose."""
    with patch("zero_jax._compiler_proxy_ops.map_transpose", create=True) as mock_op:
        mod.map_transpose()
        mock_op.assert_called_once_with()


def test_nonzero_outputs() -> None:
    """Test nonzero_outputs."""
    with patch("zero_jax._compiler_proxy_ops.nonzero_outputs", create=True) as mock_op:
        mod.nonzero_outputs()
        mock_op.assert_called_once_with()


def test_nonzero_tangent_outputs() -> None:
    """Test nonzero_tangent_outputs."""
    with patch(
        "zero_jax._compiler_proxy_ops.nonzero_tangent_outputs", create=True
    ) as mock_op:
        mod.nonzero_tangent_outputs()
        mock_op.assert_called_once_with()


def test_rearrange_binders() -> None:
    """Test rearrange_binders."""
    with patch(
        "zero_jax._compiler_proxy_ops.rearrange_binders", create=True
    ) as mock_op:
        mod.rearrange_binders()
        mock_op.assert_called_once_with()


def test_recast_to_float0() -> None:
    """Test recast_to_float0."""
    with patch("zero_jax._compiler_proxy_ops.recast_to_float0", create=True) as mock_op:
        mod.recast_to_float0()
        mock_op.assert_called_once_with()


def test_replace_float0s() -> None:
    """Test replace_float0s."""
    with patch("zero_jax._compiler_proxy_ops.replace_float0s", create=True) as mock_op:
        mod.replace_float0s()
        mock_op.assert_called_once_with()


def test_standard_jvp() -> None:
    """Test standard_jvp."""
    with patch("zero_jax._compiler_proxy_ops.standard_jvp", create=True) as mock_op:
        mod.standard_jvp()
        mock_op.assert_called_once_with()


def test_standard_jvp2() -> None:
    """Test standard_jvp2."""
    with patch("zero_jax._compiler_proxy_ops.standard_jvp2", create=True) as mock_op:
        mod.standard_jvp2()
        mock_op.assert_called_once_with()


def test_traceable() -> None:
    """Test traceable."""
    with patch("zero_jax._compiler_proxy_ops.traceable", create=True) as mock_op:
        mod.traceable()
        mock_op.assert_called_once_with()


def test_unpair_pval() -> None:
    """Test unpair_pval."""
    with patch("zero_jax._compiler_proxy_ops.unpair_pval", create=True) as mock_op:
        mod.unpair_pval()
        mock_op.assert_called_once_with()


def test_vjp() -> None:
    """Test vjp."""
    with patch("zero_jax._compiler_proxy_ops.vjp", create=True) as mock_op:
        mod.vjp()
        mock_op.assert_called_once_with()


def test_zero_jvp() -> None:
    """Test zero_jvp."""
    with patch("zero_jax._compiler_proxy_ops.zero_jvp", create=True) as mock_op:
        mod.zero_jvp()
        mock_op.assert_called_once_with()


def test_zeros_like_aval() -> None:
    """Test zeros_like_aval."""
    with patch("zero_jax._compiler_proxy_ops.zeros_like_aval", create=True) as mock_op:
        mod.zeros_like_aval()
        mock_op.assert_called_once_with()


def test_zeros_like_jaxval() -> None:
    """Test zeros_like_jaxval."""
    with patch(
        "zero_jax._compiler_proxy_ops.zeros_like_jaxval", create=True
    ) as mock_op:
        mod.zeros_like_jaxval()
        mock_op.assert_called_once_with()
