"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.interpreters.batching as mod


def test_Array() -> None:
    """Test Array."""
    obj = mod.Array()
    assert obj is not None


def test_AxisSize() -> None:
    """Test AxisSize."""
    with patch("ml_switcheroo_compiler.ops.AxisSize") as mock_op:
        mod.AxisSize()
        mock_op.assert_called_once_with()


def test_BatchTrace() -> None:
    """Test BatchTrace."""
    obj = mod.BatchTrace()
    assert obj is not None


def test_BatchTracer() -> None:
    """Test BatchTracer."""
    obj = mod.BatchTracer()
    assert obj is not None


def test_BatchingRule() -> None:
    """Test BatchingRule."""
    with patch("ml_switcheroo_compiler.ops.BatchingRule") as mock_op:
        mod.BatchingRule()
        mock_op.assert_called_once_with()


def test_Elt() -> None:
    """Test Elt."""
    with patch("ml_switcheroo_compiler.ops.Elt") as mock_op:
        mod.Elt()
        mock_op.assert_called_once_with()


def test_FromEltHandler() -> None:
    """Test FromEltHandler."""
    with patch("ml_switcheroo_compiler.ops.FromEltHandler") as mock_op:
        mod.FromEltHandler()
        mock_op.assert_called_once_with()


def test_GetIdx() -> None:
    """Test GetIdx."""
    with patch("ml_switcheroo_compiler.ops.GetIdx") as mock_op:
        mod.GetIdx()
        mock_op.assert_called_once_with()


def test_IndexedAxisSize() -> None:
    """Test IndexedAxisSize."""
    obj = mod.IndexedAxisSize()
    assert obj is not None


def test_Jumble() -> None:
    """Test Jumble."""
    obj = mod.Jumble()
    assert obj is not None


def test_JumbleAxis() -> None:
    """Test JumbleAxis."""
    obj = mod.JumbleAxis()
    assert obj is not None


def test_JumbleTy() -> None:
    """Test JumbleTy."""
    obj = mod.JumbleTy()
    assert obj is not None


def test_MakeIotaHandler() -> None:
    """Test MakeIotaHandler."""
    with patch("ml_switcheroo_compiler.ops.MakeIotaHandler") as mock_op:
        mod.MakeIotaHandler()
        mock_op.assert_called_once_with()


def test_MapSpec() -> None:
    """Test MapSpec."""
    with patch("ml_switcheroo_compiler.ops.MapSpec") as mock_op:
        mod.MapSpec()
        mock_op.assert_called_once_with()


def test_NotMapped() -> None:
    """Test NotMapped."""
    obj = mod.NotMapped()
    assert obj is not None


def test_RaggedAxis() -> None:
    """Test RaggedAxis."""
    obj = mod.RaggedAxis()
    assert obj is not None


def test_ToEltHandler() -> None:
    """Test ToEltHandler."""
    with patch("ml_switcheroo_compiler.ops.ToEltHandler") as mock_op:
        mod.ToEltHandler()
        mock_op.assert_called_once_with()


def test_Vmappable() -> None:
    """Test Vmappable."""
    with patch("ml_switcheroo_compiler.ops.Vmappable") as mock_op:
        mod.Vmappable()
        mock_op.assert_called_once_with()


def test_Zero() -> None:
    """Test Zero."""
    obj = mod.Zero()
    assert obj is not None


def test_ZeroIfMapped() -> None:
    """Test ZeroIfMapped."""
    obj = mod.ZeroIfMapped()
    assert obj is not None


def test_batch() -> None:
    """Test batch."""
    with patch("ml_switcheroo_compiler.ops.batch") as mock_op:
        mod.batch()
        mock_op.assert_called_once_with()


def test_batch_custom_jvp_subtrace() -> None:
    """Test batch_custom_jvp_subtrace."""
    with patch("ml_switcheroo_compiler.ops.batch_custom_jvp_subtrace") as mock_op:
        mod.batch_custom_jvp_subtrace()
        mock_op.assert_called_once_with()


def test_batch_custom_vjp_bwd() -> None:
    """Test batch_custom_vjp_bwd."""
    with patch("ml_switcheroo_compiler.ops.batch_custom_vjp_bwd") as mock_op:
        mod.batch_custom_vjp_bwd()
        mock_op.assert_called_once_with()


def test_batch_jaxpr() -> None:
    """Test batch_jaxpr."""
    with patch("ml_switcheroo_compiler.ops.batch_jaxpr") as mock_op:
        mod.batch_jaxpr()
        mock_op.assert_called_once_with()


def test_batch_jaxpr2() -> None:
    """Test batch_jaxpr2."""
    with patch("ml_switcheroo_compiler.ops.batch_jaxpr2") as mock_op:
        mod.batch_jaxpr2()
        mock_op.assert_called_once_with()


def test_batch_jaxpr_axes() -> None:
    """Test batch_jaxpr_axes."""
    with patch("ml_switcheroo_compiler.ops.batch_jaxpr_axes") as mock_op:
        mod.batch_jaxpr_axes()
        mock_op.assert_called_once_with()


def test_batch_subtrace() -> None:
    """Test batch_subtrace."""
    with patch("ml_switcheroo_compiler.ops.batch_subtrace") as mock_op:
        mod.batch_subtrace()
        mock_op.assert_called_once_with()


def test_bdim_at_front() -> None:
    """Test bdim_at_front."""
    with patch("ml_switcheroo_compiler.ops.bdim_at_front") as mock_op:
        mod.bdim_at_front()
        mock_op.assert_called_once_with()


def test_broadcast() -> None:
    """Test broadcast."""
    with patch("ml_switcheroo_compiler.ops.broadcast") as mock_op:
        mod.broadcast()
        mock_op.assert_called_once_with()


def test_broadcast_batcher() -> None:
    """Test broadcast_batcher."""
    with patch("ml_switcheroo_compiler.ops.broadcast_batcher") as mock_op:
        mod.broadcast_batcher()
        mock_op.assert_called_once_with()


def test_defbroadcasting() -> None:
    """Test defbroadcasting."""
    with patch("ml_switcheroo_compiler.ops.defbroadcasting") as mock_op:
        mod.defbroadcasting()
        mock_op.assert_called_once_with()


def test_defreducer() -> None:
    """Test defreducer."""
    with patch("ml_switcheroo_compiler.ops.defreducer") as mock_op:
        mod.defreducer()
        mock_op.assert_called_once_with()


def test_defvectorized() -> None:
    """Test defvectorized."""
    with patch("ml_switcheroo_compiler.ops.defvectorized") as mock_op:
        mod.defvectorized()
        mock_op.assert_called_once_with()


def test_flatten_fun_for_vmap() -> None:
    """Test flatten_fun_for_vmap."""
    with patch("ml_switcheroo_compiler.ops.flatten_fun_for_vmap") as mock_op:
        mod.flatten_fun_for_vmap()
        mock_op.assert_called_once_with()


def test_from_elt() -> None:
    """Test from_elt."""
    with patch("ml_switcheroo_compiler.ops.from_elt") as mock_op:
        mod.from_elt()
        mock_op.assert_called_once_with()


def test_is_vmappable() -> None:
    """Test is_vmappable."""
    with patch("ml_switcheroo_compiler.ops.is_vmappable") as mock_op:
        mod.is_vmappable()
        mock_op.assert_called_once_with()


def test_make_iota() -> None:
    """Test make_iota."""
    with patch("ml_switcheroo_compiler.ops.make_iota") as mock_op:
        mod.make_iota()
        mock_op.assert_called_once_with()


def test_matchaxis() -> None:
    """Test matchaxis."""
    with patch("ml_switcheroo_compiler.ops.matchaxis") as mock_op:
        mod.matchaxis()
        mock_op.assert_called_once_with()


def test_moveaxis() -> None:
    """Test moveaxis."""
    with patch("ml_switcheroo_compiler.ops.moveaxis") as mock_op:
        mod.moveaxis()
        mock_op.assert_called_once_with()


def test_reducer_batcher() -> None:
    """Test reducer_batcher."""
    with patch("ml_switcheroo_compiler.ops.reducer_batcher") as mock_op:
        mod.reducer_batcher()
        mock_op.assert_called_once_with()


def test_register_vmappable() -> None:
    """Test register_vmappable."""
    with patch("ml_switcheroo_compiler.ops.register_vmappable") as mock_op:
        mod.register_vmappable()
        mock_op.assert_called_once_with()


def test_to_elt() -> None:
    """Test to_elt."""
    with patch("ml_switcheroo_compiler.ops.to_elt") as mock_op:
        mod.to_elt()
        mock_op.assert_called_once_with()


def test_unregister_vmappable() -> None:
    """Test unregister_vmappable."""
    with patch("ml_switcheroo_compiler.ops.unregister_vmappable") as mock_op:
        mod.unregister_vmappable()
        mock_op.assert_called_once_with()


def test_vectorized_batcher() -> None:
    """Test vectorized_batcher."""
    with patch("ml_switcheroo_compiler.ops.vectorized_batcher") as mock_op:
        mod.vectorized_batcher()
        mock_op.assert_called_once_with()


def test_vtile() -> None:
    """Test vtile."""
    with patch("ml_switcheroo_compiler.ops.vtile") as mock_op:
        mod.vtile()
        mock_op.assert_called_once_with()
