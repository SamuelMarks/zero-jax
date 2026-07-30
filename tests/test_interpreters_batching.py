"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.interpreters.batching as mod


def test_Array() -> None:
    """Test Array."""
    obj = mod.Array(id=1, name="test_Array", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Array"
    assert obj.value == "test_value"


def test_AxisSize() -> None:
    """Test AxisSize."""
    with patch("zero_jax._compiler_proxy_ops.AxisSize", create=True) as mock_op:
        mod.AxisSize()
        mock_op.assert_called_once_with()


def test_BatchTrace() -> None:
    """Test BatchTrace."""
    obj = mod.BatchTrace(id=1, name="test_BatchTrace", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_BatchTrace"
    assert obj.value == "test_value"


def test_BatchTracer() -> None:
    """Test BatchTracer."""
    obj = mod.BatchTracer(id=1, name="test_BatchTracer", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_BatchTracer"
    assert obj.value == "test_value"


def test_BatchingRule() -> None:
    """Test BatchingRule."""
    with patch("zero_jax._compiler_proxy_ops.BatchingRule", create=True) as mock_op:
        mod.BatchingRule()
        mock_op.assert_called_once_with()


def test_Elt() -> None:
    """Test Elt."""
    with patch("zero_jax._compiler_proxy_ops.Elt", create=True) as mock_op:
        mod.Elt()
        mock_op.assert_called_once_with()


def test_FromEltHandler() -> None:
    """Test FromEltHandler."""
    with patch("zero_jax._compiler_proxy_ops.FromEltHandler", create=True) as mock_op:
        mod.FromEltHandler()
        mock_op.assert_called_once_with()


def test_GetIdx() -> None:
    """Test GetIdx."""
    with patch("zero_jax._compiler_proxy_ops.GetIdx", create=True) as mock_op:
        mod.GetIdx()
        mock_op.assert_called_once_with()


def test_IndexedAxisSize() -> None:
    """Test IndexedAxisSize."""
    obj = mod.IndexedAxisSize(id=1, name="test_IndexedAxisSize", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_IndexedAxisSize"
    assert obj.value == "test_value"


def test_Jumble() -> None:
    """Test Jumble."""
    obj = mod.Jumble(id=1, name="test_Jumble", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Jumble"
    assert obj.value == "test_value"


def test_JumbleAxis() -> None:
    """Test JumbleAxis."""
    obj = mod.JumbleAxis(id=1, name="test_JumbleAxis", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_JumbleAxis"
    assert obj.value == "test_value"


def test_JumbleTy() -> None:
    """Test JumbleTy."""
    obj = mod.JumbleTy(id=1, name="test_JumbleTy", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_JumbleTy"
    assert obj.value == "test_value"


def test_MakeIotaHandler() -> None:
    """Test MakeIotaHandler."""
    with patch("zero_jax._compiler_proxy_ops.MakeIotaHandler", create=True) as mock_op:
        mod.MakeIotaHandler()
        mock_op.assert_called_once_with()


def test_MapSpec() -> None:
    """Test MapSpec."""
    with patch("zero_jax._compiler_proxy_ops.MapSpec", create=True) as mock_op:
        mod.MapSpec()
        mock_op.assert_called_once_with()


def test_not_mapped() -> None:
    """Test not_mapped instance."""
    assert mod.not_mapped is not None


def test_primitive_batchers() -> None:
    """Test primitive_batchers."""
    assert mod.primitive_batchers() is None


def test_NotMapped() -> None:
    """Test NotMapped."""
    obj = mod.NotMapped(id=1, name="test_NotMapped", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_NotMapped"
    assert obj.value == "test_value"


def test_RaggedAxis() -> None:
    """Test RaggedAxis."""
    obj = mod.RaggedAxis(id=1, name="test_RaggedAxis", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_RaggedAxis"
    assert obj.value == "test_value"


def test_ToEltHandler() -> None:
    """Test ToEltHandler."""
    with patch("zero_jax._compiler_proxy_ops.ToEltHandler", create=True) as mock_op:
        mod.ToEltHandler()
        mock_op.assert_called_once_with()


def test_Vmappable() -> None:
    """Test Vmappable."""
    with patch("zero_jax._compiler_proxy_ops.Vmappable", create=True) as mock_op:
        mod.Vmappable()
        mock_op.assert_called_once_with()


def test_Zero() -> None:
    """Test Zero."""
    obj = mod.Zero(id=1, name="test_Zero", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Zero"
    assert obj.value == "test_value"


def test_ZeroIfMapped() -> None:
    """Test ZeroIfMapped."""
    obj = mod.ZeroIfMapped(id=1, name="test_ZeroIfMapped", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_ZeroIfMapped"
    assert obj.value == "test_value"


def test_batch() -> None:
    """Test batch."""
    with patch("zero_jax._compiler_proxy_ops.batch", create=True) as mock_op:
        mod.batch()
        mock_op.assert_called_once_with()


def test_batch_custom_jvp_subtrace() -> None:
    """Test batch_custom_jvp_subtrace."""
    with patch(
        "zero_jax._compiler_proxy_ops.batch_custom_jvp_subtrace", create=True
    ) as mock_op:
        mod.batch_custom_jvp_subtrace()
        mock_op.assert_called_once_with()


def test_batch_custom_vjp_bwd() -> None:
    """Test batch_custom_vjp_bwd."""
    with patch(
        "zero_jax._compiler_proxy_ops.batch_custom_vjp_bwd", create=True
    ) as mock_op:
        mod.batch_custom_vjp_bwd()
        mock_op.assert_called_once_with()


def test_batch_jaxpr() -> None:
    """Test batch_jaxpr."""
    with patch("zero_jax._compiler_proxy_ops.batch_jaxpr", create=True) as mock_op:
        mod.batch_jaxpr()
        mock_op.assert_called_once_with()


def test_batch_jaxpr2() -> None:
    """Test batch_jaxpr2."""
    with patch("zero_jax._compiler_proxy_ops.batch_jaxpr2", create=True) as mock_op:
        mod.batch_jaxpr2()
        mock_op.assert_called_once_with()


def test_batch_jaxpr_axes() -> None:
    """Test batch_jaxpr_axes."""
    with patch("zero_jax._compiler_proxy_ops.batch_jaxpr_axes", create=True) as mock_op:
        mod.batch_jaxpr_axes()
        mock_op.assert_called_once_with()


def test_batch_subtrace() -> None:
    """Test batch_subtrace."""
    with patch("zero_jax._compiler_proxy_ops.batch_subtrace", create=True) as mock_op:
        mod.batch_subtrace()
        mock_op.assert_called_once_with()


def test_bdim_at_front() -> None:
    """Test bdim_at_front."""
    with patch("zero_jax._compiler_proxy_ops.bdim_at_front", create=True) as mock_op:
        mod.bdim_at_front()
        mock_op.assert_called_once_with()


def test_broadcast() -> None:
    """Test broadcast."""
    with patch("zero_jax._compiler_proxy_ops.broadcast", create=True) as mock_op:
        mod.broadcast()
        mock_op.assert_called_once_with()


def test_broadcast_batcher() -> None:
    """Test broadcast_batcher."""
    with patch(
        "zero_jax._compiler_proxy_ops.broadcast_batcher", create=True
    ) as mock_op:
        mod.broadcast_batcher()
        mock_op.assert_called_once_with()


def test_defbroadcasting() -> None:
    """Test defbroadcasting."""
    with patch("zero_jax._compiler_proxy_ops.defbroadcasting", create=True) as mock_op:
        mod.defbroadcasting()
        mock_op.assert_called_once_with()


def test_defreducer() -> None:
    """Test defreducer."""
    with patch("zero_jax._compiler_proxy_ops.defreducer", create=True) as mock_op:
        mod.defreducer()
        mock_op.assert_called_once_with()


def test_defvectorized() -> None:
    """Test defvectorized."""
    with patch("zero_jax._compiler_proxy_ops.defvectorized", create=True) as mock_op:
        mod.defvectorized()
        mock_op.assert_called_once_with()


def test_flatten_fun_for_vmap() -> None:
    """Test flatten_fun_for_vmap."""
    with patch(
        "zero_jax._compiler_proxy_ops.flatten_fun_for_vmap", create=True
    ) as mock_op:
        mod.flatten_fun_for_vmap()
        mock_op.assert_called_once_with()


def test_from_elt() -> None:
    """Test from_elt."""
    with patch("zero_jax._compiler_proxy_ops.from_elt", create=True) as mock_op:
        mod.from_elt()
        mock_op.assert_called_once_with()


def test_is_vmappable() -> None:
    """Test is_vmappable."""
    with patch("zero_jax._compiler_proxy_ops.is_vmappable", create=True) as mock_op:
        mod.is_vmappable()
        mock_op.assert_called_once_with()


def test_make_iota() -> None:
    """Test make_iota."""
    with patch("zero_jax._compiler_proxy_ops.make_iota", create=True) as mock_op:
        mod.make_iota()
        mock_op.assert_called_once_with()


def test_matchaxis() -> None:
    """Test matchaxis."""
    with patch("zero_jax._compiler_proxy_ops.matchaxis", create=True) as mock_op:
        mod.matchaxis()
        mock_op.assert_called_once_with()


def test_moveaxis() -> None:
    """Test moveaxis."""
    with patch("zero_jax._compiler_proxy_ops.moveaxis", create=True) as mock_op:
        mod.moveaxis()
        mock_op.assert_called_once_with()


def test_reducer_batcher() -> None:
    """Test reducer_batcher."""
    with patch("zero_jax._compiler_proxy_ops.reducer_batcher", create=True) as mock_op:
        mod.reducer_batcher()
        mock_op.assert_called_once_with()


def test_register_vmappable() -> None:
    """Test register_vmappable."""
    with patch(
        "zero_jax._compiler_proxy_ops.register_vmappable", create=True
    ) as mock_op:
        mod.register_vmappable()
        mock_op.assert_called_once_with()


def test_to_elt() -> None:
    """Test to_elt."""
    with patch("zero_jax._compiler_proxy_ops.to_elt", create=True) as mock_op:
        mod.to_elt()
        mock_op.assert_called_once_with()


def test_unregister_vmappable() -> None:
    """Test unregister_vmappable."""
    with patch(
        "zero_jax._compiler_proxy_ops.unregister_vmappable", create=True
    ) as mock_op:
        mod.unregister_vmappable()
        mock_op.assert_called_once_with()


def test_vectorized_batcher() -> None:
    """Test vectorized_batcher."""
    with patch(
        "zero_jax._compiler_proxy_ops.vectorized_batcher", create=True
    ) as mock_op:
        mod.vectorized_batcher()
        mock_op.assert_called_once_with()


def test_vtile() -> None:
    """Test vtile."""
    with patch("zero_jax._compiler_proxy_ops.vtile", create=True) as mock_op:
        mod.vtile()
        mock_op.assert_called_once_with()
