"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.batching as mod


def test_Array() -> None:
    """Test Array."""
    obj = mod.Array()
    assert obj is not None


def test_AxisSize() -> None:
    """Test AxisSize."""
    with pytest.raises(NotImplementedError):
        mod.AxisSize()


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
    with pytest.raises(NotImplementedError):
        mod.BatchingRule()


def test_Elt() -> None:
    """Test Elt."""
    with pytest.raises(NotImplementedError):
        mod.Elt()


def test_FromEltHandler() -> None:
    """Test FromEltHandler."""
    with pytest.raises(NotImplementedError):
        mod.FromEltHandler()


def test_GetIdx() -> None:
    """Test GetIdx."""
    with pytest.raises(NotImplementedError):
        mod.GetIdx()


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
    with pytest.raises(NotImplementedError):
        mod.MakeIotaHandler()


def test_MapSpec() -> None:
    """Test MapSpec."""
    with pytest.raises(NotImplementedError):
        mod.MapSpec()


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
    with pytest.raises(NotImplementedError):
        mod.ToEltHandler()


def test_Vmappable() -> None:
    """Test Vmappable."""
    with pytest.raises(NotImplementedError):
        mod.Vmappable()


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
    with pytest.raises(NotImplementedError):
        mod.batch()


def test_batch_custom_jvp_subtrace() -> None:
    """Test batch_custom_jvp_subtrace."""
    with pytest.raises(NotImplementedError):
        mod.batch_custom_jvp_subtrace()


def test_batch_custom_vjp_bwd() -> None:
    """Test batch_custom_vjp_bwd."""
    with pytest.raises(NotImplementedError):
        mod.batch_custom_vjp_bwd()


def test_batch_jaxpr() -> None:
    """Test batch_jaxpr."""
    with pytest.raises(NotImplementedError):
        mod.batch_jaxpr()


def test_batch_jaxpr2() -> None:
    """Test batch_jaxpr2."""
    with pytest.raises(NotImplementedError):
        mod.batch_jaxpr2()


def test_batch_jaxpr_axes() -> None:
    """Test batch_jaxpr_axes."""
    with pytest.raises(NotImplementedError):
        mod.batch_jaxpr_axes()


def test_batch_subtrace() -> None:
    """Test batch_subtrace."""
    with pytest.raises(NotImplementedError):
        mod.batch_subtrace()


def test_bdim_at_front() -> None:
    """Test bdim_at_front."""
    with pytest.raises(NotImplementedError):
        mod.bdim_at_front()


def test_broadcast() -> None:
    """Test broadcast."""
    with pytest.raises(NotImplementedError):
        mod.broadcast()


def test_broadcast_batcher() -> None:
    """Test broadcast_batcher."""
    with pytest.raises(NotImplementedError):
        mod.broadcast_batcher()


def test_defbroadcasting() -> None:
    """Test defbroadcasting."""
    with pytest.raises(NotImplementedError):
        mod.defbroadcasting()


def test_defreducer() -> None:
    """Test defreducer."""
    with pytest.raises(NotImplementedError):
        mod.defreducer()


def test_defvectorized() -> None:
    """Test defvectorized."""
    with pytest.raises(NotImplementedError):
        mod.defvectorized()


def test_flatten_fun_for_vmap() -> None:
    """Test flatten_fun_for_vmap."""
    with pytest.raises(NotImplementedError):
        mod.flatten_fun_for_vmap()


def test_from_elt() -> None:
    """Test from_elt."""
    with pytest.raises(NotImplementedError):
        mod.from_elt()


def test_is_vmappable() -> None:
    """Test is_vmappable."""
    with pytest.raises(NotImplementedError):
        mod.is_vmappable()


def test_make_iota() -> None:
    """Test make_iota."""
    with pytest.raises(NotImplementedError):
        mod.make_iota()


def test_matchaxis() -> None:
    """Test matchaxis."""
    with pytest.raises(NotImplementedError):
        mod.matchaxis()


def test_moveaxis() -> None:
    """Test moveaxis."""
    with pytest.raises(NotImplementedError):
        mod.moveaxis()


def test_reducer_batcher() -> None:
    """Test reducer_batcher."""
    with pytest.raises(NotImplementedError):
        mod.reducer_batcher()


def test_register_vmappable() -> None:
    """Test register_vmappable."""
    with pytest.raises(NotImplementedError):
        mod.register_vmappable()


def test_to_elt() -> None:
    """Test to_elt."""
    with pytest.raises(NotImplementedError):
        mod.to_elt()


def test_unregister_vmappable() -> None:
    """Test unregister_vmappable."""
    with pytest.raises(NotImplementedError):
        mod.unregister_vmappable()


def test_vectorized_batcher() -> None:
    """Test vectorized_batcher."""
    with pytest.raises(NotImplementedError):
        mod.vectorized_batcher()


def test_vtile() -> None:
    """Test vtile."""
    with pytest.raises(NotImplementedError):
        mod.vtile()
