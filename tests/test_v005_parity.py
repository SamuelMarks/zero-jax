"""Unit and branch coverage tests for ml-ecosystem-snapshots v0.0.5 parity additions."""

from __future__ import annotations

import pytest

import zero_jax as jax
import zero_jax.experimental.pallas as pl
import zero_jax.lax as lax
import zero_jax.nn as jnn
import zero_jax.numpy as jnp
import zero_jax.random as random
import zero_jax.sharding as sharding


def test_array_methods_v005_coverage() -> None:
    """Test all 33 newly added ``jax.Array`` methods and 7 aligned signatures."""
    arr = jnp.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    assert arr.item(0) == 3.0
    assert jnp.array(7.0).item() == 7.0
    assert arr.astype(jnp.int32, copy=True, device="cpu").dtype == jnp.int32
    assert arr.reshape(3, 2, order="C", out_sharding=None).shape == (3, 2)
    assert arr.reshape((6,), order="C", out_sharding=None).shape == (6,)
    assert arr.transpose().shape == (3, 2)
    assert arr.transpose((1, 0)).shape == (3, 2)
    assert arr.transpose(1, 0).shape == (3, 2)
    assert arr.ravel(order="C", out_sharding=None).shape == (6,)
    assert arr.flatten(order="C", out_sharding=None).shape == (6,)
    assert float(arr.sum(promote_integers=True).item()) == 21.0
    assert float(arr.sum(initial=1.0, dtype=jnp.float32).item()) == 22.0
    assert arr.addressable_data(0).shape == (2, 3)

    bool_arr = jnp.array([[True, False], [True, True]])
    assert bool(bool_arr.all().item()) is False
    assert bool(bool_arr.all(where=jnp.array([[True, False], [True, True]])).item()) is True
    assert bool(bool_arr.any().item()) is True
    assert bool(bool_arr.any(where=jnp.array([[False, True], [False, False]])).item()) is False

    assert int(arr.argmax().item()) == 3
    assert int(arr.argmin().item()) == 1
    assert arr.argpartition(1, axis=-1).shape == (2, 3)
    assert arr.argsort(axis=-1, descending=False).shape == (2, 3)
    assert arr.argsort(axis=-1, descending=True).shape == (2, 3)
    assert arr.argsort(axis=None, descending=True).shape == (6,)
    assert jnp.array([1, 2], dtype=jnp.int32).byteswap().shape == (2,)
    assert jnp.array([0, 1], dtype=jnp.int32).choose([jnp.array([10, 20]), jnp.array([30, 40])]).tolist() == [10, 40]
    assert arr.clip(2.0, 5.0).shape == (2, 3)
    assert arr.compress(jnp.array([True, False, True]), axis=1).shape == (2, 2)
    assert arr.conj().shape == (2, 3)
    assert arr.conjugate().shape == (2, 3)
    assert arr.copy().shape == (2, 3)
    arr.copy_to_host_async()
    assert arr.cumprod(axis=1).shape == (2, 3)
    assert arr.cumsum(axis=1).shape == (2, 3)
    assert arr.diagonal().shape == (2,)
    assert arr.dot(jnp.ones((3, 2))).shape == (2, 2)
    assert arr.dot(jnp.ones((3, 2)), preferred_element_type=jnp.float32).dtype == jnp.float32
    nz = arr.nonzero()
    assert len(nz) == 2
    assert float(arr.prod().item()) == 720.0
    assert float(arr.prod(where=jnp.array([[True, True, True], [False, False, False]]), initial=2.0).item()) == 12.0
    assert float(arr.ptp().item()) == 5.0
    assert arr.repeat(2, axis=0).shape == (4, 3)
    assert arr.round(decimals=1).shape == (2, 3)
    assert jnp.array([1.0, 3.0, 5.0]).searchsorted(jnp.array([2.0, 4.0])).tolist() == [1, 2]
    assert arr.sort(axis=-1, descending=False).shape == (2, 3)
    assert arr.sort(axis=-1, descending=True).shape == (2, 3)
    assert arr.sort(axis=None, descending=True).shape == (6,)
    assert float(arr.std().item()) > 0.0
    assert float(arr.std(correction=1).item()) > 0.0
    assert arr.swapaxes(0, 1).shape == (3, 2)
    assert arr.take(jnp.array([0, 2])).shape == (2,)
    assert arr.to_device("cpu:0").shape == (2, 3)
    assert float(arr.trace().item()) == 7.0
    assert float(arr.var().item()) > 0.0
    assert float(arr.var(correction=1).item()) > 0.0
    assert arr.view().shape == (2, 3)
    assert arr.view(dtype=jnp.float32).shape in ((2, 3), (2, 6))


def test_pallas_v005_coverage() -> None:
    """Test all 49 endpoints in ``jax.experimental.pallas``."""
    rev = pl.RevisitMode()
    assert rev.IMMEDIATE == "immediate"
    buf = pl.Buffered(2, use_lookahead=True, revisit=rev, prefetched_count=1)
    assert buf.buffer_count == 2
    bs = pl.BlockSpec((4, 4), lambda i: (i, 0), memory_space="default", pipeline_mode=buf)
    assert bs.block_shape == (4, 4)
    assert pl.BlockSpec(None).block_shape is None
    assert pl.Blocked(8).block_size == 8
    assert pl.BoundedSlice(16).block_size == 16
    cp = pl.CompilerParams(1, flag=True)
    assert cp.args == (1,)
    ce = pl.CostEstimate(10, 2, 64, 0)
    assert ce.flops == 10
    el = pl.Element(4, padding=(1, 1))
    assert el.padding == (1, 1)
    gs = pl.GridSpec((2, 2), in_specs=bs, out_specs=bs, scratch_shapes=())
    assert gs.grid == (2, 2)
    assert pl.Indirect(4).block_size == 4
    mr = pl.MemoryRef("aval", "smem")
    assert mr.memory_space == "smem"
    ms = pl.MemorySpace()
    assert ms.DEFAULT == "default"
    sl = pl.Slice(0, 4, 1)
    assert sl.size == 4
    sq = pl.Squeezed()
    assert isinstance(sq, pl.Squeezed)

    sem = pl.semaphore()
    assert pl.semaphore_read(sem) == 0
    assert pl.semaphore_read("non_sem") == 0
    pl.semaphore_signal(sem, 3)
    pl.semaphore_signal("non_sem", 1)
    assert pl.semaphore_read(sem) == 3
    pl.semaphore_wait(sem, 2, decrement=True)
    pl.semaphore_wait(sem, 1, decrement=False)
    assert pl.semaphore_read(sem) == 1

    assert pl.ANY(42) == 42
    assert pl.align_to(13, 8) == 16
    assert pl.broadcast_to(jnp.array([1.0, 2.0]), (3, 2)).shape == (3, 2)
    assert pl.cdiv(10, 3) == 4
    pl.debug_check(True, "ok")
    with pytest.raises(AssertionError):
        pl.debug_check(False, "fail")
    pl.debug_print("val=%d", 1)
    pl.delay(100)
    assert pl.dslice(4).size == 4
    assert pl.ds(2, 6, 2).start == 2
    e_arr = pl.empty((2, 2), dtype=jnp.float32)
    assert e_arr.shape == (2, 2)
    assert pl.empty_like(e_arr).shape == (2, 2)
    assert pl.empty_ref_like(e_arr).shape == (2, 2)
    with pl.enable_debug_checks(True) as dbg:
        assert dbg is True
    with pl.enable_poison_buffers(False) as psn:
        assert psn is False
    assert isinstance(pl.estimate_cost(lambda x: x, e_arr), pl.CostEstimate)
    assert pl.get_global("scratch") == "scratch"

    @pl.kernel(out_type=())
    def _k1(x: int) -> int:
        """Return x.

        Args:
            x: Input integer.

        Returns:
            int: Output integer.
        """
        return x

    assert _k1(5) == 5
    assert pl.kernel(lambda x: x + 1)(5) == 6

    total = 0

    @pl.loop(0, 4)
    def _stateless_loop(i: int) -> None:
        """Accumulate i.

        Args:
            i: Loop index.
        """
        nonlocal total
        total += i

    assert total == 6
    res_carry = pl.loop(0, 4, init_carry=10)(lambda i, c: c + i)
    assert res_carry == 16

    assert "pallas_kernel" in pl.lower_as_mlir(lambda x: x, e_arr)
    assert pl.multiple_of(16, 8) == 16
    assert pl.next_power_of_2(0) == 1
    assert pl.next_power_of_2(5) == 8
    with pytest.raises(ValueError):
        pl.next_power_of_2(-1)
    assert pl.num_programs(0) == 1
    assert pl.program_id(0) == 0

    def _copy_kernel(x_ref: jax.Array, o_ref: jax.Array) -> None:
        """No-op kernel.

        Args:
            x_ref: Input array reference.
            o_ref: Output array reference.
        """
        _ = (x_ref, o_ref)

    out_single = pl.pallas_call(_copy_kernel, out_shape=jnp.zeros((2, 2)))(e_arr)
    assert out_single.shape == (2, 2)
    out_tuple = pl.pallas_call(lambda x, o1, o2: None, out_shape=(jnp.zeros((2,)), jnp.zeros((2,))))(e_arr)
    assert len(out_tuple) == 2

    with pl.pallas_export_experimental(True) as exp_flag:
        assert exp_flag is True
    assert pl.run_scoped(lambda a, b=1: a + b, 2, b=3) == 5
    assert pl.run_state(lambda s: None)(42) == 42
    assert pl.select_ref(1, "a", "b", "c") == "b"
    assert pl.strides_from_shape((2, 3, 4)) == (12, 4, 1)
    assert pl.when(True)(lambda: 99) == 99
    assert pl.when(False)(lambda: 99) is None
    assert pl.with_scoped(10, extra=5)(lambda x, s, extra=0: x + s + extra)(2) == 17


def test_sharding_v005_coverage() -> None:
    """Test all 14 new ``jax.sharding`` classes/functions and 5 aligned classes."""
    ad = sharding.AbstractDevice("tpu", 2, "tpu")
    assert ad.device_kind == "tpu"
    am = sharding.AbstractMesh((2, 4), ("x", "y"), abstract_device=ad)
    assert am.size == 8
    assert sharding.AbstractMesh().size == 1
    at = sharding.AxisType()
    assert at.Auto == "Auto"
    mat = sharding.ManualAxisType("x")
    assert mat.args == ("x",)
    uk = sharding.UnreducedKind()
    assert uk.sum == "sum"

    m = sharding.Mesh(("cpu:0",), ("x",))
    with sharding.set_mesh(m) as active_m:
        assert active_m is m
        assert sharding.get_mesh() is m
    assert isinstance(sharding.get_mesh(), sharding.Mesh)

    with sharding.use_abstract_mesh(am) as active_am:
        assert active_am is am
        assert sharding.get_abstract_mesh() is am
    assert isinstance(sharding.get_abstract_mesh(), sharding.AbstractMesh)

    @sharding.auto_axes(axes="x")
    def _f_auto(x: int) -> int:
        """Return x.

        Args:
            x: Input integer.

        Returns:
            int: Output integer.
        """
        return x

    assert _f_auto(3) == 3
    assert sharding.auto_axes(lambda x: x)(4) == 4

    @sharding.explicit_axes(axes="x")
    def _f_exp(x: int) -> int:
        """Return x.

        Args:
            x: Input integer.

        Returns:
            int: Output integer.
        """
        return x

    assert _f_exp(3) == 3
    assert sharding.explicit_axes(lambda x: x)(4) == 4

    sds = sharding.make_single_device_sharding("cpu:0", memory_kind="device")
    assert sds.memory_kind == "device"
    with sharding.no_collectives(True) as nc:
        assert nc is True
    assert sharding.reshard(42, sds) == 42


def test_nn_and_random_v005_coverage() -> None:
    """Test ``jax.nn`` attention/scaled matmul and ``jax.random`` new endpoints."""
    q3 = jnp.ones((2, 4, 2, 8))
    k3 = jnp.ones((2, 4, 2, 8))
    v3 = jnp.ones((2, 4, 2, 8))
    out3 = jnn.dot_product_attention(q3, k3, v3, bias=jnp.zeros((2, 2, 4, 4)), mask=jnp.ones((2, 2, 4, 4), dtype=jnp.bool_), is_causal=True)
    assert out3.shape == (2, 4, 2, 8)
    out3_res, lse3 = jnn.dot_product_attention(q3, k3, v3, return_residual=True)
    assert out3_res.shape == (2, 4, 2, 8) and lse3.shape == (2, 2, 4)

    q2 = jnp.ones((4, 8))
    k2 = jnp.ones((4, 8))
    v2 = jnp.ones((4, 8))
    out2 = jnn.dot_product_attention(q2, k2, v2, bias=jnp.zeros((4, 4)), mask=jnp.ones((4, 4), dtype=jnp.bool_), is_causal=True)
    assert out2.shape == (4, 8)
    out2_res, lse2 = jnn.dot_product_attention(q2, k2, v2, return_residual=True)
    assert out2_res.shape == (4, 8) and lse2.shape == (4,)

    cfg = jnn.get_scaled_dot_general_config("nvfp4")
    assert cfg["mode"] == "nvfp4"
    sdg = jnn.scaled_dot_general(jnp.ones((2, 3)), jnp.ones((3, 4)), (((1,), (0,)), ((), ())))
    assert sdg.shape == (2, 4)
    sm1 = jnn.scaled_matmul(jnp.ones((2, 3)), jnp.ones((3, 2)), jnp.ones((2, 1)), jnp.ones((1, 2)), preferred_element_type=jnp.float32)
    assert sm1.shape == (2, 2)
    sm2 = jnn.scaled_matmul(jnp.ones((2, 3)), jnp.ones((3, 2)), preferred_element_type=None)
    assert sm2.shape == (2, 2)

    k_int = random.Key(0)
    assert k_int.shape == (2,)
    k_arr = random.Key(jnp.array([0, 1]))
    assert k_arr.shape == (2,)
    assert random.key_dtype() == jnp.uint32
    mn1 = random.multinomial(k_int, 10, jnp.array([0.2, 0.8]))
    assert mn1.shape == (2,)
    mn2 = random.multinomial(k_int, 10, jnp.array([0.5, 0.5]), shape=(3,))
    assert mn2.shape == (3, 2)


def test_lax_and_numpy_v005_coverage() -> None:
    """Test all 29 newly added ``jax.lax`` functions and 7 newly added ``jax.numpy`` functions."""
    assert lax.switch(0, [lambda x: x + 1, lambda x: x + 2], operand=5) == 6
    assert lax.axis_size("x") == 1
    assert lax.broadcast_like(jnp.array([1.0, 2.0]), jnp.zeros((3, 2))).shape == (3, 2)
    comp_fn = lax.composite(lambda x: x, "my_op")
    assert comp_fn(3) == 3
    assert lax.dce_sink(7) == 7
    assert lax.empty((2, 3)).shape == (2, 3)
    assert float(lax.log2(jnp.array(8.0)).item()) == pytest.approx(3.0)
    assert int(lax.mulhi(jnp.array(1 << 20), jnp.array(1 << 20)).item()) == (1 << 8)
    assert float(lax.one_minus_square(jnp.array(0.5)).item()) == pytest.approx(0.75)
    assert lax.optimization_barrier(9) == 9
    assert lax.pcast(jnp.array([1.0]), "x") is not None
    assert float(lax.polynomial(jnp.array(2.0), [1.0, 0.0, 1.0]).item()) == pytest.approx(5.0)
    assert lax.psum_scatter_start(jnp.array([1.0])) is not None
    assert lax.psum_scatter_start() is None
    out_buf = jnp.zeros((2, 2))
    assert lax.ragged_all_to_all(out_buf, out_buf, [0], [1], [0], [1], axis_name="x") is out_buf
    assert lax.ragged_dot_general(jnp.ones((2, 3)), jnp.ones((3, 2)), [2]).shape == (2, 2)
    assert bool(lax.reduce_and(jnp.array([True, False]), [0]).item()) is False
    assert float(lax.reduce_max(jnp.array([1.0, 5.0]), [0]).item()) == 5.0
    assert float(lax.reduce_min(jnp.array([1.0, 5.0]), [0]).item()) == 1.0
    assert bool(lax.reduce_or(jnp.array([True, False]), [0]).item()) is True
    assert float(lax.reduce_prod(jnp.array([2.0, 3.0]), [0]).item()) == 6.0
    assert float(lax.reduce_sum(jnp.array([2.0, 3.0]), [0]).item()) == 5.0
    assert int(lax.reduce_xor(jnp.array([1, 3, 2], dtype=jnp.int32), 0).item()) == 0
    assert lax.scaled_dot(jnp.ones((2, 3)), jnp.ones((3, 2)), jnp.ones((2, 1)), jnp.ones((1, 2)), preferred_element_type=jnp.float32).shape == (2, 2)
    assert lax.scaled_dot(jnp.ones((2, 3)), jnp.ones((3, 2)), dimension_numbers=(((1,), (0,)), ((), ()))).shape == (2, 2)
    assert lax.scaled_dot(jnp.ones((2, 3)), jnp.ones((3, 2))).shape == (2, 2)
    sdn = lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
    sub_res = lax.scatter_sub(jnp.array([10.0, 20.0]), jnp.array([[0]]), jnp.array([3.0]), sdn)
    assert sub_res.tolist() == [7.0, 20.0]
    sav = lax.shape_as_value((2, 3))
    assert len(sav) == 2 and int(sav[0].item()) == 2
    sp = lax.split(jnp.arange(6), [2, 4], axis=0)
    assert len(sp) == 2 and sp[0].shape == (2,) and sp[1].shape == (4,)
    st = lax.stack([jnp.array([1, 2]), jnp.array([3, 4])], axis=0)
    assert st.shape == (2, 2)
    assert lax.stage(12) == 12
    assert lax.tile(jnp.array([1, 2]), (2,)).shape == (4,)
    ust = lax.unstack(st, axis=0)
    assert len(ust) == 2

    cp1 = jnp.cumulative_prod(jnp.array([1, 2, 3]), include_initial=False)
    assert cp1.tolist() == [1, 2, 6]
    cp2 = jnp.cumulative_prod(jnp.array([1, 2, 3]), axis=-1, include_initial=True)
    assert cp2.tolist() == [1, 1, 2, 6]
    mv = jnp.matvec(jnp.ones((2, 3)), jnp.ones((3,)))
    assert mv.shape == (2,)
    vm = jnp.vecmat(jnp.ones((2,)), jnp.ones((2, 3)))
    assert vm.shape == (3,)
    mn_val, mx_val = jnp.minmax(jnp.array([3.0, 1.0, 5.0]), initial=(0.0, 10.0))
    assert float(mn_val.item()) == 0.0 and float(mx_val.item()) == 10.0
    paa_flat = jnp.put_along_axis(jnp.zeros((2, 2)), jnp.array([0, 3]), jnp.array([9.0, 8.0]), axis=None)
    assert paa_flat.tolist() == [[9.0, 0.0], [0.0, 8.0]]
    paa_ax = jnp.put_along_axis(jnp.zeros((2, 3)), jnp.array([[1], [2]]), jnp.array([[5.0], [7.0]]), axis=-1)
    assert paa_ax.tolist() == [[0.0, 5.0, 0.0], [0.0, 0.0, 7.0]]
    sp_val = jnp.spacing(jnp.array(1.0))
    assert float(sp_val.item()) > 0.0
    vals, idxs = jnp.top_k(jnp.array([1.0, 5.0, 3.0]), 2)
    assert vals.tolist() == [5.0, 3.0] and idxs.tolist() == [1, 2]
