"""Comprehensive unit and branch coverage tests for zero-jax and compiler proxy operations."""

from __future__ import annotations

import pytest

import zero_jax
import zero_jax._compiler_proxy_creation as proxy_creation
import zero_jax._compiler_proxy_ops as proxy_ops
import zero_jax.experimental as exp_mod
import zero_jax.experimental.compilation_cache.compilation_cache as comp_cache
import zero_jax.experimental.x64_context as x64_ctx
import zero_jax.experimental.x64_context.config.logging_config as x64_log_cfg
import zero_jax.experimental.x64_context.config.profiler as x64_prof
import zero_jax.export as export_mod
import zero_jax.extend.backend as ext_backend
import zero_jax.extend.core as ext_core
import zero_jax.extend.ffi as ext_ffi
import zero_jax.extend.linear_util as ext_lu
import zero_jax.extend.random as ext_random
import zero_jax.extend.source_info_util as ext_src_info
import zero_jax.lax as lax
import zero_jax.nn as nn
import zero_jax.numpy as jnp
import zero_jax.random as random
import zero_jax.scipy.linalg as jsp_linalg
import zero_jax.tree_util as tree_util
from zero_jax._types import StatefulProtocol
from zero_jax.flatten_util import ravel_pytree


def test_all_ops_executed() -> None:
    """Verify compiler proxy operations, creation bindings, and unimplemented op stubs."""
    t_z = proxy_creation.zeros((2, 2))
    t_o = proxy_creation.ones((2, 2))
    t_f = proxy_creation.full((2, 2), 3.0)
    t_zl = proxy_creation.zeros_like(t_z)
    t_ol = proxy_creation.ones_like(t_o)
    t_fl = proxy_creation.full_like(t_f, 5.0)
    t_ar = proxy_creation.arange(0, 4, 1)
    t_eye = proxy_creation.eye(2)
    t_id = proxy_creation.identity(2)
    t_lin = proxy_creation.linspace(0.0, 1.0, 5)
    assert t_z.shape == (2, 2)
    assert t_o.shape == (2, 2)
    assert t_f.shape == (2, 2)
    assert t_zl.shape == (2, 2)
    assert t_ol.shape == (2, 2)
    assert t_fl.shape == (2, 2)
    assert t_ar.shape == (4,)
    assert t_eye.shape == (2, 2)
    assert t_id.shape == (2, 2)
    assert t_lin.shape == (5,)

    assert proxy_ops.rrelu(t_o).shape == (2, 2)

    with pytest.raises(AttributeError):
        proxy_ops.AbstractedAxesSpec()


def test_api_ad_and_misc_and_transformations_branches() -> None:
    """Cover all remaining branches in zero_jax.api (ad, misc, transformations)."""
    fn = lambda x: x * 2.0

    # ad.py wrappers
    j_fn = zero_jax.jacobian(fn)
    assert j_fn(3.0).shape == ()
    h_fn = zero_jax.hessian(lambda x: x * x)
    assert h_fn(3.0).shape == ()

    _, vjp_fn = zero_jax.vjp(fn, 3.0)
    assert vjp_fn(1.0) == (2.0,)

    _, lin_fn = zero_jax.linearize(fn, 3.0)
    assert lin_fn(4.0) == 8.0

    cjvp = zero_jax.custom_jvp(fn)
    assert cjvp.defjvp(fn) is fn
    assert cjvp(3.0) == 6.0

    cvjp = zero_jax.custom_vjp(fn)
    cvjp.defvjp(fn, fn)
    assert cvjp(3.0) == 6.0

    # misc.py functions
    assert zero_jax.default_prng_impl() == "threefry2x32"
    assert zero_jax.default_matmul_precision() == "highest"
    zero_jax.jax2tf_associative_scan_reductions()
    zero_jax.enable_custom_prng(True)
    zero_jax.enable_custom_vjp_by_custom_transpose(True)
    with zero_jax.debug_key_reuse(True):
        pass
    with zero_jax.legacy_prng_key():
        pass
    with zero_jax.threefry_partitionable():
        pass

    # transformations.py branches: cached jit, multi-output jit, grad with callable/stateful
    jit_fn = zero_jax.jit(lambda x: (x + 1.0, x + 1.0))
    r1, _r2 = jit_fn(jnp.array([1.0, 2.0]))
    r3, _r4 = jit_fn(jnp.array([3.0, 4.0]))
    assert r1.shape == (2,) and r3.shape == (2,)

    def dup_out(x: zero_jax.Array) -> tuple[zero_jax.Array, zero_jax.Array]:
        """Return the same tensor twice to exercise duplicate output ID branch."""
        y = x + 1.0
        return y, y

    jit_dup = zero_jax.jit(dup_out)
    d1, d2 = jit_dup(jnp.array([1.0]))
    assert d1.shape == (1,) and d2.shape == (1,)

    class _DummyStateful:
        """Stateful test helper implementing StatefulProtocol."""

        def __init__(self) -> None:
            """Initialize _DummyStateful."""
            self.state = {"w": jnp.array([1.0])}

    s_obj = _DummyStateful()
    assert isinstance(s_obj, StatefulProtocol)

    grad_with_static = zero_jax.grad(lambda s, x: jnp.sum(x * 2.0))
    assert grad_with_static(s_obj, jnp.array([3.0])) == 0.0

    grad_normal = zero_jax.grad(lambda x, s: jnp.sum(x * 3.0))
    g_res = grad_normal(jnp.array([2.0]), s_obj)
    assert g_res is not None

    # vmap with 0-D scalar argument
    vmap_scalar = zero_jax.vmap(lambda x: x + 1.0)
    assert vmap_scalar(jnp.array(2.0)).shape == ()


def test_experimental_export_extend_and_flatten_util() -> None:
    """Cover experimental, export, extend, partial_eval.config, and flatten_util."""
    # experimental/__init__.py
    earr = exp_mod.EArray(1, id="1", name="e", value=2, dummy=3)
    assert earr.id == "1"
    with pytest.raises(AttributeError):
        exp_mod.disable_x64()
    with pytest.raises(AttributeError):
        exp_mod.enable_x64()
    with pytest.raises(AttributeError):
        exp_mod.io_callback()

    # compilation_cache
    with pytest.raises(AttributeError):
        comp_cache.initialize_cache("/tmp")
    with pytest.raises(AttributeError):
        comp_cache.is_initialized()
    with pytest.raises(AttributeError):
        comp_cache.reset_cache()
    with pytest.raises(AttributeError):
        comp_cache.set_cache_dir("/tmp")

    # x64_context
    with pytest.raises(AttributeError):
        x64_ctx.contextmanager()
    with pytest.raises(AttributeError):
        x64_ctx.disable_x64()
    with pytest.raises(AttributeError):
        x64_ctx.enable_x64()
    with pytest.raises(AttributeError):
        x64_log_cfg.disable_all_debug_logging()
    with pytest.raises(AttributeError):
        x64_log_cfg.enable_debug_logging()
    assert x64_prof is not None

    # export
    assert export_mod.DisabledSafetyCheck.apply() is None
    assert export_mod.Exported.apply() is None
    assert export_mod.SymbolicScope.apply() is None
    assert export_mod.default_export_platform() is None
    assert export_mod.deserialize(b"") is None
    assert export_mod.export(lambda x: x) is None

    # extend
    assert ext_backend.stub() is None
    assert ext_core.stub() is None
    assert ext_ffi.stub() is None
    assert ext_lu.stub() is None
    assert ext_random.stub() is None
    assert ext_src_info.stub() is None

    # partial_eval.config
    import sys

    pe_cfg_real = sys.modules["zero_jax.interpreters.partial_eval.config"]
    assert pe_cfg_real.transfer_guard("allow") is None
    assert pe_cfg_real.transfer_guard_device_to_device("allow") is None
    assert pe_cfg_real.transfer_guard_device_to_host("allow") is None

    # flatten_util: empty PyTree, scalar PyTree, array PyTree
    flat_empty, unflatten_empty = ravel_pytree([])
    assert flat_empty.shape == (0,)
    assert unflatten_empty(flat_empty) == []

    flat_mixed, unflatten_mixed = ravel_pytree({"a": jnp.array([1.0, 2.0]), "b": 3.0})
    assert flat_mixed.shape == (3,)
    recon = unflatten_mixed(flat_mixed)
    assert isinstance(recon, dict)
    assert recon["a"].shape == (2,)


def test_lax_nn_random_scipy_tree_util_branches() -> None:
    """Cover remaining branches in lax, nn, random, scipy.linalg, and tree_util."""
    # lax/missing_funcs.py & lax/primitives.py
    assert lax.dtype(123) is None
    assert lax.dtype(jnp.array([1.0])) is not None
    assert lax.reshape(jnp.ones((2, 3)), (6,), dimensions=(1, 0)).shape == (6,)
    assert lax.slice(jnp.array([1, 2, 3, 4]), [0], [4], strides=[2]).shape == (2,)
    assert lax.slice_in_dim(jnp.array([1, 2, 3, 4]), 1, None).shape == (3,)
    assert lax.slice_in_dim(jnp.array([1, 2, 3, 4]), None, 2).shape == (2,)
    from zero_jax.lax import primitives as lax_prims

    assert lax_prims.top_k(jnp.array([1.0, 3.0, 2.0]), 2)[0].shape == (2,)
    assert lax.fft(jnp.array([1.0, 2.0]), "FFT", (2,)).shape == (2,)
    assert lax_prims.fft(jnp.array([1.0, 2.0]), "FFT", (2,)).shape == (2,)

    # nn/initializers.py & nn/missing_funcs.py
    from zero_jax.nn.initializers import _compute_fans

    assert _compute_fans((5,)) == (5, 5)
    k = random.PRNGKey(0)
    assert nn.initializers.uniform(0.05)(k, (2, 2)).shape == (2, 2)
    assert nn.initializers.normal(0.05)(k, (2, 2)).shape == (2, 2)
    assert nn.initializers.truncated_normal(0.05)(k, (2, 2)).shape == (2, 2)
    assert nn.initializers.variance_scaling(1.0, "fan_in", "uniform")(k, (2, 2)).shape == (2, 2)
    assert nn.initializers.orthogonal()(k, (2, 2)).shape == (2, 2)
    assert nn.initializers.delta_orthogonal()(k, (3, 2, 2)).shape == (3, 2, 2)
    assert nn.glu(jnp.ones((2, 4))).shape == (2, 2)
    assert nn.standardize(jnp.array([1.0, 2.0, 3.0]), mean=jnp.array([2.0]), variance=jnp.array([1.0])).shape == (3,)

    # random/prng.py explicit dtype/shape branches
    assert random.uniform(k, (2,), dtype=jnp.float32).shape == (2,)
    assert random.normal(k, (2,), dtype=jnp.float32).shape == (2,)
    assert random.randint(k, (2,), 0, 5, dtype=jnp.int32).shape == (2,)
    assert random.bernoulli(k, 0.5, shape=(2,)).shape == (2,)
    assert random.truncated_normal(k, -1.0, 1.0, shape=None, dtype=jnp.float32).shape == ()

    # scipy/linalg/__init__.py
    a = jnp.eye(2)._tensor
    b = jnp.ones((2, 1))._tensor
    with pytest.raises(Exception):
        jsp_linalg.cholesky_solve((a, False), b)
    with pytest.raises(Exception):
        jsp_linalg.lu_reconstruct(a, jnp.array([0, 1])._tensor)
    with pytest.raises(Exception):
        jsp_linalg.lu_matrix_inverse(a, jnp.array([0, 1])._tensor)

    # tree_util/missing_funcs.py & tree_util/pytree.py
    assert tree_util.all_leaves([1, [2, 3]]) is False
    assert tree_util.all_leaves([1, 2]) is True
    assert tree_util.tree_map_with_path(lambda p, x, y: x + y, {"a": 1}, {"a": 2}) == {"a": 3}
    assert tree_util.tree_reduce(lambda x, y: x + y, [1, 2, 3], initializer=10) == 16
    with pytest.raises(ValueError):
        outer_def = tree_util.tree_structure([1, 2])
        inner_def = tree_util.tree_structure([1, 2, 3])
        tree_util.tree_transpose(outer_def, inner_def, [[1, 2], [3, 4]])

    class _CustomNode:
        """Custom PyTree node for testing registration."""

        def __init__(self, x: int, y: int) -> None:
            """Initialize _CustomNode."""
            self.x = x
            self.y = y

    tree_util.register_pytree_node(
        _CustomNode,
        lambda n: ((n.x, n.y), "meta"),
        lambda meta, children: _CustomNode(children[0], children[1]),
    )
    cn = _CustomNode(3, 4)
    leaves, cdef = tree_util.tree_flatten(cn)
    assert leaves == [3, 4]
    recon_cn = tree_util.tree_unflatten(cdef, list(leaves))
    assert isinstance(recon_cn, _CustomNode) and recon_cn.x == 3 and recon_cn.y == 4


def test_numpy_constants_fft_linalg_and_lax_numpy_branches() -> None:
    """Cover remaining lines and branches in numpy/constants.py, fft.py, linalg.py, and lax_numpy.py."""
    # numpy/config.py & numpy/constants.py
    with jnp.printoptions(precision=4):
        pass
    assert jnp.index_exp[1] == (1,)
    assert jnp.index_exp[1, 2] == (1, 2)
    assert jnp.s_[1:3] == slice(1, 3)

    # numpy/fft.py
    x1 = jnp.array([1.0, 2.0, 3.0, 4.0])
    x2 = jnp.ones((2, 4))
    assert jnp.fft.fft(x1).shape == (4,)
    assert jnp.fft.ifft(x1).shape == (4,)
    assert jnp.fft.fft2(x2).shape == (2, 4)
    assert jnp.fft.ifft2(x2).shape == (2, 4)
    assert jnp.fft.fftn(x2).shape == (2, 4)
    assert jnp.fft.ifftn(x2).shape == (2, 4)
    assert jnp.fft.rfft(x1).shape == (3,)
    assert jnp.fft.irfft(jnp.fft.rfft(x1)).shape == (4,)
    assert jnp.fft.rfft2(x2).shape == (2, 3)
    assert jnp.fft.irfft2(jnp.fft.rfft2(x2)).shape == (2, 4)
    assert jnp.fft.rfftn(x2).shape == (2, 3)
    assert jnp.fft.irfftn(jnp.fft.rfftn(x2)).shape == (2, 4)
    assert jnp.fft.hfft(x1).shape == (6,)
    assert jnp.fft.ihfft(x1).shape == (3,)
    assert jnp.fft.fftfreq(4).shape == (4,)
    assert jnp.fft.rfftfreq(4).shape == (3,)
    assert jnp.fft.fftshift(x1).shape == (4,)
    assert jnp.fft.ifftshift(x1).shape == (4,)

    # numpy/linalg.py
    m = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    v = jnp.array([1.0, 2.0])
    v3 = jnp.array([1.0, 0.0, 0.0])
    w3 = jnp.array([0.0, 1.0, 0.0])
    assert jnp.linalg.cross(v3, w3).shape == (3,)
    assert jnp.linalg.diagonal(m).shape == (2,)
    assert jnp.linalg.eigh(m) is not None
    assert jnp.linalg.eigvalsh(m) is not None
    assert jnp.linalg.cond(m) is not None
    assert jnp.linalg.eig(m) is not None
    assert jnp.linalg.eigvals(m) is not None
    assert jnp.linalg.lstsq(m, v) is not None
    assert jnp.linalg.matmul(m, m).shape == (2, 2)
    assert jnp.linalg.matrix_norm(m) is not None
    assert jnp.linalg.matrix_power(m, 2).shape == (2, 2)
    assert jnp.linalg.matrix_rank(m) is not None
    assert jnp.linalg.matrix_transpose(m).shape == (2, 2)
    assert jnp.linalg.multi_dot([m, m]).shape == (2, 2)
    assert jnp.linalg.norm(m) is not None
    assert jnp.linalg.outer(v, v).shape == (2, 2)
    assert jnp.linalg.pinv(m).shape == (2, 2)
    assert jnp.linalg.slogdet(m) is not None
    assert jnp.linalg.svdvals(m) is not None
    assert jnp.linalg.tensordot(m, m).shape == ()
    assert jnp.linalg.tensorinv(jnp.reshape(jnp.eye(4), (2, 2, 2, 2))) is not None
    assert jnp.linalg.tensorsolve(jnp.reshape(jnp.eye(4), (2, 2, 2, 2)), jnp.ones((2, 2))) is not None
    assert jnp.linalg.trace(m) is not None
    assert jnp.linalg.vecdot(v, v) is not None
    assert jnp.linalg.vector_norm(v) is not None

    # lax_numpy.py dunders and edge cases
    a = jnp.array([2.0, 4.0])
    assert a.block_until_ready() is a
    assert repr(a).startswith("Array(")
    assert a.__pow__(2.0).shape == (2,)
    assert a.__rpow__(2.0).shape == (2,)
    assert a.__neg__().shape == (2,)
    assert a.__le__(4.0).shape == (2,)
    assert a.__ge__(2.0).shape == (2,)
    assert (2.0**a).shape == (2,)
    assert (8.0 / a).shape == (2,)
    assert (a // 2.0).shape == (2,)
    assert (8.0 // a).shape == (2,)
    assert (a >= 2.0).shape == (2,)
    assert list(iter(a)) is not None

    # indexing with ndarray, Tensor, tuple of Tensors/ndarrays
    idx_arr = jnp.array([0, 1])
    assert a[idx_arr].shape == (2,)
    m2 = jnp.ones((3, 3))
    assert m2[(jnp.array([0]), jnp.array([1])._tensor)].shape == (1,)

    with pytest.raises(ValueError):
        bool(a)

    # clip, max, sum, zeros_like, zeros, ones, empty, ones_like, empty_like, linspace, eye, meshgrid
    assert jnp.clip(a, None, 3.0).shape == (2,)
    assert jnp.clip(a, 1.0, None).shape == (2,)
    assert jnp.max(a, where=jnp.array([True, False]), initial=0.0) is not None
    assert jnp.max(a, where=jnp.array([True, False])) is not None
    assert jnp.sum(a, where=jnp.array([True, False])) is not None
    assert jnp.zeros_like(a, dtype=jnp.float32).shape == (2,)
    assert jnp.zeros((2,), dtype=jnp.float32).shape == (2,)
    assert jnp.broadcast_shapes() == ()
    assert jnp.broadcast_shapes((2,), (1, 2)) == (1, 2)
    assert jnp.choose(jnp.array([0, 1]), [jnp.array([1, 2]), jnp.array([3, 4])]).shape == (2,)
    assert jnp.ones((2,), dtype=jnp.float32).shape == (2,)
    assert jnp.empty((2,), dtype=jnp.float32).shape == (2,)
    assert jnp.ones_like(a, dtype=jnp.float32).shape == (2,)
    assert jnp.empty_like(a, dtype=jnp.float32).shape == (2,)
    assert jnp.linspace(0.0, 1.0, 1, endpoint=True).shape == (1,)
    assert jnp.linspace(0, 4, 3, dtype=jnp.float32).shape == (3,)
    assert jnp.eye(2, 3, k=0).shape == (2, 3)
    assert jnp.eye(2, 2, k=1).shape == (2, 2)
    assert jnp.eye(2, 2, k=1, dtype=jnp.float32).shape == (2, 2)
    mg = jnp.meshgrid(jnp.array([1, 2]), sparse=True, indexing="ij")
    assert len(mg) == 1

    # ravel, tile, isneginf, isposinf, searchsorted, signbit, block, apply_along_axis, dtype, isscalar, partition, ComplexWarning
    assert jnp.ravel(m, order="F").shape == (4,)
    with pytest.raises(ValueError):
        jnp.ravel(m, order="Z")
    assert jnp.tile(a, [2]).shape == (4,)
    with pytest.raises(ValueError):
        jnp.isneginf(a, out=a)
    with pytest.raises(ValueError):
        jnp.isposinf(a, out=a)
    assert jnp.searchsorted(jnp.array([1, 2, 3]), jnp.array([2]), sorter=jnp.array([0, 1, 2])).shape == (1,)
    with pytest.raises(ValueError):
        jnp.signbit(a, out=a)

    cw = jnp.ComplexWarning("warn")
    assert isinstance(cw, Warning)
    assert jnp.block([[jnp.array([1, 2]), jnp.array([3, 4])]]).shape == (1, 4)
    assert jnp.ndim(jnp.array([1, 2])) == 1
    assert jnp.apply_along_axis(lambda x: x + 1, 0, jnp.array([1, 2])).shape == (2,)

    assert jnp.dtype(jnp.float32) == jnp.float32
    assert jnp.dtype("float32") == jnp.float32
    assert jnp.dtype("invalid_dtype_xyz") == "invalid_dtype_xyz"

    assert jnp.isscalar(a) is False
    assert jnp.isscalar(5) is True
    assert jnp.isscalar(None) is True
    assert jnp.isscalar([1, 2]) is False

    assert jnp.partition(jnp.array([3, 1, 2]), 1) is not None
    assert jnp.uint(3) is not None
    assert jnp.uint4(3) is not None
    assert jnp.average(jnp.array([1.0, 2.0]), weights=jnp.array([1.0, 1.0])) is not None
    with pytest.raises(AttributeError):
        jnp.modf(jnp.array([1.5, 2.5]))
    with pytest.raises(AttributeError):
        jnp.stable_unique(jnp.array([2, 1, 2]))
    from unittest.mock import patch

    from zero_jax.numpy.lax_numpy import _safe_bool

    assert _safe_bool(True) is True
    assert jnp.unravel_index(2, (2, 2)) is not None
    with patch("zero_jax._compiler_proxy_ops.vectorize", return_value=lambda *args, **kwargs: args[0]):
        vec_fn = jnp.vectorize(lambda x: x + 1)
        assert vec_fn(jnp.array([1, 2])) is not None
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_ir import DType

        vec_fn._vec = Tensor(lambda *args, **kwargs: args[0], TensorConfig((), DType("float32"), "cpu"))
        assert vec_fn(jnp.array([1, 2])) is not None


def test_config_internal_mock_p_and_ndarray_methods_coverage() -> None:
    """Cover _config_internal, mock_p.bind, ndarray.at/methods, and remaining AD/API branches."""
    import os

    import zero_jax._config_internal as cfg_int
    import zero_jax.api.core as api_core
    import zero_jax.lax.mock_p as mock_p

    # _config_internal
    fh = cfg_int.FlagHolder("my_flag", default=True)
    assert fh.value is True
    fh2 = cfg_int.FlagHolder("my_flag", default=False, value=True)
    assert fh2.value is True

    assert cfg_int.bool_env("", default=True) is True
    assert cfg_int.int_env("", default=7) == 7
    assert cfg_int.str_env("", default="abc") == "abc"
    os.environ["_ZERO_JAX_TEST_BOOL"] = "true"
    os.environ["_ZERO_JAX_TEST_INT"] = "42"
    os.environ["_ZERO_JAX_TEST_STR"] = "hello"
    assert cfg_int.bool_env("_ZERO_JAX_TEST_BOOL") is True
    assert cfg_int.int_env("_ZERO_JAX_TEST_INT") == 42
    assert cfg_int.str_env("_ZERO_JAX_TEST_STR") == "hello"
    del os.environ["_ZERO_JAX_TEST_BOOL"]
    del os.environ["_ZERO_JAX_TEST_INT"]
    del os.environ["_ZERO_JAX_TEST_STR"]

    with cfg_int.enable_x64(True):
        assert cfg_int.config.jax_enable_x64 is True
    assert cfg_int.config.jax_enable_x64 is False

    with cfg_int.default_device("cpu:0"):
        assert cfg_int.config.read("jax_default_device") == "cpu:0"
    with cfg_int.default_matmul_precision("high"):
        assert cfg_int.config.read("jax_default_matmul_precision") == "high"

    # api.core.default_device
    assert api_core.default_device().platform == "cpu"

    # mock_p.bind
    assert mock_p.abs_p.bind(-3.0) == 3.0
    unknown_p = mock_p.MockPrimitive("nonexistent_op_p")
    assert unknown_p.bind(5.0) == 5.0
    assert unknown_p.bind() is None

    # ndarray.at and methods
    arr = jnp.array([1.0, 2.0, 3.0])
    assert arr.at[jnp.array(0)].get() == 1.0
    assert arr.at[jnp.array(0)].set(9.0)[0] == 9.0
    mat = jnp.ones((2, 2))
    assert mat.at[(jnp.array(0), jnp.array(1)._tensor)].set(9.0)[0, 1] == 9.0
    assert arr.at[0].add(5.0)[0] == 6.0
    assert arr.at[0].multiply(4.0)[0] == 4.0
    assert arr.at[1].divide(2.0)[1] == 1.0
    assert arr.at[1].power(3.0)[1] == 8.0
    assert arr.at[2].min(1.0)[2] == 1.0
    assert arr.at[0].max(9.0)[0] == 9.0
    assert arr.at[0].apply(lambda x: x + 10.0)[0] == 11.0

    assert arr.size == 3
    assert mat.T.shape == (2, 2)
    assert mat.mT.shape == (2, 2)
    assert jnp.array(7.0).item() == 7.0
    assert arr.tolist() == [1.0, 2.0, 3.0]
    assert arr.astype("int32").shape == (3,)
    assert jnp.ones((2, 3)).reshape((6,)).shape == (6,)
    assert jnp.ones((2, 3)).reshape(3, 2).shape == (3, 2)
    assert mat.transpose().shape == (2, 2)
    assert mat.transpose((1, 0)).shape == (2, 2)
    assert mat.transpose(1, 0).shape == (2, 2)
    assert mat.ravel().shape == (4,)
    assert mat.flatten().shape == (4,)
    assert jnp.ones((1, 3)).squeeze(axis=0).shape == (3,)
    assert arr.sum() == 6.0
    assert arr.mean() == 2.0
    assert arr.min() == 1.0
    assert arr.max() == 3.0
    assert (mat @ mat).shape == (2, 2)
    assert (mat.__rmatmul__(mat)).shape == (2, 2)

    # eval_shape with scalar and ShapeDtypeStruct
    sa_out = zero_jax.eval_shape(lambda x, c: x + c, zero_jax.ShapeDtypeStruct((2, 2), "float32"), 1.0)
    assert sa_out.shape == (2, 2)

    # vjp multi-output branch
    _, vjp_multi = zero_jax.vjp(lambda x: (x * 2.0, x * 3.0), jnp.array(2.0))
    assert vjp_multi((jnp.array(1.0), jnp.array(1.0)))[0] == 5.0

    # closure_convert with non-callable and object with non-ndarray attr
    class _WithMixedAttrs:
        """Callable with both ndarray and non-ndarray attributes."""

        def __init__(self) -> None:
            """Initialize _WithMixedAttrs."""
            self.label = "test"
            self.w = jnp.array(1.0)

        def __call__(self, x: float) -> float:
            """Call _WithMixedAttrs."""
            return x

    zero_jax.closure_convert(_WithMixedAttrs())
    non_fn: zero_jax._types.VariadicCallable[zero_jax._types.PyTree] = 123
    zero_jax.closure_convert(non_fn)

    # stop_gradient with non-proxy leaf during tracing
    from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer

    _tracer.start_tracing(name="sg_test")
    try:
        assert zero_jax.lax.stop_gradient(jnp.array(5.0)).shape == ()
    finally:
        _tracer.stop_tracing()

    # api_util.rebase_donate_argnums when donated arg is also in static_argnums
    assert zero_jax.api_util.rebase_donate_argnums((0, 1), (0,)) == (0,)
