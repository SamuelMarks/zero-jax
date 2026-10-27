"""Cross-ecosystem integration tests for zero-jax with zero-flax, zero-chex, zero-grain, zero-optax, and zero-orbax."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import zero_chex
import zero_flax
import zero_grain
import zero_optax
import zero_orbax
from zero_flax import nnx
from zero_flax.core import FrozenDict
from zero_orbax.core import GlobalArraySharding, SerializedArray

import zero_jax
import zero_jax._compiler_patches as compiler_patches
import zero_jax.ad_checkpoint as ad_checkpoint
import zero_jax.numpy as jnp
from zero_jax import lax
from zero_jax._types import (
    ArrayLikeSpec,
    ConfigValue,
    CustomStateMappingProtocol,
    CustomVariableProtocol,
    MappableDataclassProtocol,
    PyTree,
    RecordProtocol,
    ToListProtocol,
)
from zero_jax.api.transformations import _JitWrapped, _reduce_grad_to_shape, _resolve_leaf_axes
from zero_jax.sharding import Mesh, NamedSharding, PartitionSpec, PositionalSharding, Sharding, SingleDeviceSharding
from zero_jax.tree_util import (
    DictKey,
    FlattenedIndexKey,
    GetAttrKey,
    SequenceKey,
    register_dataclass,
    tree_flatten,
    tree_flatten_with_path,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)


@zero_chex.dataclass
class _TrainingMetrics:
    """Chex mappable dataclass holding step loss and prediction arrays."""

    loss: jnp.ndarray
    preds: jnp.ndarray


class _UnregisteredKeyedNode:
    """Custom PyTree node satisfying ``_TreeFlattenWithKeysProtocol`` without explicit registration."""

    def __init__(self, left: PyTree, right: PyTree, tag: str = "node") -> None:
        """Initialize ``_UnregisteredKeyedNode``.

        Args:
            left: Left child leaf or subtree.
            right: Right child leaf or subtree.
            tag: Static metadata tag.
        """
        self.left = left
        self.right = right
        self.tag = tag

    def tree_flatten_with_keys(self) -> tuple[Iterable[ArrayLikeSpec], ArrayLikeSpec]:
        """Return keyed children and static tag metadata.

        Returns:
            tuple[Iterable[ArrayLikeSpec], ArrayLikeSpec]: Key-child pairs and tag.
        """
        return [(GetAttrKey("left"), self.left), self.right], self.tag

    @classmethod
    def tree_unflatten(cls, aux: ArrayLikeSpec, children: Sequence[ArrayLikeSpec]) -> _UnregisteredKeyedNode:
        """Reconstruct ``_UnregisteredKeyedNode`` from ``aux`` and ``children``.

        Args:
            aux: Static tag string.
            children: Two-element sequence ``(left, right)``.

        Returns:
            _UnregisteredKeyedNode: Reconstructed node.
        """
        return cls(children[0], children[1], tag=str(aux))


def test_universal_pytree_protocols_and_containers() -> None:
    """Verify PyTree flattening, unflattening, mapping, and key paths across all ecosystem protocols."""
    # 1. Chex MappableDataclassProtocol
    metrics = _TrainingMetrics(loss=jnp.array(1.5), preds=jnp.array([2.0, 3.0]))
    assert isinstance(metrics, MappableDataclassProtocol)
    m_leaves, m_def = tree_flatten(metrics)
    assert len(m_leaves) == 2
    assert m_def.num_leaves == 2
    doubled_metrics = tree_map(lambda x: x * 2.0, metrics)
    assert isinstance(doubled_metrics, _TrainingMetrics)
    zero_chex.assert_trees_all_close(doubled_metrics.loss, jnp.array(3.0))
    zero_chex.assert_trees_all_close(doubled_metrics.preds, jnp.array([4.0, 6.0]))

    # 2. Flax State, Variable/Param, and FrozenDict
    var_param = nnx.Param(jnp.array([1.0, 2.0]), sharding=("data",))
    assert isinstance(var_param, CustomVariableProtocol)
    state = nnx.State({"w": var_param, "b": nnx.Variable(jnp.array([0.5]), tag="bias")})
    assert isinstance(state, CustomStateMappingProtocol)
    s_path_leaves, _s_def = tree_flatten_with_path(state)
    assert len(s_path_leaves) == 2
    assert s_path_leaves[0][0] == (DictKey("b"), GetAttrKey("value"))
    assert s_path_leaves[1][0] == (DictKey("w"), GetAttrKey("value"))
    mapped_state = tree_map(lambda x: x + 1.0, state)
    assert isinstance(mapped_state, nnx.State)
    assert isinstance(mapped_state["w"], nnx.Param)
    assert mapped_state["w"].get_metadata("sharding") == ("data",)
    assert mapped_state["b"].get_metadata("tag") == "bias"

    empty_state = nnx.State({})
    assert tree_structure(empty_state).num_leaves == 0

    fdict = FrozenDict({"alpha": jnp.array([1.0, 2.0]), "beta": jnp.array([3.0])})
    assert isinstance(fdict, CustomStateMappingProtocol)
    mapped_fdict = tree_map(lambda x: x * 3.0, fdict)
    assert isinstance(mapped_fdict, FrozenDict)
    zero_chex.assert_trees_all_close(mapped_fdict["alpha"], jnp.array([3.0, 6.0]))

    # 3. Optax OptState and MaskedNode
    opt = zero_optax.adam(1e-2)
    opt_state = opt.init({"w": jnp.array([1.0, 2.0])})
    opt_leaves, opt_def = tree_flatten(opt_state)
    rebuilt_opt_state = tree_unflatten(opt_def, opt_leaves)
    assert type(rebuilt_opt_state) is type(opt_state)
    masked = zero_optax.MaskedNode()
    assert tree_leaves(masked) == []
    assert tree_structure(masked).num_leaves == 0
    assert tree_unflatten(tree_structure(masked), []) == masked

    # 4. Grain Record & RecordMetadata
    rec = zero_grain.Record(
        metadata=zero_grain.RecordMetadata(index=0, record_key=1),
        data={"x": jnp.array([2.0, 4.0])},
    )
    assert isinstance(rec, RecordProtocol)
    rec_path_leaves, rec_def = tree_flatten_with_path(rec)
    assert rec_path_leaves[0][0] == (GetAttrKey("data"), DictKey("x"))
    mapped_rec = tree_map(lambda x: x * 0.5, rec)
    assert isinstance(mapped_rec, zero_grain.Record)
    assert mapped_rec.metadata == rec.metadata
    zero_chex.assert_trees_all_close(mapped_rec.data["x"], jnp.array([1.0, 2.0]))
    _ = tree_unflatten(rec_def, [jnp.array([5.0, 6.0])])

    # 5. Orbax SerializedArray & CustomVariableProtocol tensor conversion
    sa = SerializedArray(data=[1.0, 2.0], shape=(2,), dtype="float32")
    assert isinstance(sa, ToListProtocol)
    arr_from_sa = jnp.asarray(sa)
    zero_chex.assert_trees_all_close(arr_from_sa, jnp.array([1.0, 2.0]))
    arr_from_var = jnp.asarray(var_param)
    zero_chex.assert_trees_all_close(arr_from_var, jnp.array([1.0, 2.0]))

    # 6. Unregistered _TreeFlattenWithKeysProtocol, MappableDataclassProtocol, and RecordProtocol nodes
    ukn = _UnregisteredKeyedNode(jnp.array(2.0), jnp.array(5.0), tag="custom")
    ukn_path_leaves, ukn_def = tree_flatten_with_path(ukn)
    assert ukn_path_leaves[0][0] == (GetAttrKey("left"),)
    assert ukn_path_leaves[1][0] == (FlattenedIndexKey(1),)
    ukn_rebuilt = tree_unflatten(ukn_def, [jnp.array(20.0), jnp.array(50.0)])
    assert isinstance(ukn_rebuilt, _UnregisteredKeyedNode)
    assert ukn_rebuilt.tag == "custom"
    zero_chex.assert_trees_all_close(ukn_rebuilt.left, jnp.array(20.0))

    class _UnregisteredMappable:
        """Unregistered container satisfying ``MappableDataclassProtocol``."""

        def __init__(self, a: PyTree, b: PyTree) -> None:
            """Initialize ``_UnregisteredMappable``.

            Args:
                a: First field value.
                b: Second field value.
            """
            self.a = a
            self.b = b

        def __chex_mappable_fields__(self) -> tuple[str, ...]:
            """Return field names ``('a', 'b')``."""
            return ("a", "b")

        def __chex_field_values__(self) -> tuple[PyTree, ...]:
            """Return field values ``(self.a, self.b)``."""
            return (self.a, self.b)

    um = _UnregisteredMappable(jnp.array(1.0), jnp.array(2.0))
    um_mapped = tree_map(lambda x: x + 3.0, um)
    assert isinstance(um_mapped, _UnregisteredMappable)
    zero_chex.assert_trees_all_close(um_mapped.a, jnp.array(4.0))

    import dataclasses as _dc

    @_dc.dataclass
    class _UnregisteredRecord:
        """Unregistered dataclass satisfying ``RecordProtocol``."""

        metadata: str
        data: PyTree

    urec = _UnregisteredRecord(metadata="m0", data=jnp.array([3.0, 6.0]))
    urec_mapped = tree_map(lambda x: x * 2.0, urec)
    assert isinstance(urec_mapped, _UnregisteredRecord)
    assert urec_mapped.metadata == "m0"
    zero_chex.assert_trees_all_close(urec_mapped.data, jnp.array([6.0, 12.0]))

    # 7. Key equality, hashing, repr, and register_dataclass nested preservation
    dk = DictKey("a")
    sk = SequenceKey(0)
    gk = GetAttrKey("f")
    fk = FlattenedIndexKey(1)
    assert hash(dk) == hash(DictKey("a")) and repr(dk) == "['a']"
    assert hash(sk) == hash(SequenceKey(0)) and repr(sk) == "[0]"
    assert hash(gk) == hash(GetAttrKey("f")) and repr(gk) == ".f"
    assert hash(fk) == hash(FlattenedIndexKey(1)) and repr(fk) == "[<flat index 1>]"
    assert dk == DictKey("a") and sk == SequenceKey(0) and gk == GetAttrKey("f") and fk == FlattenedIndexKey(1)


def test_ad_checkpoint_policies_and_flax_remat() -> None:
    """Verify ad_checkpoint policies, checkpoint_name graph tagging, and Flax nnx.Remat delegation."""
    assert ad_checkpoint.everything_saveable("x") is True
    assert ad_checkpoint.nothing_saveable("x") is False
    assert ad_checkpoint.checkpoint_dots("dot_general") is True
    assert ad_checkpoint.checkpoint_dots("add") is False
    assert ad_checkpoint.checkpoint_dots_with_no_batch_dims("matmul", batch_dims=0) is True
    assert ad_checkpoint.checkpoint_dots_with_no_batch_dims("matmul", batch_dims=(0,)) is False

    only_h1 = ad_checkpoint.save_only_these_names("h1")
    assert only_h1("h1") is True
    assert only_h1("h2") is False

    except_h2 = ad_checkpoint.save_any_names_but_these("h2")
    assert except_h2("h1") is True
    assert except_h2("h2") is False

    _ = ad_checkpoint.get_saved_checkpoint_names()

    @zero_jax.checkpoint(policy=only_h1)
    def forward_block(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """Compute a two-stage checkpointed block with named intermediates.

        Args:
            x: Input vector.
            w: Weight vector.

        Returns:
            jnp.ndarray: Scalar loss output.
        """
        h1 = ad_checkpoint.checkpoint_name(x * w, "h1")
        h2 = ad_checkpoint.checkpoint_name(h1 + x, "h2")
        return jnp.sum(h2 * h2)

    x_in = jnp.array([1.0, 2.0])
    w_in = jnp.array([3.0, 4.0])
    loss_val = forward_block(x_in, w_in)
    saved_names = ad_checkpoint.get_saved_checkpoint_names()
    assert saved_names == ["h1"]
    zero_chex.assert_trees_all_close(loss_val, jnp.array(116.0))

    # Trace & differentiate through checkpoint_name and Cast
    jit_block = zero_jax.jit(zero_jax.checkpoint(forward_block, policy="dots_saveable"))
    zero_chex.assert_trees_all_close(jit_block(x_in, w_in), jnp.array(116.0))

    grad_w = zero_jax.grad(forward_block, argnums=1)(x_in, w_in)
    zero_chex.assert_trees_all_close(grad_w, jnp.array([8.0, 40.0]))
    _ = ad_checkpoint.get_saved_checkpoint_names()

    # Direct decorator usage (@zero_jax.remat) and Flax nnx.Remat
    @zero_jax.remat
    def simple_fn(v: jnp.ndarray) -> jnp.ndarray:
        """Compute squared sum under default remat policy.

        Args:
            v: Input array.

        Returns:
            jnp.ndarray: Squared sum.
        """
        tagged = ad_checkpoint.checkpoint_name(v * 2.0, "doubled")
        return jnp.sum(tagged)

    assert simple_fn(jnp.array([1.0, 2.0])).item() == 6.0
    assert ad_checkpoint.get_saved_checkpoint_names() == ["doubled"]

    # Non-ndarray leaf inside traced checkpoint_name
    @zero_jax.jit
    def traced_with_meta(v: jnp.ndarray) -> jnp.ndarray:
        """Tag a PyTree containing both an ndarray and a static integer leaf.

        Args:
            v: Input array.

        Returns:
            jnp.ndarray: Output array.
        """
        tagged_tree = ad_checkpoint.checkpoint_name({"arr": v, "meta": 1}, "mixed")
        return tagged_tree["arr"] + 1.0

    zero_chex.assert_trees_all_close(traced_with_meta(jnp.array([1.0, 2.0])), jnp.array([2.0, 3.0]))
    _ = ad_checkpoint.get_saved_checkpoint_names()

    layer = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    remat_layer = nnx.Remat(layer, prevent_cse=True, policy=ad_checkpoint.checkpoint_dots)
    out_remat = remat_layer(jnp.ones((2, 2)))
    assert out_remat.shape == (2, 2)


def test_pytree_scan_vmap_jit_grad_and_ecosystem_training(tmp_path: Path) -> None:
    """Verify end-to-end training loop combining Flax NNX, Optax, Grain, Chex, and Orbax over zero-jax.

    Args:
        tmp_path: Temporary directory path fixture.
    """

    # 1. PyTree lax.scan with tuple carry (c, h), dict xs, reverse=True, and unroll=2
    def lstm_like_step(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        x_step: dict[str, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], dict[str, jnp.ndarray]]:
        """Update ``(c, h)`` tuple carry from dictionary input ``x_step``.

        Args:
            carry: Tuple ``(c, h)`` of state vectors.
            x_step: Step dictionary containing ``"inp"``.

        Returns:
            tuple[tuple[jnp.ndarray, jnp.ndarray], dict[str, jnp.ndarray]]: Updated carry and step output.
        """
        c, h = carry
        new_c = c + x_step["inp"]
        new_h = h + new_c
        return (new_c, new_h), {"out": new_h}

    init_carry = (jnp.zeros((2,)), jnp.zeros((2,)))
    xs_tree = {"inp": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
    final_carry, ys_out = lax.scan(lstm_like_step, init_carry, xs_tree, reverse=False, unroll=2)
    zero_chex.assert_trees_all_close(final_carry[0], jnp.array([9.0, 12.0]))
    zero_chex.assert_trees_all_close(ys_out["out"], jnp.array([[1.0, 2.0], [5.0, 8.0], [14.0, 20.0]]))

    rev_carry, rev_ys = lax.scan(lstm_like_step, init_carry, xs_tree, reverse=True, unroll=1)
    zero_chex.assert_trees_all_close(rev_carry[0], jnp.array([9.0, 12.0]))
    assert rev_ys["out"].shape == (3, 2)

    # 2. JIT trace hooks and _JitWrapped notification
    traced_events: list[PyTree] = []
    _JitWrapped.clear_trace_hooks()
    _JitWrapped.register_trace_hook(lambda fn, key: traced_events.append((fn.__name__, key)))

    @zero_jax.jit
    def compiled_linear(state: nnx.State, graphdef: nnx.GraphDef, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate merged NNX module inside JIT.

        Args:
            state: Split NNX parameter state.
            graphdef: Static module graph definition.
            x: Input batch array.

        Returns:
            jnp.ndarray: Output predictions.
        """
        model = nnx.merge(graphdef, state)
        return model(x)

    # 3. Structured PyTree in_axes / out_axes and collective axis_name in vmap
    state_in = {"scale": jnp.array([2.0, 3.0]), "bias": jnp.array(1.0)}
    axes_spec = {"scale": 0, "bias": None}
    vmapped_affine = zero_jax.vmap(
        lambda st, x: {"y": st["scale"] * x + st["bias"], "shared": st["bias"]},
        in_axes=(axes_spec, 0),
        out_axes={"y": 0, "shared": None},
    )
    v_res = vmapped_affine(state_in, jnp.array([10.0, 20.0]))
    zero_chex.assert_trees_all_close(v_res["y"], jnp.array([21.0, 61.0]))
    zero_chex.assert_trees_all_close(v_res["shared"], jnp.array(1.0))

    # Collective ops (psum, pmean, all_gather) inside vmap with axis_name
    def collective_step(v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute psum, pmean, and all_gather across mapped axis ``"batch"``.

        Args:
            v: Per-replica scalar or vector.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Sum, mean, and gathered array.
        """
        return (
            lax.psum(v, axis_name="batch"),
            lax.pmean(v, axis_name="batch"),
            lax.all_gather(v, axis_name="batch"),
        )

    c_sum, c_mean, c_gather = zero_jax.vmap(collective_step, axis_name="batch")(jnp.array([2.0, 4.0, 6.0]))
    zero_chex.assert_trees_all_close(c_sum, jnp.array([12.0, 12.0, 12.0]))
    zero_chex.assert_trees_all_close(c_mean, jnp.array([4.0, 4.0, 4.0]))
    zero_chex.assert_trees_all_close(c_gather, jnp.array([[2.0, 4.0, 6.0]] * 3))

    # Also test vmap with axis_name when no collectives are called inside fun
    no_coll = zero_jax.vmap(lambda v: v * 2.0, axis_name="batch", out_axes=1)(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    assert no_coll.shape == (2, 2)

    # 4. Grain dataset -> Flax NNX + Optax + JIT + value_and_grad -> Orbax checkpointing
    samples = [
        {"x": jnp.array([[1.0, 0.0], [0.0, 1.0]]), "y": jnp.array([[2.0], [3.0]])},
        {"x": jnp.array([[1.0, 1.0], [2.0, 0.0]]), "y": jnp.array([[5.0], [4.0]])},
    ]
    dataset = zero_grain.MapDataset.source(samples)
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(42), kernel_metadata={"role": "weights"})
    graphdef, state = nnx.split(model)
    tx = zero_optax.adam(learning_rate=0.05)
    opt_state = tx.init(state)

    @zero_jax.jit
    def forward_loss(
        st: nnx.State,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate JIT-compiled mean squared error for state ``st``.

        Args:
            st: NNX state dictionary.
            x_batch: Input feature matrix.
            y_batch: Target output matrix.

        Returns:
            jnp.ndarray: Scalar MSE loss.
        """
        m = nnx.merge(graphdef, st)
        preds = m(x_batch)
        return jnp.mean(jnp.asarray(zero_optax.l2_loss(preds._tensor, y_batch._tensor)))

    def loss_fn(
        st: nnx.State,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate differentiable MSE loss for state ``st``.

        Args:
            st: NNX state dictionary.
            x_batch: Input feature matrix.
            y_batch: Target output matrix.

        Returns:
            jnp.ndarray: Scalar MSE loss.
        """
        m = nnx.merge(graphdef, st)
        preds = m(x_batch)
        return jnp.mean(jnp.asarray(zero_optax.l2_loss(preds._tensor, y_batch._tensor)))

    losses: list[float] = []
    for batch in dataset:
        jit_loss = forward_loss(state, batch["x"], batch["y"])
        loss_val, grads = zero_jax.value_and_grad(loss_fn)(state, batch["x"], batch["y"])
        zero_chex.assert_trees_all_close(jit_loss, loss_val)
        assert isinstance(grads, nnx.State)
        assert grads["kernel"].get_metadata("role") == "weights"
        updates, opt_state = tx.update(grads, opt_state, state)
        state = nnx.State.from_optax_tree(zero_optax.apply_updates(state, updates), template=state)
        losses.append(float(loss_val.item()))

    assert len(losses) == 2
    assert len(traced_events) >= 1
    _JitWrapped.clear_trace_hooks()

    # Save and restore trained state with zero_orbax
    ckpt_dir = tmp_path / "orbax_ckpt"
    checkpointer = zero_orbax.checkpoint.StandardCheckpointer()
    pure_state = {"kernel": state["kernel"].value, "bias": state["bias"].value}
    checkpointer.save(ckpt_dir, pure_state)
    restored = checkpointer.restore(ckpt_dir)
    zero_chex.assert_trees_all_close(jnp.asarray(restored["kernel"]), state["kernel"].value)


def test_sharding_and_host_device_count_hooks() -> None:
    """Verify set_host_device_count, sharding properties, Chex sharding assertions, and Orbax GlobalArraySharding."""
    with zero_jax.set_host_device_count(4):
        zero_chex.set_n_cpu_devices(4)
        try:
            assert zero_jax.device_count() == 4
            assert zero_jax.local_device_count() == 4
            devs = zero_jax.devices("cpu")
            local_devs = zero_jax.local_devices("cpu")
            assert len(devs) == 4 and len(local_devs) == 4
            assert devs[0] == zero_jax.Device("cpu", 0)
            assert hash(devs[0]) == hash(zero_jax.Device("cpu", 0))
            assert repr(devs[1]) == "cpu:1"

            shard_opts = zero_grain.ShardByJaxProcess(drop_remainder=True)
            assert shard_opts.shard_index == 0
            assert shard_opts.shard_count in (1, 4)
        finally:
            zero_chex.set_n_cpu_devices(1)

    assert zero_jax.device_count() == 1

    # Sharding canonical properties (.mesh, .spec, .devices, .axes)
    base_sharding = Sharding(devices=("cpu:0", "cpu:1"))
    assert base_sharding.devices == ("cpu:0", "cpu:1")
    assert base_sharding.mesh is None
    assert base_sharding.spec is None

    mesh = Mesh(devices=("cpu:0", "cpu:1"), axis_names=("data",))
    pspec = PartitionSpec("data", None)
    named = NamedSharding(mesh, pspec)
    assert mesh.mesh is mesh
    assert pspec.spec is pspec
    assert pspec.axes == ("data", None)
    assert named.mesh is mesh
    assert named.spec is pspec
    assert named.devices == ("cpu:0", "cpu:1")

    pos_sharding = PositionalSharding(("cpu:0", "cpu:1"))
    single_sharding = SingleDeviceSharding("cpu:0")
    assert pos_sharding.devices == ("cpu:0", "cpu:1")
    assert single_sharding.devices == ("cpu:0",)

    # Default sharding on ndarray and device_put with Sharding / device string
    raw_arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(raw_arr.sharding, SingleDeviceSharding)
    assert zero_jax.device_put(raw_arr, None) is raw_arr
    single_put = zero_jax.device_put(raw_arr, "cpu:0")
    assert isinstance(single_put.sharding, SingleDeviceSharding)

    sharded_tree = zero_jax.device_put({"a": raw_arr, "meta": "static"}, named)
    assert sharded_tree["a"].sharding is named
    zero_chex.assert_tree_is_sharded({"a": sharded_tree["a"]}, devices=("cpu:0", "cpu:1"))

    # Orbax GlobalArraySharding compatibility with zero_jax Mesh and PartitionSpec
    orbax_sharding = GlobalArraySharding()
    sliced = orbax_sharding.get_local_shard(raw_arr, host_id=0, num_hosts=2, partition_spec=pspec, mesh=mesh)
    assert sliced.shape == (1, 2)


def test_internal_helpers_coverage() -> None:
    """Exercise remaining edge branches in _compiler_patches, _resolve_leaf_axes, and _reduce_grad_to_shape."""
    # _reduce_grad_to_shape singleton axis summation
    raw_g = jnp.ones((2, 3, 4))
    reduced_g = _reduce_grad_to_shape(raw_g, (1, 4))
    assert reduced_g.shape == (1, 4)

    # _resolve_leaf_axes tuple and fallback branches
    assert _resolve_leaf_axes((1, 2), (0, None)) == [0, None]
    assert _resolve_leaf_axes({"a": 1, "b": 2}, "invalid_spec") == [0, 0]

    # 1D dot VJP branch
    grad_1d = zero_jax.grad(lambda v: jnp.dot(v, v))(jnp.array([2.0, 3.0]))
    zero_chex.assert_trees_all_close(grad_1d, jnp.array([4.0, 6.0]))

    # Exercise plain flatten registered by register_dataclass
    @zero_chex.dataclass
    class _RegBox:
        """Simple registered dataclass container."""

        val: int

    register_dataclass(_RegBox, data_fields=["val"], meta_fields=[])
    from zero_jax.tree_util.pytree import _pytree_registry

    assert _pytree_registry[_RegBox][0](_RegBox(val=7))[0] == [7]

    # _np_cast branches (DType, str, bfloat16, int4, and kwargs fallback)
    from ml_switcheroo_compiler.backends.numpy.eager import np as _np
    from ml_switcheroo_ir import DType

    arr_in = _np.array([1.0, 2.0], dtype=_np.float32)
    assert compiler_patches._np_cast(_np, arr_in, DType.Float64).dtype == _np.float64
    assert compiler_patches._np_cast(_np, arr_in, "bfloat16").dtype == _np.float32
    assert compiler_patches._np_cast(_np, arr_in, "int4").dtype == _np.int8
    assert compiler_patches._np_cast(_np, arr_in, dtype="float32").dtype == _np.float32
    assert compiler_patches._np_checkpoint_name(_np, arr_in).shape == (2,)

    assert abs(compiler_patches._scalar_polygamma(0, 1.0) - (-0.57721566)) < 1e-4
    assert compiler_patches._scalar_betainc(1.0, 1.0, 0.0) == 0.0
    assert compiler_patches._scalar_betainc(1.0, 1.0, 1.0) == 1.0
    assert compiler_patches._scalar_erfinv(-1.0) == float("-inf")
    assert compiler_patches._scalar_erfinv(1.0) == float("inf")
    assert compiler_patches._np_cross(_np, _np.array([1.0, 0.0, 0.0]), _np.array([0.0, 1.0, 0.0]), axes={"axis": -1, "axisa": None}, axis=-1, label="x").shape == (3,)
    assert compiler_patches._np_lu(_np, _np.zeros((2, 2)))[0].shape == (2, 2)
