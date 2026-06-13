def test_to_dtype_coverage():
    from zero_jax.nn.activation import _to_dtype as act_to_dtype
    from zero_jax.nn.initializers import _to_dtype as init_to_dtype
    from ml_switcheroo_compiler.core.dtype import DType
    import pytest

    for to_dtype in (act_to_dtype, init_to_dtype):
        to_dtype(DType.Float32)

        class Dummy1:
            name = "float32"

        to_dtype(Dummy1())

        to_dtype("float32")
        to_dtype(float)
        to_dtype(int)
        to_dtype(bool)

        class Dummy2:
            pass

        with pytest.raises(ValueError):
            to_dtype(Dummy2)

        with pytest.raises(ValueError):
            to_dtype(123)


def test_pytree_coverage():
    from zero_jax.tree_util.pytree import tree_leaves, tree_map

    tree_leaves({"a": 1})
    tree_map(lambda x: x, {"a": 1})


def test_pytree_rest_coverage():
    from zero_jax.tree_util.pytree import tree_map

    tree_map(lambda x, y: x + y, {"a": 1}, {"a": 2})
