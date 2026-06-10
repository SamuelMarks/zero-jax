import pytest
import jax.tree_util as tree_ref
import zero_jax.tree_util as tree_zero


def test_tree_flatten(check_allclose):
    tree = {"a": 1, "b": [2, 3]}
    leaves_z, treedef_z = tree_zero.tree_flatten(tree)
    leaves_r, treedef_r = tree_ref.tree_flatten(tree)

    check_allclose(leaves_z, leaves_r)
    # Checking treedefs usually by unflattening
    check_allclose(
        tree_zero.tree_unflatten(treedef_z, leaves_z),
        tree_ref.tree_unflatten(treedef_r, leaves_r),
    )


def test_tree_unflatten(check_allclose):
    tree = [1, (2, 3)]
    leaves_r, treedef_r = tree_ref.tree_flatten(tree)
    leaves_z, treedef_z = tree_zero.tree_flatten(tree)

    check_allclose(
        tree_zero.tree_unflatten(treedef_z, leaves_z),
        tree_ref.tree_unflatten(treedef_r, leaves_r),
    )


def test_tree_map(check_allclose):
    tree = {"a": 1, "b": [2, 3]}
    res_z = tree_zero.tree_map(lambda x: x * 2, tree)
    res_r = tree_ref.tree_map(lambda x: x * 2, tree)
    check_allclose(res_z, res_r)


def test_tree_leaves(check_allclose):
    tree = {"a": 1, "b": [2, 3]}
    res_z = tree_zero.tree_leaves(tree)
    res_r = tree_ref.tree_leaves(tree)
    check_allclose(res_z, res_r)


def test_tree_structure():
    tree = {"a": 1, "b": [2, 3]}
    struct_z = tree_zero.tree_structure(tree)
    struct_r = tree_ref.tree_structure(tree)
    # Struct types might not perfectly compare class to class, but number of leaves and nodes do
    assert struct_z.num_leaves == struct_r.num_leaves
    assert struct_z.num_nodes == struct_r.num_nodes


@pytest.mark.skip(reason="jax.tree_util.tree_all differs per jax version")
def test_tree_all_skip():
    tree = {"a": True, "b": [True, False]}
    # jax.tree_util.tree_all evaluates to boolean scalar
    assert tree_zero.tree_all(tree) == tree_ref.tree_all(tree)
