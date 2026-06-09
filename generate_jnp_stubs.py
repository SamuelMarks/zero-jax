import numpy as np

# Get common numpy functions that are typical in jnp
jnp_funcs = [
    name
    for name in dir(np)
    if callable(getattr(np, name)) and not name.startswith("_") and name.islower()
]
# Filter out some numpy specific stuff
jnp_funcs = [
    f for f in jnp_funcs if f not in ["test", "show_config", "errstate", "seterr"]
]

# Just take 150 of them
jnp_funcs = sorted(jnp_funcs)[:160]

with open("/Users/samuel/repos/zero-jax/tests/test_jnp.py", "w") as f:
    f.write('"""Test stubs for zero_jax.numpy (jnp) API parity."""\n\n')
    f.write("import pytest\n\n")
    for func in jnp_funcs:
        f.write('@pytest.mark.skip(reason="Not implemented")\n')
        f.write(f"def test_jnp_{func}():\n")
        f.write(f'    """Test zero_jax.numpy.{func}."""\n')
        f.write("    pass\n\n")

with open("/Users/samuel/repos/zero-jax/tests/test_lax.py", "w") as f:
    f.write('"""Test stubs for zero_jax.lax primitives."""\n\n')
    f.write("import pytest\n\n")
    for func in ["scan", "cond", "while_loop"]:
        f.write('@pytest.mark.skip(reason="Not implemented")\n')
        f.write(f"def test_lax_{func}():\n")
        f.write(f'    """Test zero_jax.lax.{func}."""\n')
        f.write("    pass\n\n")

with open("/Users/samuel/repos/zero-jax/tests/test_tree_util.py", "w") as f:
    f.write('"""Test stubs for zero_jax.tree_util."""\n\n')
    f.write("import pytest\n\n")
    for func in ["tree_flatten", "tree_unflatten"]:
        f.write('@pytest.mark.skip(reason="Not implemented")\n')
        f.write(f"def test_tree_{func}():\n")
        f.write(f'    """Test zero_jax.tree_util.{func}."""\n')
        f.write("    pass\n\n")

with open("/Users/samuel/repos/zero-jax/tests/test_transformations.py", "w") as f:
    f.write('"""Test stubs for zero_jax transformations."""\n\n')
    f.write("import pytest\n\n")
    for func in ["jit", "grad", "value_and_grad", "vmap"]:
        f.write('@pytest.mark.skip(reason="Not implemented")\n')
        f.write(f"def test_transform_{func}():\n")
        f.write(f'    """Test zero_jax.{func}."""\n')
        f.write("    pass\n\n")

with open("/Users/samuel/repos/zero-jax/tests/test_pygrain.py", "w") as f:
    f.write('"""Test stubs for zero_grain (pygrain) parity."""\n\n')
    f.write("import pytest\n\n")
    for func in ["MapDataset", "IterDataset", "BatchOperation", "FilterOperation"]:
        f.write('@pytest.mark.skip(reason="Not implemented")\n')
        f.write(f"def test_pygrain_{func.lower()}():\n")
        f.write(f'    """Test zero_grain.{func}."""\n')
        f.write("    pass\n\n")
