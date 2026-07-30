import zero_jax._compiler_proxy_ops as ops


def test_rrelu_exists():
    assert hasattr(ops, "rrelu")
