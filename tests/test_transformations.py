from zero_jax import jit, grad, value_and_grad, vmap, disable_jit, pmap


def test_jit():
    @jit
    def f(x):
        return x + 1

    assert f(1) == 2


def test_grad():
    @grad
    def f(x):
        return x * x

    assert f(2) == 4


def test_value_and_grad():
    @value_and_grad
    def f(x):
        return x * 2

    val, g = f(2)
    assert val == 4
    assert g == 2


def test_vmap():
    @vmap
    def f(x):
        return x

    assert f(5) == 5


def test_disable_jit():
    with disable_jit():
        pass


def test_pmap():
    @pmap
    def f(x):
        return x

    assert f(5) == 5
