import pytest

"""Tests for zero_jax.nn.activation."""

import numpy as np
from zero_jax.nn.activation import gelu, logsumexp, one_hot, softmax


def test_gelu():
    """Test gelu."""
    x = np.array([-1.0, 0.0, 1.0])
    y_approx = gelu(x, approximate=True)
    y_exact = gelu(x, approximate=False)
    assert y_approx.shape == x.shape
    assert y_exact.shape == x.shape


def test_logsumexp():
    """Test logsumexp."""
    x = np.array([1.0, 2.0, 3.0])
    res = logsumexp(x)
    assert hasattr(res, "shape") or isinstance(res, float)

    # Test with b, keepdims, return_sign, where
    res2, sign2 = logsumexp(
        x,
        b=np.array([1.0, 1.0, 1.0]),
        keepdims=True,
        return_sign=True,
        where=np.array([True, True, False]),
    )
    assert sign2 == 1.0

    # Test all infs
    x_inf = np.array([-np.inf, -np.inf])
    logsumexp(x_inf)


def test_one_hot():
    """Test one_hot."""
    x = np.array([0, 1, 2])
    y = one_hot(x, num_classes=3)
    assert y.shape == (3, 3)

    y2 = one_hot(x, num_classes=3, axis=0)
    assert y2.shape == (3, 3)


def test_softmax():
    """Test softmax."""
    x = np.array([1.0, 2.0, 3.0])
    y = softmax(x)
    assert np.allclose(np.sum(y), 1.0)

    # Test with where
    y2 = softmax(x, where=np.array([True, False, True]))
    assert y2[1] == 0.0


def test_sigmoid():
    from zero_jax.nn.activation import sigmoid
    import numpy as np

    x = np.array([0.0])
    assert sigmoid(x) == 0.5


def test_log_sigmoid():
    from zero_jax.nn.activation import log_sigmoid
    import numpy as np

    x = np.array([0.0])
    res = log_sigmoid(x)
    assert res < 0.0


def test_gelu_eager_non_approx():
    from zero_jax.nn.activation import gelu
    import numpy as np

    x = np.array([0.0, 1.0, -1.0])
    res2 = gelu(x, approximate=False)
    assert res2.shape == (3,)


def test_logsumexp_eager_extra():
    from zero_jax.nn.activation import logsumexp
    import numpy as np

    x = np.array([0.0, 1.0, 2.0])
    res, sign = logsumexp(
        x, b=np.array([1, 1, 1]), where=np.array([True, True, False]), return_sign=True
    )
    assert sign > 0


def test_softmax_eager_extra():
    from zero_jax.nn.activation import softmax
    import numpy as np

    x = np.array([0.0, 1.0, 2.0])
    res2 = softmax(x, where=np.array([True, True, False]))
    assert res2.shape == (3,)


def test_selu():
    from zero_jax.nn.activation import selu
    import numpy as np

    x = np.array([0.0, 1.0, -1.0])
    res = selu(x)
    assert res.shape == x.shape


def test_log_softmax():
    from zero_jax.nn.activation import log_softmax
    import numpy as np

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = log_softmax(x, axis=-1)
    assert res.shape == x.shape


def test_erf():
    from zero_jax.nn.activation import _erf
    import numpy as np

    x = np.array([0.0, 1.0, -1.0])
    res = _erf(x)
    assert res.shape == x.shape
