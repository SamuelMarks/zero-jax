"""PRNG state manipulation."""

from typing import Any

import ml_switcheroo.random as random
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def split(key: Any, num: int = 2) -> Any:
    """Splits a PRNG key into `num` new keys."""
    if key is None:
        return None
    return _wrap(random.split(_to_tensor(key), num))


def fold_in(key: Any, data: Any) -> Any:
    """Folds in data to a PRNG key."""
    if key is None:
        return None
    return _wrap(random.fold_in(_to_tensor(key), data))


def PRNGKey(seed: int) -> Any:
    """PRNGKey function."""
    return _wrap(random.PRNGKey(seed))


def uniform(
    key: Any, shape: Any, dtype: Any = None, minval: float = 0.0, maxval: float = 1.0
) -> Any:
    """Uniform function."""
    from ml_switcheroo.core.config import config

    if dtype is None:  # pragma: no cover
        dtype = config.default_float_dtype  # pragma: no cover
    return _wrap(  # pragma: no cover
        random.uniform(
            _to_tensor(key), shape=shape, dtype=dtype, minval=minval, maxval=maxval
        )
    )


def normal(key: Any, shape: Any, dtype: Any = None) -> Any:
    """Normal function."""
    from ml_switcheroo.core.config import config

    if dtype is None:  # pragma: no cover
        dtype = config.default_float_dtype  # pragma: no cover
    return _wrap(
        random.normal(_to_tensor(key), shape=shape, dtype=dtype)
    )  # pragma: no cover


def randint(key: Any, shape: Any, minval: int, maxval: int, dtype: Any = None) -> Any:
    """Randint function."""
    from ml_switcheroo.core.dtype import DType

    if dtype is None:  # pragma: no cover
        dtype = DType.Int32  # pragma: no cover
    return _wrap(  # pragma: no cover
        random.randint(
            _to_tensor(key), shape=shape, minval=minval, maxval=maxval, dtype=dtype
        )
    )


def bernoulli(key: Any, p: float = 0.5, shape: Any = None) -> Any:
    """Bernoulli function."""
    if shape is None:  # pragma: no cover
        shape = ()  # pragma: no cover
    return _wrap(
        random.bernoulli(_to_tensor(key), p=p, shape=shape)
    )  # pragma: no cover


def categorical(key: Any, logits: Any, axis: int = -1, shape: Any = None) -> Any:
    """Categorical function."""
    # Eager hack since native not in Switcheroo easily
    t = _to_tensor(logits)
    import numpy as np
    import ml_switcheroo
    from zero_jax.numpy.lax_numpy import _wrap

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        res = np.random.choice(
            t.data.shape[axis], p=np.exp(t.data) / np.sum(np.exp(t.data))
        )
        return _wrap(
            ml_switcheroo.Tensor(
                np.array(res), (), ml_switcheroo.core.dtype.DType.Int32, t.device
            )
        )
    raise NotImplementedError("categorical purely eager")  # pragma: no cover


def permutation(key: Any, x: Any, axis: int = 0, independent: bool = False) -> Any:
    """Permutation function."""
    t = _to_tensor(x)
    import numpy as np
    import ml_switcheroo
    from zero_jax.numpy.lax_numpy import _wrap

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        res = np.random.permutation(t.data)
        return _wrap(ml_switcheroo.Tensor(res, res.shape, t.dtype, t.device))
    raise NotImplementedError("permutation purely eager")  # pragma: no cover


def choice(
    key: Any,
    a: Any,
    shape: Any = (),
    replace: bool = True,
    p: Any = None,
    axis: int = 0,
) -> Any:
    """Choice function."""
    t = _to_tensor(a)
    import numpy as np
    import ml_switcheroo
    from zero_jax.numpy.lax_numpy import _wrap

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        res = np.random.choice(t.data, size=shape, replace=replace)
        return _wrap(ml_switcheroo.Tensor(res, res.shape, t.dtype, t.device))
    raise NotImplementedError("choice purely eager")  # pragma: no cover
