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

    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(
        random.uniform(
            _to_tensor(key), shape=shape, dtype=dtype, minval=minval, maxval=maxval
        )
    )


def normal(key: Any, shape: Any, dtype: Any = None) -> Any:
    """Normal function."""
    from ml_switcheroo.core.config import config

    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(random.normal(_to_tensor(key), shape=shape, dtype=dtype))


def randint(key: Any, shape: Any, minval: int, maxval: int, dtype: Any = None) -> Any:
    """Randint function."""
    from ml_switcheroo.core.dtype import DType

    if dtype is None:
        dtype = DType.Int32
    return _wrap(
        random.randint(
            _to_tensor(key), shape=shape, minval=minval, maxval=maxval, dtype=dtype
        )
    )


def bernoulli(key: Any, p: float = 0.5, shape: Any = None) -> Any:
    """Bernoulli function."""
    if shape is None:
        shape = ()
    return _wrap(random.bernoulli(_to_tensor(key), p=p, shape=shape))


def categorical(key: Any, logits: Any, axis: int = -1, shape: Any = None) -> Any:
    """Categorical function."""
    import ml_switcheroo.random as random

    return _wrap(
        random.categorical(_to_tensor(key), _to_tensor(logits), axis=axis, shape=shape)
    )


def permutation(key: Any, x: Any, axis: int = 0, independent: bool = False) -> Any:
    """Permutation function."""
    import ml_switcheroo.random as random

    return _wrap(
        random.permutation(
            _to_tensor(key),
            _to_tensor(x) if hasattr(x, "shape") else x,
            axis=axis,
            independent=independent,
        )
    )


def choice(
    key: Any,
    a: Any,
    shape: Any = (),
    replace: bool = True,
    p: Any = None,
    axis: int = 0,
) -> Any:
    """Choice function."""
    import ml_switcheroo.random as random
    from zero_jax.numpy.lax_numpy import _wrap, _to_tensor

    return _wrap(
        random.choice(
            _to_tensor(key),
            _to_tensor(a),
            shape=shape,
            replace=replace,
            p=_to_tensor(p) if p is not None else None,
            axis=axis,
        )
    )
