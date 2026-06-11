"""PRNG state manipulation."""

from typing import Any

import ml_switcheroo.random as random
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def split(key: Any, num: int = 2) -> Any:
    """Splits a PRNG key into `num` new keys.

    Args:
        key: The original PRNG key.
        num: The number of new keys to generate. Defaults to 2.

    Returns:
        An array containing `num` new PRNG keys.
    """
    if key is None:
        return None
    return _wrap(random.split(_to_tensor(key), num))


def fold_in(key: Any, data: Any) -> Any:
    """Folds in data to a PRNG key to derive a new key.

    Args:
        key: The original PRNG key.
        data: The integer data to fold in.

    Returns:
        A new PRNG key derived from the original key and data.
    """
    if key is None:
        return None
    return _wrap(random.fold_in(_to_tensor(key), data))


def PRNGKey(seed: int) -> Any:
    """Creates a PRNG key given an integer seed.

    Args:
        seed: The integer seed for the PRNG key.

    Returns:
        A PRNG key.
    """
    return _wrap(random.PRNGKey(seed))


def uniform(
    key: Any, shape: Any, dtype: Any = None, minval: float = 0.0, maxval: float = 1.0
) -> Any:
    """Samples uniform random values from a given key.

    Args:
        key: The PRNG key to use for sampling.
        shape: The shape of the output array.
        dtype: The dtype of the output array.
        minval: The minimum value of the uniform distribution.
        maxval: The maximum value of the uniform distribution.

    Returns:
        An array of uniform random values.
    """
    from ml_switcheroo.core.config import config

    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(
        random.uniform(
            _to_tensor(key), shape=shape, dtype=dtype, minval=minval, maxval=maxval
        )
    )


def normal(key: Any, shape: Any, dtype: Any = None) -> Any:
    """Samples standard normal random values from a given key.

    Args:
        key: The PRNG key to use for sampling.
        shape: The shape of the output array.
        dtype: The dtype of the output array.

    Returns:
        An array of standard normal random values.
    """
    from ml_switcheroo.core.config import config

    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(random.normal(_to_tensor(key), shape=shape, dtype=dtype))


def randint(key: Any, shape: Any, minval: int, maxval: int, dtype: Any = None) -> Any:
    """Samples uniform random integers from a given key.

    Args:
        key: The PRNG key to use for sampling.
        shape: The shape of the output array.
        minval: The minimum integer value (inclusive).
        maxval: The maximum integer value (exclusive).
        dtype: The integer dtype of the output array.

    Returns:
        An array of uniform random integers.
    """
    from ml_switcheroo.core.dtype import DType

    if dtype is None:
        dtype = DType.Int32
    return _wrap(
        random.randint(
            _to_tensor(key), shape=shape, minval=minval, maxval=maxval, dtype=dtype
        )
    )


def bernoulli(key: Any, p: float = 0.5, shape: Any = None) -> Any:
    """Samples Bernoulli random variables from a given key.

    Args:
        key: The PRNG key to use for sampling.
        p: The probability of success (getting 1).
        shape: The shape of the output array.

    Returns:
        A boolean array of Bernoulli random variables.
    """
    if shape is None:
        shape = ()
    return _wrap(random.bernoulli(_to_tensor(key), p=p, shape=shape))


def categorical(key: Any, logits: Any, axis: int = -1, shape: Any = None) -> Any:
    """Samples categorical random variables from a given key.

    Args:
        key: The PRNG key to use for sampling.
        logits: Unnormalized log probabilities.
        axis: The axis along which to sample.
        shape: The shape of the output array.

    Returns:
        An array of categorical random samples.
    """
    import ml_switcheroo.random as random

    return _wrap(
        random.categorical(_to_tensor(key), _to_tensor(logits), axis=axis, shape=shape)
    )


def permutation(key: Any, x: Any, axis: int = 0, independent: bool = False) -> Any:
    """Randomly permutes a sequence or array.

    Args:
        key: The PRNG key to use for permutation.
        x: An integer specifying the range (if an int) or an array to permute.
        axis: The axis along which to permute.
        independent: Whether to permute independently along other axes.

    Returns:
        A randomly permuted sequence or array.
    """
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
    """Generates a random sample from a given 1-D array.

    Args:
        key: The PRNG key to use for sampling.
        a: A 1-D array or integer specifying the choices.
        shape: The shape of the output array.
        replace: Whether the sample is with or without replacement.
        p: The probabilities associated with each entry in a.
        axis: The axis along which to perform the selection.

    Returns:
        An array containing the random samples.
    """
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
