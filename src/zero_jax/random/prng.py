"""PRNG state manipulation."""

from __future__ import annotations

from ml_switcheroo_compiler import random

from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def split(key: ArrayLikeSpec, num: int = 2) -> ArrayLikeSpec:
    """Splits a PRNG key into `num` new keys.

    Args:
        key: The original PRNG key.
        num: The number of new keys to generate. Defaults to 2.

    Returns:
        An array containing `num` new PRNG keys.
    """
    if key is None:
        return None
    from ml_switcheroo_compiler.ops import random_stateless

    return _wrap(random_stateless.stateless_split(_to_tensor(key), num))


def fold_in(key: ArrayLikeSpec, data: ArrayLikeSpec) -> ArrayLikeSpec:
    """Folds in data to a PRNG key to derive a new key.

    Args:
        key: The original PRNG key.
        data: The integer data to fold in.

    Returns:
        A new PRNG key derived from the original key and data.
    """
    if key is None:
        return None
    from ml_switcheroo_compiler.ops import random_stateless

    key_t = _to_tensor(key)
    return _wrap(random_stateless.stateless_fold_in(key_t, data))


def PRNGKey(seed: int, *, impl: ConfigValue = None) -> ArrayLikeSpec:
    """Creates a PRNG key given an integer seed.

    Args:
        seed: The integer seed for the PRNG key.
        impl: Optional PRNG implementation specification.

    Returns:
        A PRNG key.
    """
    _ = impl
    from zero_jax.numpy.lax_numpy import array

    return array([0, int(seed)])


def uniform(
    key: ArrayLikeSpec,
    shape: ArrayLikeSpec = (),
    dtype: ArrayLikeSpec = None,
    minval: float = 0.0,
    maxval: float = 1.0,
    *,
    out_sharding: ConfigValue = None,
) -> ArrayLikeSpec:
    """Samples uniform random values from a given key.

    Args:
        key: The PRNG key to use for sampling.
        shape: The shape of the output array.
        dtype: The dtype of the output array.
        minval: The minimum value of the uniform distribution.
        maxval: The maximum value of the uniform distribution.
        out_sharding: Optional output sharding specification.

    Returns:
        An array of uniform random values.
    """
    _ = out_sharding
    from ml_switcheroo_compiler.core.config import config

    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(random.uniform(_to_tensor(key), shape=shape, dtype=dtype, minval=minval, maxval=maxval))


def normal(
    key: ArrayLikeSpec,
    shape: ArrayLikeSpec = (),
    dtype: ArrayLikeSpec = None,
    *,
    out_sharding: ConfigValue = None,
) -> ArrayLikeSpec:
    """Samples standard normal random values from a given key.

    Args:
        key: The PRNG key to use for sampling.
        shape: The shape of the output array.
        dtype: The dtype of the output array.
        out_sharding: Optional output sharding specification.

    Returns:
        An array of standard normal random values.
    """
    _ = out_sharding
    from ml_switcheroo_compiler.core.config import config

    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(random.normal(_to_tensor(key), shape=shape, dtype=dtype))


def randint(
    key: ArrayLikeSpec,
    shape: ArrayLikeSpec,
    minval: int,
    maxval: int,
    dtype: ArrayLikeSpec = None,
    *,
    out_sharding: ConfigValue = None,
) -> ArrayLikeSpec:
    """Samples uniform random integers from a given key.

    Args:
        key: The PRNG key to use for sampling.
        shape: The shape of the output array.
        minval: The minimum integer value (inclusive).
        maxval: The maximum integer value (exclusive).
        dtype: The integer dtype of the output array.
        out_sharding: Optional output sharding specification.

    Returns:
        An array of uniform random integers.
    """
    _ = out_sharding
    from ml_switcheroo_compiler.core.dtype import DType

    if dtype is None:
        dtype = DType.Int32
    return _wrap(random.randint(_to_tensor(key), shape=shape, minval=minval, maxval=maxval, dtype=dtype))


def bernoulli(
    key: ArrayLikeSpec,
    p: float = 0.5,
    shape: ArrayLikeSpec = None,
    mode: str = "low",
    *,
    out_sharding: ConfigValue = None,
) -> ArrayLikeSpec:
    """Samples Bernoulli random variables from a given key.

    Args:
        key: The PRNG key to use for sampling.
        p: The probability of success (getting 1).
        shape: The shape of the output array.
        mode: Precision mode (``"low"`` or ``"high"``).
        out_sharding: Optional output sharding specification.

    Returns:
        A boolean array of Bernoulli random variables.
    """
    _ = (mode, out_sharding)
    if shape is None:
        shape = ()
    return _wrap(random.bernoulli(_to_tensor(key), p=p, shape=shape))


def categorical(
    key: ArrayLikeSpec,
    logits: ArrayLikeSpec,
    axis: int = -1,
    shape: ArrayLikeSpec = None,
    *,
    replace: bool = True,
    mode: str | None = None,
    out_sharding: ConfigValue = None,
) -> ArrayLikeSpec:
    """Samples categorical random variables from a given key.

    Args:
        key: The PRNG key to use for sampling.
        logits: Unnormalized log probabilities.
        axis: The axis along which to sample.
        shape: The shape of the output array.
        replace: Whether to sample with replacement.
        mode: Optional sampling precision mode.
        out_sharding: Optional output sharding specification.

    Returns:
        An array of categorical random samples.
    """
    _ = (replace, mode, out_sharding)
    from ml_switcheroo_compiler import random

    return _wrap(random.categorical(_to_tensor(key), _to_tensor(logits), axis=axis, shape=shape))


def permutation(
    key: ArrayLikeSpec,
    x: ArrayLikeSpec,
    axis: int = 0,
    independent: bool = False,
    *,
    out_sharding: ConfigValue = None,
) -> ArrayLikeSpec:
    """Randomly permutes a sequence or array.

    Args:
        key: The PRNG key to use for permutation.
        x: An integer specifying the range (if an int) or an array to permute.
        axis: The axis along which to permute.
        independent: Whether to permute independently along other axes.
        out_sharding: Optional output sharding specification.

    Returns:
        A randomly permuted sequence or array.
    """
    _ = out_sharding
    from ml_switcheroo_compiler import random

    from zero_jax._types import ArrayProtocol

    return _wrap(
        random.permutation(
            _to_tensor(key),
            _to_tensor(x) if isinstance(x, ArrayProtocol) else x,
            axis=axis,
            independent=independent,
        )
    )


def choice(
    key: ArrayLikeSpec,
    a: ArrayLikeSpec,
    shape: ArrayLikeSpec = (),
    replace: bool = True,
    p: ArrayLikeSpec = None,
    axis: int = 0,
    mode: str | None = None,
) -> ArrayLikeSpec:
    """Generates a random sample from a given 1-D array.

    Args:
        key: The PRNG key to use for sampling.
        a: A 1-D array or integer specifying the choices.
        shape: The shape of the output array.
        replace: Whether the sample is with or without replacement.
        p: The probabilities associated with each entry in a.
        axis: The axis along which to perform the selection.
        mode: Optional sampling mode.

    Returns:
        An array containing the random samples.
    """
    _ = mode
    from ml_switcheroo_compiler import random

    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

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


def truncated_normal(
    key: ArrayLikeSpec,
    lower: ArrayLikeSpec,
    upper: ArrayLikeSpec,
    shape: ArrayLikeSpec = None,
    dtype: ArrayLikeSpec = None,
    *,
    out_sharding: ConfigValue = None,
) -> ArrayLikeSpec:
    """Samples truncated normal random values from a given key.

    Args:
        key: The PRNG key to use for sampling.
        lower: The lower bound.
        upper: The upper bound.
        shape: The shape of the output array.
        dtype: The dtype of the output array.
        out_sharding: Optional output sharding specification.

    Returns:
        An array of truncated normal random values.
    """
    _ = out_sharding
    from ml_switcheroo_compiler.core.config import config

    if dtype is None:
        dtype = config.default_float_dtype
    if shape is None:
        shape = ()
    return _wrap(
        random.truncated_normal(
            _to_tensor(key),
            _to_tensor(lower),
            _to_tensor(upper),
            shape=shape,
            dtype=dtype,
        )
    )
