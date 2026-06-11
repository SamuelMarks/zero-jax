Extracting target APIs from /Users/samuel/repos/zero-jax/src...
Scoring compliance...

--- Compliance Report ---
Overall Compliance: 0.0%

Breakdown by Module:
  - jax.nn: 0.0% (0/4)
  - jax.nn.initializers: 0.0% (0/19)

Missing APIs (58):

|   | Framework | Namespace | Symbol | FQN | Signature | Docstring |
|---|---|---|---|---|---|---|
| [ ] | jax | jax.nn | gelu | jax.nn.gelu | `(x: ArrayLike, approximate: bool=True) -> jax.Array` | Gaussian error linear unit activation function. |
| [ ] | jax | jax.nn.initializers | constant | jax.nn.initializers.constant | `(value: ArrayLike, dtype: Any='```(jnp.float_)```') -> jax.nn.initializers.Initializer` | Builds an initializer that returns arrays full of a constant ``value``. |
| [ ] | jax | jax.nn.initializers | delta_orthogonal | jax.nn.initializers.delta_orthogonal | `(scale: RealNumeric=1.0, column_axis: int=-1, dtype: Any='```(jnp.float_)```') -> jax.nn.initializers.Initializer` | Builds an initializer for delta orthogonal kernels. |
| [ ] | jax | jax.nn.initializers | glorot_normal | jax.nn.initializers.glorot_normal | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a Glorot normal initializer (aka Xavier normal initializer). |
| [ ] | jax | jax.nn.initializers | glorot_uniform | jax.nn.initializers.glorot_uniform | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a Glorot uniform initializer (aka Xavier uniform initializer). |
| [ ] | jax | jax.nn.initializers | he_normal | jax.nn.initializers.he_normal | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a He normal initializer (aka Kaiming normal initializer). |
| [ ] | jax | jax.nn.initializers | he_uniform | jax.nn.initializers.he_uniform | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a He uniform initializer (aka Kaiming uniform initializer). |
| [ ] | jax | jax.nn.initializers | kaiming_normal | jax.nn.initializers.kaiming_normal | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a He normal initializer (aka Kaiming normal initializer). |
| [ ] | jax | jax.nn.initializers | kaiming_uniform | jax.nn.initializers.kaiming_uniform | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a He uniform initializer (aka Kaiming uniform initializer). |
| [ ] | jax | jax.nn.initializers | lecun_normal | jax.nn.initializers.lecun_normal | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a Lecun normal initializer. |
| [ ] | jax | jax.nn.initializers | lecun_uniform | jax.nn.initializers.lecun_uniform | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a Lecun uniform initializer. |
| [ ] | jax | jax.nn.initializers | normal | jax.nn.initializers.normal | `(stddev: RealNumeric=0.01, dtype: Any='```(jnp.float_)```') -> jax.nn.initializers.Initializer` | Builds an initializer that returns real normally-distributed random arrays. |
| [ ] | jax | jax.nn.initializers | ones | jax.nn.initializers.ones | `(key: KeyArray, shape: core.Shape, dtype: Any='```(jnp.float_)```') -> jax.Array` | An initializer that returns a constant array full of ones. |
| [ ] | jax | jax.nn.initializers | orthogonal | jax.nn.initializers.orthogonal | `(scale: RealNumeric=1.0, column_axis: int=-1, dtype: Any='```(jnp.float_)```') -> jax.nn.initializers.Initializer` | Builds an initializer that returns uniformly distributed orthogonal matrices. |
| [ ] | jax | jax.nn.initializers | truncated_normal | jax.nn.initializers.truncated_normal | `(stddev: RealNumeric=0.01, dtype: Any='```(jnp.float_)```', lower: RealNumeric=-2.0, upper: RealNumeric=2.0) -> jax.nn.initializers.Initializer` | Builds an initializer that returns truncated-normal random arrays. |
| [ ] | jax | jax.nn.initializers | uniform | jax.nn.initializers.uniform | `(scale: RealNumeric=0.01, dtype: Any='```(jnp.float_)```') -> jax.nn.initializers.Initializer` | Builds an initializer that returns real uniformly-distributed random arrays. |
| [ ] | jax | jax.nn.initializers | variance_scaling | jax.nn.initializers.variance_scaling | `(scale: RealNumeric, mode: Literal['fan_in'] | Literal['fan_out'] | Literal['fan_avg'], distribution: Literal['truncated_normal'] | Literal['normal'] | Literal['uniform'], in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Initializer that adapts its scale to the shape of the weights tensor. |
| [ ] | jax | jax.nn.initializers | xavier_normal | jax.nn.initializers.xavier_normal | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a Glorot normal initializer (aka Xavier normal initializer). |
| [ ] | jax | jax.nn.initializers | xavier_uniform | jax.nn.initializers.xavier_uniform | `(in_axis: int | Sequence[int]=-2, out_axis: int | Sequence[int]=-1, batch_axis: Sequence[int]=(), dtype='```(jnp.float_)```')` | Builds a Glorot uniform initializer (aka Xavier uniform initializer). |
| [ ] | jax | jax.nn.initializers | zeros | jax.nn.initializers.zeros | `(key: KeyArray, shape: core.Shape, dtype: Any='```(jnp.float_)```') -> jax.Array` | An initializer that returns a constant array full of zeros. |
| [ ] | jax | jax.nn | logsumexp | jax.nn.logsumexp | `(a: ArrayLike, axis: Axis='```(None)```', b: ArrayLike | None='```(None)```', keepdims: bool=False, return_sign: bool=False, where: ArrayLike | None='```(None)```')` | Log-sum-exp reduction. |
| [ ] | jax | jax.nn | one_hot | jax.nn.one_hot | `(x: Any, num_classes: int, dtype='```(jnp.float_)```', axis: int | AxisName=-1)` | One-hot encodes the given indices. |
| [ ] | jax | jax.nn | softmax | jax.nn.softmax | `(x: ArrayLike, axis: int | tuple[int, ...] | None=-1, where: ArrayLike | None='```(None)```', initial='```(_UNSPECIFIED)```')` | Softmax function. |
