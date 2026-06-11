# Delivery Plan: Porting Official JAX Test Suite to `zero-jax`

This plan outlines the steps required to port the official JAX test suite to verify that `zero-jax` outputs are 1-to-1 compatible (or `allclose` where applicable) with the official `jax` reference implementation.

## Phase 1: Test Infrastructure Setup
- [x] Determine the specific version of `jax` and `jaxlib` to use as the reference implementation.
- [x] Create `requirements-test.txt` containing dependencies for comparative testing (`pytest`, `numpy`, `jax`, `jaxlib`, etc.).
- [x] Implement a dual-testing fixture/utility in `tests/conftest.py` that allows running the same test payload against both `zero_jax` and official `jax`.
- [x] Add assertion helpers (e.g., `check_allclose`) to compare complex pytrees, arrays, and scalars outputted by both libraries.
- [ ] Configure `pytest.ini` or `pyproject.toml` to easily toggle reference tests and handle expected failures.
- [ ] Create a utility to programmatically generate random inputs of various shapes and dtypes for fuzz testing.
- [ ] Setup a mechanism to skip tests if official `jax` is not installed (optional, but good for pure `zero-jax` environments).

## Phase 2: Porting `jax.numpy` (JNP) Tests
- [ ] Port/write parity tests for Array Creation:
    - [ ] `zeros`, `ones`, `empty`, `full`
    - [ ] `zeros_like`, `ones_like`, `empty_like`, `full_like`
    - [ ] `array`, `asarray`
    - [ ] `arange`, `linspace`, `logspace`
    - [ ] `eye`, `identity`
    - [ ] `meshgrid`
- [ ] Port/write parity tests for Basic Math Operations:
    - [ ] `add`, `sub`, `multiply`, `divide`, `true_divide`, `floor_divide`
    - [ ] `power`, `mod`, `remainder`, `divmod`
    - [ ] `abs`, `negative`, `positive`
    - [ ] `sign`, `rint`, `floor`, `ceil`, `trunc`
- [ ] Port/write parity tests for Trigonometric and Exponential Functions:
    - [ ] `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`
    - [ ] `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`
    - [ ] `exp`, `exp2`, `expm1`
    - [ ] `log`, `log2`, `log10`, `log1p`
- [ ] Port/write parity tests for Reduction Operations (testing various `axis` and `keepdims` arguments):
    - [ ] `sum`, `prod`
    - [ ] `mean`, `var`, `std`
    - [ ] `max`, `min`, `amax`, `amin`
    - [ ] `argmax`, `argmin`
    - [ ] `any`, `all`
- [ ] Port/write parity tests for Array Manipulation:
    - [ ] `reshape`, `ravel`, `squeeze`, `expand_dims`
    - [ ] `transpose`, `swapaxes`, `moveaxis`
    - [ ] `concatenate`, `stack`, `vstack`, `hstack`, `dstack`
    - [ ] `split`, `array_split`, `vsplit`, `hsplit`, `dsplit`
    - [ ] `tile`, `repeat`
    - [ ] `pad`
- [ ] Port/write parity tests for Indexing and Slicing:
    - [ ] Basic slicing (e.g., `x[1:3]`)
    - [ ] Advanced indexing (e.g., integer arrays, boolean masks)
    - [ ] `where`, `take`, `take_along_axis`
- [ ] Port/write parity tests for Linear Algebra (`jnp.linalg`):
    - [ ] `dot`, `vdot`, `inner`, `outer`, `matmul`, `tensordot`, `einsum`
    - [ ] `norm`
    - [ ] (If supported) `cholesky`, `svd`, `qr`, `inv`, `solve`
- [ ] Fix any failing `zero_jax.numpy` implementations to match reference behavior exactly.

## Phase 3: Porting `jax.nn` Tests
- [ ] Port/write parity tests for Activations:
    - [ ] `relu`, `relu6`
    - [ ] `sigmoid`, `hard_sigmoid`
    - [ ] `tanh`, `hard_tanh`
    - [ ] `gelu`, `swish`, `silu`
    - [ ] `elu`, `celu`, `selu`
    - [ ] `softmax`, `log_softmax`
- [ ] Port/write parity tests for Other NN Functions:
    - [ ] `one_hot`
    - [ ] `logsumexp`
- [ ] Port/write parity tests for Initializers (`jax.nn.initializers`):
    - [ ] `zeros`, `ones`, `constant`
    - [ ] `uniform`, `normal`
    - [ ] `glorot_uniform`, `glorot_normal` (Xavier)
    - [ ] `he_uniform`, `he_normal` (Kaiming)
    - [ ] `orthogonal`
- [ ] Fix any failing `zero_jax.nn` implementations to match reference behavior exactly.

## Phase 4: Porting `jax.lax` Tests
- [ ] Port/write parity tests for Core Lax Primitives:
    - [ ] `add`, `sub`, `mul`, `div`
    - [ ] `broadcast`, `broadcast_in_dim`
    - [ ] `reshape`, `transpose`
    - [ ] `slice`, `dynamic_slice`, `dynamic_update_slice`
    - [ ] `gather`, `scatter`
    - [ ] `conv`, `conv_general_dilated`
    - [ ] `dot_general`
    - [ ] `reduce_sum`, `reduce_max`, `reduce_min`
    - [ ] `select`, `clamp`
- [ ] Port/write parity tests for Control Flow Operations:
    - [ ] `cond`
    - [ ] `switch`
    - [ ] `while_loop`
    - [ ] `fori_loop`
    - [ ] `scan`
    - [ ] `map`
- [ ] Fix any failing `zero_jax.lax` implementations to match reference behavior exactly.

## Phase 5: Porting `jax.random` Tests
- [ ] Port/write parity tests for PRNG Management:
    - [ ] `PRNGKey`, `key`
    - [ ] `split` (with various num splits)
    - [ ] `fold_in`
- [ ] Port/write parity tests for Random Sampling (ensuring exact sequence match):
    - [ ] `uniform`
    - [ ] `normal`
    - [ ] `randint`, `choice`
    - [ ] `bernoulli`
    - [ ] `categorical`
    - [ ] `permutation`, `shuffle`
- [ ] Test interactions between random functions and `jit`/`vmap`.
- [ ] Fix any failing `zero_jax.random` implementations.

## Phase 6: Porting `jax.tree_util` Tests
- [ ] Port/write parity tests for Core Tree Operations:
    - [ ] `tree_map` (with single and multiple pytrees)
    - [ ] `tree_flatten`
    - [ ] `tree_unflatten`
    - [ ] `tree_leaves`
    - [ ] `tree_structure`
    - [ ] `tree_all`, `tree_any`
    - [ ] `tree_reduce`
- [ ] Test operations on various built-in types (lists, tuples, dicts, namedtuples).
- [ ] Test custom PyTree Registration:
    - [ ] `register_pytree_node`
    - [ ] `register_pytree_node_class`
    - [ ] Test flattening/unflattening of custom classes.
- [ ] Fix any failing `zero_jax.tree_util` implementations to match reference behavior exactly.

## Phase 7: Porting `jax.api` (Transformations) Tests
- [ ] Port/write parity tests for `jit`:
    - [ ] Basic JIT compilation of functions.
    - [ ] `jit` with `static_argnums` and `static_argnames`.
    - [ ] Behavior of `jit` with global state/side effects (should match reference).
- [ ] Port/write parity tests for `vmap`:
    - [ ] Basic vectorization.
    - [ ] `vmap` with specific `in_axes` and `out_axes`.
    - [ ] Nested `vmap`.
- [ ] Port/write parity tests for Autodiff (`grad`, `value_and_grad`):
    - [ ] Gradients of scalar-output functions.
    - [ ] `grad` with `argnums` (single and multiple arguments).
    - [ ] `has_aux=True` behavior.
    - [ ] Nested `grad` (higher-order derivatives).
- [ ] Port/write parity tests for `eval_shape`:
    - [ ] Evaluating shape/dtype of complex functions without execution.
- [ ] Port/write parity tests for `vjp` and `jvp` (if supported/planned).
- [ ] Fix any failing `zero_jax.api` implementations to match reference behavior exactly.

## Phase 8: Porting `jax.experimental` Tests
- [ ] Port/write parity tests for `checkify`:
    - [ ] `checkify.check`
    - [ ] `checkify.checkify` decorator
    - [ ] Behavior with errors vs successful checks.
- [ ] Fix any failing `zero_jax.experimental.checkify` implementations to match reference behavior exactly.

## Phase 9: Integration and Edge Cases
- [ ] Test combination of transformations (e.g., `jit(vmap(grad(f)))`).
- [ ] Test behavior with different dtypes (`float32`, `float64`, `int32`, `bool_`).
- [ ] Test behavior with NaN and Inf values.
- [ ] Test handling of weak types (Python scalars vs JAX arrays).

## Phase 10: CI Integration & Maintenance
- [ ] Update `.github/workflows/ci.yml` to run the comparative test suite alongside the standard test suite.
- [ ] Set up caching for reference dependencies (if necessary) to speed up CI.
- [ ] Resolve any performance bottlenecks in running the dual test suite in CI.
- [ ] Document the process for bringing in new tests from upstream `jax` in the future or how to add parity tests for new `zero-jax` features.
