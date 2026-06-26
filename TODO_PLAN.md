# zero-jax (Tier 3/4) Implementation Plan

This document tracks the API routing, namespace mimicry, argument normalization, and object-oriented state tracking required to reach 100% API parity with the official JAX framework. 

All underlying logic must map directly to the `ml-switcheroo-compiler` backend. **No raw array manipulations or mathematical implementations should occur in this repository.**

## Sub-module API Routing

### `jax.scipy`
- [x] `jax.scipy.cluster`
  - [x] `vq`
- [x] `jax.scipy.fft`
  - [x] `dct`, `dctn`, `idct`, `idctn`
- [x] `jax.scipy.integrate`
  - [x] `trapezoid`
- [x] `jax.scipy.linalg`
  - [x] `block_diag`, `cho_factor`, `cho_solve`, `cholesky`, `det`, `eigh`, `eigh_tridiagonal`, `expm`, `expm_frechet`, `funm`, `hessenberg`, `hilbert`, `inv`, `lu`, `lu_factor`, `lu_solve`, `polar`, `qr`, `rsf2csf`, `schur`, `solve`, `solve_triangular`, `sqrtm`, `svd`, `toeplitz`
- [x] `jax.scipy.ndimage`
  - [x] `map_coordinates`
- [x] `jax.scipy.signal`
  - [x] `convolve`, `convolve2d`, `correlate`, `correlate2d`, `csd`, `detrend`, `fftconvolve`, `istft`, `stft`, `welch`
- [x] `jax.scipy.sparse.linalg`
  - [x] `bicgstab`, `cg`, `gmres`
- [x] `jax.scipy.special`
  - [x] `bernoulli`, `bessel_jn`, `beta`, `betainc`, `betaln`, `digamma`, `entr`, `erf`, `erfc`, `erfinv`, `exp1`, `expi`, `expit`, `expn`, `factorial`, `gamma`, `gammainc`, `gammaincc`, `gammaln`, `gammasgn`, `hyp1f1`, `i0`, `i0e`, `i1`, `i1e`, `kl_div`, `log_ndtr`, `logit`, `logsumexp`, `lpmn`, `lpmn_values`, `multigammaln`, `ndtr`, `ndtri`, `poch`, `polygamma`, `rel_entr`, `spence`, `sph_harm`, `xlog1py`, `xlogy`, `zeta`
- [x] `jax.scipy.stats`
  - [x] `bernoulli`: `cdf`, `logpmf`, `pmf`, `ppf`
  - [x] `beta`: `cdf`, `logcdf`, `logpdf`, `logsf`, `pdf`, `sf`
  - [x] `betabinom`: `logpmf`, `pmf`
  - [x] `binom`: `logpmf`, `pmf`
  - [x] `cauchy`: `cdf`, `isf`, `logcdf`, `logpdf`, `logsf`, `pdf`, `ppf`, `sf`
  - [x] `chi2`: `cdf`, `logcdf`, `logpdf`, `logsf`, `pdf`, `sf`
  - [x] `dirichlet`: `logpdf`, `pdf`
  - [x] `expon`: `logpdf`, `pdf`
  - [x] `gamma`: `cdf`, `logcdf`, `logpdf`, `logsf`, `pdf`, `sf`
  - [x] `gennorm`: `cdf`, `logpdf`, `pdf`
  - [x] `geom`: `logpmf`, `pmf`
  - [x] `laplace`: `cdf`, `logpdf`, `pdf`
  - [x] `logistic`: `cdf`, `isf`, `logpdf`, `pdf`, `ppf`, `sf`
  - [x] `multinomial`: `logpmf`, `pmf`
  - [x] `multivariate_normal`: `logpdf`, `pdf`
  - [x] `nbinom`: `logpmf`, `pmf`
  - [x] `norm`: `cdf`, `isf`, `logcdf`, `logpdf`, `logsf`, `pdf`, `ppf`, `sf`
  - [x] `pareto`: `logpdf`, `pdf`
  - [x] `poisson`: `cdf`, `logpmf`, `pmf`
  - [x] `t`: `logpdf`, `pdf`
  - [x] `truncnorm`: `cdf`, `logcdf`, `logpdf`, `logsf`, `pdf`, `sf`
  - [x] `uniform`: `cdf`, `logpdf`, `pdf`, `ppf`
  - [x] `vonmises`: `logpdf`, `pdf`
  - [x] `wrapcauchy`: `logpdf`, `pdf`

### `jax.ops`
- [x] `segment_max`
- [x] `segment_min`
- [x] `segment_prod`
- [x] `segment_sum`

### `jax.tree_util` (and `jax.tree`)
- [x] `all`
- [x] `flatten`
- [x] `leaves`
- [x] `map`
- [x] `reduce`
- [x] `structure`
- [x] `transpose`
- [x] `unflatten`

### Diagnostics & Monitoring
- [x] `jax.debug`
  - [x] `breakpoint`, `callback`, `inspect_array_sharding`, `print`, `visualize_sharding`
- [x] `jax.profiler`
  - [x] `annotate_function`, `device_memory_profile`, `start_server`, `start_trace`, `stop_server`, `stop_trace`, `trace`
- [x] `jax.monitoring`
  - [x] `clear_event_listeners`, `record_event`, `register_event_listener`

### XLA / Interpreters / Distributed Hooks
- [x] `jax.interpreters.xla` overrides
- [x] `jax.interpreters.ad` backward passes (mapping to AD tape)
- [x] `jax.core` abstractions (subst_axis, traverse, trace_state, etc.)
- [x] `jax.distributed`
  - [x] `initialize`
  - [x] `shutdown`
- [x] `jax.dlpack`
  - [x] `from_dlpack`, `to_dlpack`
- [x] `jax.custom_derivatives`
  - [x] `custom_gradient`
  - [x] `closure_convert`
