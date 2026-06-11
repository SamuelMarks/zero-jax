# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# zero-jax

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-jax/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-97.1%25-green.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

## Why `zero-jax` Exists

This repository is a foundational component of the **Abstract ML Machine Ecosystem**, designed to solve the **$N \times M$ translation problem** in Machine Learning. 

Currently, the ML landscape is heavily fragmented. If you write a model in [JAX](https://github.com/google/jax), [PyTorch](https://pytorch.org/), [Keras](https://keras.io/), or [MLX](https://github.com/ml-explore/mlx) (the $N$ frontends), deploying that model efficiently across [WASM](https://webassembly.org/), [WebGPU](https://www.w3.org/TR/webgpu/), [TensorRT](https://developer.nvidia.com/tensorrt), or custom edge hardware (the $M$ backends) usually requires building and maintaining bespoke, complex translation pipelines for every single combination. 

### The Zero-Dependency Approach

`zero-jax` exists to address this by providing a **strictly zero external dependency** implementation of the JAX API surface. It relies solely on the Python Standard Library and [`numpy`](https://numpy.org/) (for eager evaluations). 

Instead of wrapping heavy C++ binaries or relying on [XLA](https://openxla.org/), `zero-jax` mimics the public JAX API—including `jnp`, `lax`, `jit`, `grad`, and `vmap`—and acts as a pure Python frontend. 

When you execute code using `zero-jax`, it dynamically traces the operations using proxy tensors and delegates the logic to the `ml-switcheroo-compiler`. This compiler maps high-level API calls into a strictly defined Intermediate Representation (IR) via proxy tensors and an AD engine. The resulting IR can then be seamlessly consumed by various backends, enabling a robust **source-to-source** and **source-to-browser** compilation pipeline.

### Part of a Larger Ecosystem

`zero-jax` is not a standalone numerical library, but rather Tier 3 of the ML Switcheroo architecture:
1. **Tier 1 (`ml-switcheroo-ir`):** Defines the canonical logical graph dialect ([ONNX](https://onnx.ai/) spec compliance).
2. **Tier 2 (`ml-switcheroo-compiler`):** The computational heart, featuring AOT tracing, ProxyTensors, reverse-mode automatic differentiation, and optimizations like Dead Code Elimination (DCE).
3. **Tier 3 (`zero-jax`):** Provides the functional foundation and JAX API parity. Pytree flattening is used to safely route state into the compiler tape.
4. **Tier 4 (Frontends & Add-ons):** Repositories like `zero-flax`, `zero-optax`, and `zero-chex` build on top of `zero-jax` to provide Neural Network layers, optimizers, and typing without any heavy external dependencies.
5. **Tier 5 (`zero-zoo`):** Headless CI pipelines that train models deterministically to assert float-for-float equivalence ("Golden Seed" testing) across all simulated frameworks.

By maintaining structural API parity with the real JAX framework (verified via `ml-framework-snapshots`), `zero-jax` allows users to drop it in as a lightweight substitute in environments where installing the massive official JAX/XLA stack is unfeasible—such as highly constrained serverless functions, or directly inside a web browser natively via [Pyodide](https://pyodide.org/) and [PyScript](https://pyscript.net/).

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
