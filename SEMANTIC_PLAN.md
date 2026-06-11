# Semantic Implementation Plan

This document tracks the status of the mathematical and semantic implementations for the `zero_jax` APIs.

## `jax.nn`
- [x] `gelu`: Fully mathematically implemented (both exact and approximate formulations).
- [x] `logsumexp`: Fully mathematically implemented (numerically stable max-subtraction trick).
- [x] `one_hot`: Fully mathematically implemented (correctly inserts new axis and sets indices).
- [x] `softmax`: Fully mathematically implemented (numerically stable log-sum-exp trick).

## `jax.nn.initializers`
- [x] `constant`: Fully mathematically implemented.
- [x] `ones`: Fully mathematically implemented.
- [x] `zeros`: Fully mathematically implemented.
- [x] `uniform`: Fully mathematically implemented.
- [x] `normal`: Fully mathematically implemented.
- [x] `truncated_normal`: Fully mathematically implemented (via rejection sampling).
- [x] `variance_scaling`: Fully mathematically implemented (computes correct fan-in/fan-out/fan-avg based on receptive fields).
- [x] `glorot_normal`: Fully mathematically implemented (maps to `variance_scaling`).
- [x] `glorot_uniform`: Fully mathematically implemented (maps to `variance_scaling`).
- [x] `he_normal` / `kaiming_normal`: Fully mathematically implemented (maps to `variance_scaling`).
- [x] `he_uniform` / `kaiming_uniform`: Fully mathematically implemented (maps to `variance_scaling`).
- [x] `lecun_normal`: Fully mathematically implemented (maps to `variance_scaling`).
- [x] `lecun_uniform`: Fully mathematically implemented (maps to `variance_scaling`).
- [x] `orthogonal`: Fully mathematically implemented (generates standard normal matrix, performs QR decomposition, handles I/O axes swapping).
- [x] `delta_orthogonal`: Fully mathematically implemented (generates orthogonal core matrix and embeds it at the spatial center of the kernel).

