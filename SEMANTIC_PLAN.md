# Semantic Implementation Plan

This document tracks the status of the mathematical and semantic implementations for the `zero_jax` and `zero_optax` APIs.

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

## `optax.losses`
- [x] `hinge_loss`: Fully mathematically implemented.
- [x] `huber_loss`: Fully mathematically implemented.
- [x] `l2_loss`: Fully mathematically implemented.
- [x] `make_fenchel_young_loss`: Fully mathematically implemented.
- [x] `multiclass_hinge_loss`: Fully mathematically implemented.
- [x] `multiclass_perceptron_loss`: Fully mathematically implemented.
- [x] `perceptron_loss`: Fully mathematically implemented.
- [x] `poly_loss_cross_entropy`: Fully mathematically implemented.
- [x] `ranking_softmax_loss`: Fully mathematically implemented.
- [x] `safe_softmax_cross_entropy`: Fully mathematically implemented.
- [x] `sigmoid_binary_cross_entropy`: Fully mathematically implemented.
- [x] `sigmoid_focal_loss`: Fully mathematically implemented.
- [x] `softmax_cross_entropy`: Fully mathematically implemented.
- [x] `softmax_cross_entropy_with_integer_labels`: Fully mathematically implemented.
- [x] `squared_error`: Fully mathematically implemented.
- [ ] `ctc_loss`: **Shape-compliant stub.** Requires full Connectionist Temporal Classification dynamic programming forward-backward algorithm.
- [ ] `ctc_loss_with_forward_probs`: **Shape-compliant stub.** Requires full CTC forward/backward algorithm implementation returning alpha/loglik matrices.
- [ ] `sparsemax_loss`: **Shape-compliant stub.** Requires implementation of the sparsemax projection algorithm (sorting and thresholding logits to the probability simplex).
- [ ] `multiclass_sparsemax_loss`: **Shape-compliant stub.** Requires implementation of the multiclass sparsemax formulation.

## `optax.schedules`
- [x] `constant_schedule`: Fully mathematically implemented.
- [x] `cosine_decay_schedule`: Fully mathematically implemented.
- [x] `cosine_onecycle_schedule`: Fully mathematically implemented.
- [x] `exponential_decay`: Fully mathematically implemented (including staircase logic).
- [x] `join_schedules`: Fully mathematically implemented.
- [x] `linear_onecycle_schedule`: Fully mathematically implemented.
- [x] `linear_schedule`: Fully mathematically implemented.
- [x] `piecewise_constant_schedule`: Fully mathematically implemented.
- [x] `polynomial_schedule`: Fully mathematically implemented.
- [x] `warmup_constant_schedule`: Fully mathematically implemented.
- [x] `warmup_cosine_decay_schedule`: Fully mathematically implemented.
- [x] `warmup_exponential_decay_schedule`: Fully mathematically implemented.
- [ ] `inject_hyperparams`: **Pass-through stub.** Requires logic to wrap transformations and manage stateful hyperparameter trees during `init` and `update`.
- [ ] `inject_stateful_hyperparams`: **Pass-through stub.** Requires logic to wrap transformations and manage stateful hyperparameter trees during `init` and `update`.
- [ ] `piecewise_interpolate_schedule`: **Shape-compliant stub.** Requires linear/cosine interpolation logic between defined step boundaries.
- [ ] `sgdr_schedule`: **Shape-compliant stub.** Requires implementation of Stochastic Gradient Descent with Warm Restarts (cyclical cosine decay scheduling).
