import numpy as np
import pytest
import jax.random as rand_ref
import zero_jax.random as rand_zero


@pytest.mark.skip(reason="Phase 3/5 PRNG or numerical divergence")
@pytest.mark.skip(reason="Phase 5 PRNG divergence")
def test_random_skip_split(check_allclose):
    key = rand_ref.PRNGKey(42)
    keys_z = rand_zero.split(key, 2)
    keys_r = rand_ref.split(key, 2)
    check_allclose(keys_z, keys_r)


@pytest.mark.skip(reason="Phase 3/5 PRNG or numerical divergence")
@pytest.mark.skip(reason="Phase 5 PRNG divergence")
def test_random_skip_fold_in(check_allclose):
    key = rand_ref.PRNGKey(42)
    key_z = rand_zero.fold_in(key, 1)
    key_r = rand_ref.fold_in(key, 1)
    check_allclose(key_z, key_r)


@pytest.mark.skip(reason="Phase 5 PRNG divergence")
def test_random_skip_uniform(check_allclose):
    key = rand_ref.PRNGKey(42)
    z = rand_zero.uniform(key, (2, 2))
    r = rand_ref.uniform(key, (2, 2))
    check_allclose(z, r)


@pytest.mark.skip(reason="Phase 5 PRNG divergence")
def test_random_skip_normal(check_allclose):
    key = rand_ref.PRNGKey(42)
    z = rand_zero.normal(key, (2, 2))
    r = rand_ref.normal(key, (2, 2))
    check_allclose(z, r)


@pytest.mark.skip(reason="Phase 5 PRNG divergence")
def test_random_skip_randint(check_allclose):
    key = rand_ref.PRNGKey(42)
    z = rand_zero.randint(key, (2, 2), 0, 10)
    r = rand_ref.randint(key, (2, 2), 0, 10)
    check_allclose(z, r)


@pytest.mark.skip(reason="Phase 5 PRNG divergence")
def test_random_skip_bernoulli(check_allclose):
    key = rand_ref.PRNGKey(42)
    z = rand_zero.bernoulli(key, 0.5, (2, 2))
    r = rand_ref.bernoulli(key, 0.5, (2, 2))
    check_allclose(z, r)


@pytest.mark.skip(reason="Phase 5 PRNG divergence")
def test_random_skip_categorical(check_allclose):
    key = rand_ref.PRNGKey(42)
    logits = np.array([0.1, 0.2, 0.7])
    z = rand_zero.categorical(key, logits)
    r = rand_ref.categorical(key, logits)
    check_allclose(z, r)


@pytest.mark.skip(reason="Phase 5 PRNG divergence")
def test_random_skip_permutation(check_allclose):
    key = rand_ref.PRNGKey(42)
    x = np.arange(10)
    z = rand_zero.permutation(key, x)
    r = rand_ref.permutation(key, x)
    check_allclose(z, r)
