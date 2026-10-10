"""Tests for component_scalar.py."""
# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
from component_scalar import Scalar

# Ensure jittability in these tests.
Scalar = jax.jit(Scalar)


test_scalar = Scalar(s=3., smin=-1., smax=4.)
test_scalar_2 = Scalar(s=2., smin=-2., smax=5.)
test_scalar_3 = Scalar(s=jnp.array([2, 5.]), smin=jnp.array([-2., -3.]), smax=jnp.array([5., 6]))


def test_num_rows():
    assert test_scalar.num_rows() == 1
    assert test_scalar_2.num_rows() == 1
    assert test_scalar_3.num_rows() == 2


def test_value_times_array():
    assert (test_scalar.value() * jnp.array([1., 2, 3]) == jnp.array([3., 6, 9])).all()
    assert (test_scalar_3.value() * jnp.array([1., 2, 3]) == jnp.array([[2., 4, 6], [5., 10, 15]])).all()


def test_merge():
    assert (test_scalar.merge(test_scalar) ==
            Scalar(s=jnp.array([3., 3.]), smin=jnp.array([-1., -1.]), smax=jnp.array([4., 4.])))
    assert (test_scalar.merge(test_scalar_2) == 
            Scalar(s=jnp.array([3., 2.]), smin=jnp.array([-1., -2.]), smax=jnp.array([4., 5.])))
    assert (test_scalar.merge(test_scalar_3) == 
            Scalar(s=jnp.array([3., 2., 5.]), smin=jnp.array([-1., -2., -3.]), smax=jnp.array([4., 5., 6.])))
    assert (test_scalar_2.merge(test_scalar_3) == 
            Scalar(s=jnp.array([2., 2., 5.]), smin=jnp.array([-2., -2., -3.]), smax=jnp.array([5., 5., 6.])))


def test_perturbation():
    for i in range(10):
        scale_key, perturb_key = jax.random.split(jax.random.key(i))
        scale = jax.random.uniform(key=scale_key)
        perturbed_scalar, _ = test_scalar_3.perturbed(scale=scale, key=perturb_key)
        
        # Perturbing preserves size and smin/smax.
        assert perturbed_scalar.num_rows() == test_scalar_3.num_rows()
        assert jnp.all(perturbed_scalar.smin == test_scalar_3.smin)
        assert jnp.all(perturbed_scalar.smax == test_scalar_3.smax)
        
        # Perturbing changes value within limits.
        assert jnp.all(perturbed_scalar.s <= perturbed_scalar.smax)
        assert jnp.all(perturbed_scalar.s >= perturbed_scalar.smin)
        
        max_shift = test_scalar_3.smax - test_scalar_3.smin
        assert jnp.all(perturbed_scalar.s <= test_scalar_3.s + max_shift)
        assert jnp.all(perturbed_scalar.s >= test_scalar_3.s - max_shift)


def test_randomization():
    for i in range(10):
        randomized_scalar, random_key = test_scalar_3.randomized(key=jax.random.key(i))
        
        # Randomization preserves size and smin/smax.
        assert randomized_scalar.num_rows() == test_scalar_3.num_rows()
        assert jnp.all(randomized_scalar.smin == test_scalar_3.smin)
        assert jnp.all(randomized_scalar.smax == test_scalar_3.smax)
        
        # Randomization changes value within limits.
        assert jnp.all(randomized_scalar.s <= randomized_scalar.smax)
        assert jnp.all(randomized_scalar.s >= randomized_scalar.smin)
