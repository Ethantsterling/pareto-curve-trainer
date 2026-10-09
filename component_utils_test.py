import os
# For testing most tests, it's quicker and cleaner to lock JAX to CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

import component_utils as cu


def test_constrain_by_wrapping():
    xs = jnp.linspace(0, 6, num=13)
    assert (
        cu.constrain_by_wrapping(xs, xmin=1, xmax=3)
        == jnp.array([2.0, 2.5, 1.0, 1.5, 2, 2.5, 1, 1.5, 2.0, 2.5, 1, 1.5, 2.0])
    ).all()


def test_constrain_by_bouncing():
    xs = jnp.linspace(0, 6, num=13)
    assert (
        cu.constrain_by_bouncing(xs, xmin=1, xmax=3)
        == jnp.array([2.0, 1.5, 1.0, 1.5, 2, 2.5, 3, 2.5, 2.0, 1.5, 1, 1.5, 2.0])
    ).all()


def test_perturb_array():
    # Max from 1 to 6, but the highest value in the array will be 5.
    test_key = jax.random.key(5)
    test_array = jnp.array([[1, 2.0, 3], [4, 5, 5]])
    intended_key, intended_subkey = jax.random.split(test_key)
    intended_shifts = jax.random.uniform(
        key=intended_key, shape=jnp.shape(test_array), minval=-0.5, maxval=0.5
    )
    intended_array = cu.constrain_by_bouncing(test_array + intended_shifts, xmin=1.0, xmax=6.0)
    perturbed_array, perturbed_subkey = cu.perturb_array(
        test_array, min=1.0, max=6.0, scale=0.1, key=test_key
    )
    assert intended_subkey == perturbed_subkey
    assert (intended_array == perturbed_array).all()


def test_output_shape_matches():
# Verify that output shape depends only on the input array's shape.
    test_key = jax.random.key(5)
    intended_key, _ = jax.random.split(test_key)
    for a_shape in [(), (3,), (3, 5)]:
        _array = jax.random.uniform(key=intended_key, shape=a_shape)
        # If min&max are 0D:
        _array_perturbed = cu.perturb_array(_array, min=0.0, max=1.0, scale=0.2, key=test_key)[
            0
        ]
        assert jnp.shape(_array) == jnp.shape(_array_perturbed)

        if len(a_shape) != 0:
            # We need to verify that using vectors for min&max still works.
            _array_perturbed = cu.perturb_array(
                _array,
                min=jnp.array([0.0, 0.0, 0.0]),
                max=jnp.array([1.0, 1.0, 1.0]),
                scale=0.2,
                key=test_key,
            )[0]
            _array_randomized = cu.randomize_array(
                _array,
                min=jnp.array([0.0, 0.0, 0.0]),
                max=jnp.array([1.0, 1.0, 1.0]),
                key=test_key,
            )[0]
            assert jnp.shape(_array) == jnp.shape(_array_perturbed)
            assert jnp.shape(_array) == jnp.shape(_array_randomized)


def test_perturbed_obeys_min_and_max():
    test_key = jax.random.key(5)
    array_key, perturb_key = jax.random.split(jax.random.key(5))
    MIN, MAX = 1.0, 4.0
    test_array = jax.random.uniform(minval=MIN, maxval=MAX, shape=500, key=array_key)
    perturbed_array, _ = cu.perturb_array(test_array, scale=0.5, min=MIN, max=MAX, key=perturb_key)
    assert (perturbed_array != test_array).all()  # Technically it could, but the chance is minute.  
    assert (perturbed_array <= 4.0).all()
    assert (perturbed_array >= 1.0).all()


def test_randomize_obeys_min_and_max():
    test_key = jax.random.key(5)
    array_key, randomize_key = jax.random.split(jax.random.key(5))
    MIN, MAX = 1.0, 4.0
    test_array = jax.random.uniform(minval=MIN, maxval=MAX, shape=500, key=array_key)
    randomized_array, _ = cu.randomize_array(test_array, min=MIN, max=MAX, key=randomize_key)
    assert (randomized_array != test_array).all()  # Technically it could, but the chance is minute.
    assert (randomized_array <= MAX).all()
    assert (randomized_array >= MIN).all()