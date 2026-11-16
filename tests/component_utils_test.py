"""Tests for component_utils.py."""

# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os

os.environ["JAX_PLATFORMS"] = "cpu"
import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position

import pareto_curve_trainer.component_utils as cu
from pareto_curve_trainer.transforms import Identity, Log, Symlog1p


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
    intended_shifts = jax.random.uniform(
        key=test_key, shape=jnp.shape(test_array), minval=-0.5, maxval=0.5
    )
    intended_array = cu.constrain_by_bouncing(test_array + intended_shifts, xmin=1.0, xmax=6.0)
    perturbed_array = cu.perturb_array(
        test_array, min_val=1.0, max_val=6.0, scale=0.1, key=test_key
    )
    assert (intended_array == perturbed_array).all()


def test_perturb_array_fx():
    """Verify that arrays are perturbed in the transformed space."""
    test_key = jax.random.key(5)
    min_val, max_val = jnp.exp(0), jnp.exp(10)
    test_array = jnp.exp(jnp.arange(1, 10))  # e^1, e^2, ..., e^9.
    scale = 0.1  # Our range is 0-10 in log space, so our shifts should be [-1, 1] in log space.
    expected_min = jnp.exp(jnp.arange(0, 9))
    expected_max = jnp.exp(jnp.arange(2, 11))
    log_perturbed_array = cu.perturb_array(
        test_array, min_val=min_val, max_val=max_val, scale=scale, key=test_key, fx=Log
    )
    assert jnp.all(expected_min <= log_perturbed_array)
    assert jnp.all(expected_max >= log_perturbed_array)

    # Verify that these guarantees do not hold if we perturb in the non-fx space.
    # There's a small chance they'd hold by accident, but realistically the 0.1 scale shift on an
    # e^10 range will almost assuredly exceed one of the lower restrictions.
    perturbed_array = cu.perturb_array(
        test_array, min_val=min_val, max_val=max_val, scale=scale, key=test_key
    )
    assert not jnp.all(expected_min <= perturbed_array)
    assert not jnp.all(expected_max >= perturbed_array)


def test_output_shape_matches():
    # Verify that output shape depends only on the input array's shape.
    test_key = jax.random.key(5)
    for a_shape in [(), (3,), (3, 5)]:
        _array = jax.random.uniform(key=test_key, shape=a_shape)
        # If min&max are 0D:
        _array_perturbed = cu.perturb_array(
            _array, min_val=0.0, max_val=1.0, scale=0.2, key=test_key
        )
        assert jnp.shape(_array) == jnp.shape(_array_perturbed)

        if len(a_shape) != 0:
            # We need to verify that using vectors for min&max still works.
            _array_perturbed = cu.perturb_array(
                _array,
                min_val=jnp.array([0.0, 0.0, 0.0]),
                max_val=jnp.array([1.0, 1.0, 1.0]),
                scale=0.2,
                key=test_key,
            )
            _array_randomized = cu.randomize_array(
                _array,
                min_val=jnp.array([0.0, 0.0, 0.0]),
                max_val=jnp.array([1.0, 1.0, 1.0]),
                key=test_key,
            )
            assert jnp.shape(_array) == jnp.shape(_array_perturbed)
            assert jnp.shape(_array) == jnp.shape(_array_randomized)


def test_perturbed_obeys_min_and_max():
    array_key, perturb_key = jax.random.split(jax.random.key(5))
    MIN, MAX = 1.0, 4.0
    test_array = jax.random.uniform(minval=MIN, maxval=MAX, shape=500, key=array_key)
    for fx in [None, Identity, Log, Symlog1p]:
        perturbed_array = cu.perturb_array(
            test_array, scale=0.5, min_val=MIN, max_val=MAX, key=perturb_key, fx=fx
        )
        assert (perturbed_array != test_array).all()  # Technically it could, but it won't.
        assert (perturbed_array <= 4.0).all()
        assert (perturbed_array >= 1.0).all()


def test_randomize_obeys_min_and_max():
    array_key, randomize_key = jax.random.split(jax.random.key(5))
    MIN, MAX = 1.0, 4.0
    test_array = jax.random.uniform(minval=MIN, maxval=MAX, shape=500, key=array_key)
    randomized_array = cu.randomize_array(test_array, min_val=MIN, max_val=MAX, key=randomize_key)
    assert (randomized_array != test_array).all()  # Technically it could, but it won't.
    assert (randomized_array <= MAX).all()
    assert (randomized_array >= MIN).all()


def test_randomize_bool_array():
    array_key, randomize_key = jax.random.split(jax.random.key(5))
    test_array = jax.random.bernoulli(key=array_key, shape=1000)
    randomized_array = cu.randomize_bool_array(test_array, key=randomize_key)
    assert not jnp.array_equal(test_array, randomized_array)  # It's possible but implausible.
    # ~half should be True, ~half should be False.
    assert jnp.isclose(jnp.sum(randomized_array), 500, atol=20, rtol=0)
    # ~half should match the test_array, ~half should differ.
    matches = test_array == randomized_array
    assert jnp.isclose(jnp.sum(matches), 500, atol=20, rtol=0)


def test_perturb_bool_array():
    test_array = jax.random.bernoulli(key=jax.random.key(54321), shape=1000)
    for key, scale in enumerate([0.01, 0.05, 0.1, 0.2, 0.5]):
        perturbed_array = cu.perturb_bool_array(test_array, scale=scale, key=jax.random.key(key))
        assert not jnp.array_equal(test_array, perturbed_array)  # It's possible but implausible.
        num_diffs = jnp.sum(test_array != perturbed_array)
        expected_num_diffs = 1000 * scale
        print(scale, num_diffs, expected_num_diffs)
        assert jnp.isclose(num_diffs, expected_num_diffs, atol=10, rtol=0.1)


def test_randomize_categorical_array():
    num_categories = 5
    array_key, randomize_key = jax.random.split(jax.random.key(5))
    test_array = jax.random.randint(minval=0, maxval=num_categories, key=array_key, shape=1000)
    randomized_array = cu.randomize_categorical_array(
        test_array, num_categories=num_categories, key=randomize_key
    )
    assert not jnp.array_equal(test_array, randomized_array)  # It's possible but implausible.
    # Should be split ~evenly between the categories.
    _, counts = jnp.unique_counts(randomized_array)
    expected_counts = jnp.array([200] * 5)
    assert jnp.allclose(counts, expected_counts, atol=10, rtol=0.1)

    # Randomly, they should match 1 / num_category of the time.
    matches = test_array == randomized_array
    assert jnp.isclose(jnp.sum(matches), 200, atol=10, rtol=0)


def test_perturb_categorical_array():
    num_categories = 5
    test_array = jax.random.randint(
        minval=0, maxval=num_categories, key=jax.random.key(54321), shape=1000
    )
    for key, scale in enumerate([0.01, 0.05, 0.1, 0.2, 0.5]):
        perturbed_array = cu.perturb_categorical_array(
            test_array, scale=scale, key=jax.random.key(key), num_categories=num_categories
        )
        assert not jnp.array_equal(test_array, perturbed_array)  # It's possible but implausible.
        num_diffs = jnp.sum(test_array != perturbed_array)
        expected_num_diffs = 1000 * scale
        print(scale, num_diffs, expected_num_diffs)
        assert jnp.isclose(num_diffs, expected_num_diffs, atol=10, rtol=0.2)

        # The original array was ~evenly distributed between categories, so the perturbed version
        # should be too.
        _, counts = jnp.unique_counts(perturbed_array)
        expected_counts = jnp.array([200] * 5)
        assert jnp.allclose(counts, expected_counts, atol=10, rtol=0.1)
