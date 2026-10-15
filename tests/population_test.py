"""Tests for population.py."""
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from pareto_curve_trainer import population
from pareto_curve_trainer import example_fixtures

# Ensure jittability in these tests.
Population = jax.jit(population.Population)


def test_extend_array():
    assert jnp.all(population._extend_array(jnp.array([1, 2, 3]), 6) ==
                   jnp.array([1,2,3,1,2,3]))
    _test_array = jnp.array([(1,2), (3,4), (5,6)])
    assert jnp.all(population._extend_array(_test_array, 2) ==
                   jnp.array([(1,2), (3,4)]))
    assert jnp.all(population._extend_array(_test_array, 5) ==
                   jnp.array([(1,2), (3,4), (5,6), (1,2), (3,4)]))


test_population = Population(jnp.array([1]), example_fixtures.test_components)
filled_test_population, key = test_population.fill(fill_size=5, key=jax.random.key(5))


def test_population_equality():
    assert test_population == test_population  # pylint: disable=comparison-with-itself
    assert filled_test_population == filled_test_population  # pylint: disable=comparison-with-itself


def test_filled_size():
    assert filled_test_population.num_trials() == 5
    assert jnp.all(filled_test_population.trial_ids == jnp.array([1,2,3,4,5]))
    assert filled_test_population.max_trial_id() == 5


def test_filter_population():
    filter_array = jnp.array([False, True, False, True, False])
    filtered_population = filled_test_population.filter(filter_array)
    assert filtered_population.num_trials() == 2

    # Filter down to the first trial, which was test_population.
    filter_array = jnp.array([True, False, False, False, False])
    filtered_population = filled_test_population.filter(filter_array)
    assert filtered_population.num_trials() == 1
    assert filtered_population == test_population


base_population = Population(jnp.array([0]), example_fixtures.test_components)
fp = base_population.stack(filled_test_population)


def test_stack():
    assert fp.num_trials() == 1 + filled_test_population.num_trials()
    assert base_population == fp.filter(jnp.array([0]))


def test_perturbed():
    random_key = jax.random.key(6)
    fp_perturbed, _ = fp.perturbed(scale=0.1, key=random_key)
    assert jnp.all(fp_perturbed.trial_ids != fp.trial_ids)
    for component_key, component in fp.components.items():
        perturbed_component = fp_perturbed.components[component_key]
        assert component.constraints_equal(perturbed_component)
        assert not component == perturbed_component

    fp_perturbed_same_ids, _ = fp.perturbed(scale=0.1, increment_index=False, key=random_key)
    assert jnp.all(fp_perturbed_same_ids.trial_ids == fp.trial_ids)
    assert fp_perturbed.components == fp_perturbed_same_ids.components
