"""Tests for population.py."""
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
# from typing import List

import population
from population import Population
import example_fixtures

def test_extend_array():
    assert jnp.all(population._extend_array(jnp.array([1, 2, 3]), 6) ==
                   jnp.array([1,2,3,1,2,3]))
    _test_array = jnp.array([(1,2), (3,4), (5,6)])
    assert jnp.all(population._extend_array(_test_array, 2) ==
                   jnp.array([(1,2), (3,4)]))
    assert jnp.all(population._extend_array(_test_array, 5) ==
                   jnp.array([(1,2), (3,4), (5,6), (1,2), (3,4)]))


# toy_key = jax.random.key(seed=5)
# size_reduction = 2
# for name, c in fp.components.items():
#   # print(c, 'becomes', _extend_component(c, 2).randomized(key=toy_key))
#   print(c)
#   extended_component = _extend_component(c, 2)
#   print('extended:', extended_component)
#   # This is breaking because of 1D-2D shenanigans.

#   randomized_component = extended_component.randomized(key=toy_key)
#   print('randomized:', randomized_component)
#   # print(_extend_component(c, 5))

# for name, c in fp.components.items():
#   print(c, 'becomes', _extend_component(c, 2))
# # print(fp)
# # print('')
# # print(fp.fill_population(fill_size=7, key=jax.random.key(6))[0])


test_population = Population.from_components(example_fixtures.test_components)
filled_test_population, key = test_population.fill_population(fill_size=5, key=jax.random.key(5))


def test_population_equality():
    assert test_population == test_population
    assert filled_test_population == filled_test_population


def test_filled_size():
    assert filled_test_population.num_trials() == 5
    assert jnp.all(filled_test_population.trial_ids == jnp.array([1,2,3,4,5]))
    assert filled_test_population.max_trial_id() == 5


def test_filter_population():
    filter_array = jnp.array([False, True, False, True, False])
    filtered_population = filled_test_population.filter_population(filter_array)
    assert filtered_population.num_trials() == 2


def test_filter_population_2():
    filter_array = jnp.array([True, False, False, False, False])
    filtered_population = filled_test_population.filter_population(filter_array)
    assert filtered_population.num_trials() == 1
    assert test_population.num_trials() == 1
    assert filtered_population.max_trial_id() == test_population.max_trial_id()
    assert jnp.all(filtered_population.trial_ids == test_population.trial_ids)
    assert filtered_population.components == test_population.components
    assert filtered_population == test_population


fp = Population(jnp.array([0]), example_fixtures.test_components).stack(filled_test_population)


def test_stack():
    assert fp.num_trials() == 6


def test_perturbed():
    print("\nPerturbed:")
    print(fp.perturbed(scale=0.1, key=jax.random.key(6))[0])

    print("\nPerturbed without changing IDs:")
    print(fp.perturbed(scale=0.1, increment_index=False, key=jax.random.key(6))[0])

    print("\nPerturbed with specifically set ids:")
    print(fp.perturbed(scale=0.1, first_id=1732, key=jax.random.key(6))[0])
