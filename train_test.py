"""Unit tests for train.py."""
# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from train import population_train
from example_fixtures import (
    test_xs, test_population, example_metrics_fn, example_model_fn, NUM_METRICS)

POPULATION_SIZE = 5
NUM_STEPS = 100

# A tiny training run to ensure everything works.
learned_population, learned_metrics = population_train(
    xs=test_xs,
    population=test_population,
    metrics_fn=example_metrics_fn,
    model_fn=example_model_fn,
    population_size=POPULATION_SIZE,
    num_steps=NUM_STEPS,
    random_key=jax.random.key(5),
)

def test_population_size():
    assert learned_population.num_trials() == POPULATION_SIZE

def test_population_ids_with_limits():
    assert jnp.all(learned_population.trial_ids >= 0)
    assert jnp.all(learned_population.trial_ids <= NUM_STEPS)

def test_learned_metrics():
    assert learned_metrics.shape == (POPULATION_SIZE, NUM_METRICS)
    assert learned_population.num_trials() == POPULATION_SIZE

def test_learned_something():
    assert not learned_population == test_population
