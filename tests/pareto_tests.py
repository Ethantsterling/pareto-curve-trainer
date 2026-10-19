"""Tests for pareto.py."""
# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from pareto_curve_trainer import pareto


test_scores_10k = jax.random.uniform(key=jax.random.key(12), shape=(10_000, 3))


def test_paretos_match():
    assert jnp.all(pareto.pareto(test_scores_10k) == pareto.is_pareto_map(test_scores_10k))
    assert jnp.all(pareto.prefix_pareto(test_scores_10k) ==
                   pareto.prefix_pareto_dense.is_pareto_map(test_scores_10k))


def test_pareto_when_all_points_are_pareto():
    all_pareto_scores = jnp.array([(1, 5), (3, 3), (4, 2), (5,1)], dtype=float)
    assert jnp.all(pareto.pareto(all_pareto_scores) == jnp.array([True, True, True, True]))
    assert jnp.all(pareto.pareto_dense(all_pareto_scores) == jnp.array([True, True, True, True]))
    assert jnp.all(pareto.prefix_pareto(all_pareto_scores) == jnp.array([True, True, True, True]))
    assert jnp.all(pareto.prefix_pareto_dense(all_pareto_scores) == jnp.array([True, True, True, True]))
    # None of the points are dominated.
    assert jnp.all(pareto.first_dominating(all_pareto_scores) == jnp.array([4, 4, 4, 4]))


def test_pareto():
    scores = jnp.array([(3, 3), (1, 2), (4, 0), (5,1)], dtype=float)
    # (1,2) and (4,0) are dominated.
    assert jnp.all(pareto.pareto(scores) == jnp.array([True, False, False, True]))
    assert jnp.all(pareto.pareto_dense(scores) == jnp.array([True, False, False, True]))
    # (1,2) is prefix-dominated. (4,0) is dominated, but not prefix-dominated.
    assert jnp.all(pareto.prefix_pareto(scores) == jnp.array([True, False, True, True]))
    assert jnp.all(pareto.prefix_pareto_dense(scores) == jnp.array([True, False, True, True]))
    assert jnp.all(pareto.first_dominating(scores) == jnp.array([4, 0, 3, 4]))
