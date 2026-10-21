"""Tests for pareto.py."""
# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import pytest
from pareto_curve_trainer import pareto

test_scores_10k = jax.random.uniform(key=jax.random.key(12), shape=(10_000, 3))


def test_paretos_match():
    assert jnp.all(pareto.pareto(test_scores_10k) == pareto.pareto_dense(test_scores_10k))
    assert jnp.all(pareto.prefix_pareto(test_scores_10k) ==
                   pareto.prefix_pareto_dense(test_scores_10k))


def test_pareto_when_all_points_are_pareto():
    all_pareto_scores = jnp.array([(1, 5), (3, 3), (4, 2), (5, 1)], dtype=float)
    assert jnp.all(pareto.pareto(all_pareto_scores) == jnp.array([True, True, True, True]))
    assert jnp.all(pareto.pareto_dense(all_pareto_scores) == jnp.array([True, True, True, True]))
    assert jnp.all(pareto.prefix_pareto(all_pareto_scores) == jnp.array([True, True, True, True]))
    assert jnp.all(pareto.prefix_pareto_dense(all_pareto_scores) ==
                   jnp.array([True, True, True, True]))
    # None of the points are dominated.
    assert jnp.all(pareto.first_dominating(all_pareto_scores) == jnp.array([4, 4, 4, 4]))


def test_pareto():
    scores = jnp.array([(3, 3), (1, 2), (4, 0), (5, 1)], dtype=float)
    # (1,2) and (4,0) are dominated.
    assert jnp.all(pareto.pareto(scores) == jnp.array([True, False, False, True]))
    assert jnp.all(pareto.pareto_dense(scores) == jnp.array([True, False, False, True]))
    # (1,2) is prefix-dominated. (4,0) is dominated, but not prefix-dominated.
    assert jnp.all(pareto.prefix_pareto(scores) == jnp.array([True, False, True, True]))
    assert jnp.all(pareto.prefix_pareto_dense(scores) == jnp.array([True, False, True, True]))
    assert jnp.all(pareto.first_dominating(scores) == jnp.array([4, 0, 3, 4]))


def test_hypervolume_1d():
    origin = jnp.array([0.])
    scores = jnp.array([[1.], [2.], [4.]])
    expected_hypervolume = jnp.array([1., 2., 4.])
    predicted_hypervolume = pareto.prefix_dominated_hypervolume(scores, origin, num_vectors_pow=5)
    # The 1D version is easy and our function should find it without trouble.
    assert jnp.allclose(expected_hypervolume, predicted_hypervolume)


def test_hypervolume_2d():
    """A simple 2D setup where we can math out the right hypervolume to expect."""
    scores = jnp.array([[1., 1.], [2., 4.], [3., 2.], [5., 5.]])
    # The prefixes scores[:1], scores[:2], and scores[:4] are easy since one point dominates.
    # The only tricky expected hypervolume is for the prefix scores[:3], because scores[1] and
    # scores[2] overlap without dominating each other.
    # They overlap in the square with opposite corners at (0, 0) and (2, 2), so we can get the
    # total hypervolume via inclusion exclusion: 2*4 + 3*2 - 2*2 = 10.
    expected_hypervolume = jnp.array([1., 8., 10., 25])
    origin = jnp.array([0, 0.])
    predicted_hypervolume = pareto.prefix_dominated_hypervolume(scores, origin, num_vectors_pow=15)
    # pareto.prefix_dominated_hypervolume is approximate, so we need to relax the isclose bounds.
    assert jnp.allclose(expected_hypervolume, predicted_hypervolume, rtol=1e-2)


def test_hypervolume_origin():
    """Translating the origin should be equivalent to an opposite translation on the scores."""
    origin = jnp.array([-0.3, 0.2, 0.1])
    shift = jnp.array([.15, -.1, 0.3])

    # Shifting scores and origin by the same value shouldn't change anything.
    origin_hypervolume = pareto.prefix_dominated_hypervolume(
        test_scores_10k, origin, num_vectors_pow=5)
    shifted_hypervolume = pareto.prefix_dominated_hypervolume(
        test_scores_10k - shift, origin - shift, num_vectors_pow=5)

    assert jnp.allclose(origin_hypervolume, shifted_hypervolume)


def test_hypervolume_scaling():
    """Multiplying a metric by k should multiply the hypervolume by k."""
    origin = jnp.array([0, 0., 0.])
    skew_multipliers = jnp.array([2., 5, 3.])
    skewed_test_scores = test_scores_10k * skew_multipliers
    hypervolume = pareto.prefix_dominated_hypervolume(
        test_scores_10k, origin, num_vectors_pow=5)
    skewed_hypervolume = pareto.prefix_dominated_hypervolume(
        skewed_test_scores, origin, num_vectors_pow=5)

    assert jnp.allclose(skewed_hypervolume, hypervolume * jnp.prod(skew_multipliers))


def test_hypervolume_too_many_vectors():
    """Validate num_vectors_pow since it's a bit unintuitive to pass an exponent."""
    origin = jnp.array([0, 0., 0.])
    with pytest.raises(ValueError):
        pareto.prefix_dominated_hypervolume(test_scores_10k, origin, num_vectors_pow=1000)
    with pytest.raises(ValueError):
        pareto.prefix_dominated_hypervolume(test_scores_10k, origin, num_vectors_pow=-2)
