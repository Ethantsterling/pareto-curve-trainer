"""Tests for component_scalar.py."""

# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os

os.environ["JAX_PLATFORMS"] = "cpu"
import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position

from pareto_curve_trainer.component_scalar import Bool, Categorical, Scalar

# Ensure jittability in these tests.
Scalar = jax.jit(Scalar)

test_scalar = Scalar(s=3.0, smin=-1.0, smax=4.0)
test_scalar_2 = Scalar(s=2.0, smin=-2.0, smax=5.0)
test_scalar_3 = Scalar(
    s=jnp.array([2, 5.0]), smin=jnp.array([-2.0, -3.0]), smax=jnp.array([5.0, 6])
)


def test_num_rows():
    assert test_scalar.num_rows() == 1
    assert test_scalar_2.num_rows() == 1
    assert test_scalar_3.num_rows() == 2


def test_value_times_array():
    assert (test_scalar.value() * jnp.array([1.0, 2, 3]) == jnp.array([3.0, 6, 9])).all()
    assert (
        test_scalar_3.value() * jnp.array([1.0, 2, 3]) == jnp.array([[2.0, 4, 6], [5.0, 10, 15]])
    ).all()


def test_scalar_merge():
    assert test_scalar.merge(test_scalar) == Scalar(
        s=jnp.array([3.0, 3.0]),
        smin=jnp.array([-1.0, -1.0]),
        smax=jnp.array([4.0, 4.0]),
    )
    assert test_scalar.merge(test_scalar_2) == Scalar(
        s=jnp.array([3.0, 2.0]),
        smin=jnp.array([-1.0, -2.0]),
        smax=jnp.array([4.0, 5.0]),
    )
    assert test_scalar.merge(test_scalar_3) == Scalar(
        s=jnp.array([3.0, 2.0, 5.0]),
        smin=jnp.array([-1.0, -2.0, -3.0]),
        smax=jnp.array([4.0, 5.0, 6.0]),
    )
    assert test_scalar_2.merge(test_scalar_3) == Scalar(
        s=jnp.array([2.0, 2.0, 5.0]),
        smin=jnp.array([-2.0, -2.0, -3.0]),
        smax=jnp.array([5.0, 5.0, 6.0]),
    )


def test_scalar_perturbation():
    for i in range(10):
        scale_key, perturb_key = jax.random.split(jax.random.key(i))
        scale = jax.random.uniform(key=scale_key)
        perturbed_scalar = test_scalar_3.perturbed(scale=scale, key=perturb_key)

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


def test_scalar_randomization():
    for i in range(10):
        randomized_scalar = test_scalar_3.randomized(key=jax.random.key(i))

        # Randomization preserves size and smin/smax.
        assert randomized_scalar.num_rows() == test_scalar_3.num_rows()
        assert jnp.all(randomized_scalar.smin == test_scalar_3.smin)
        assert jnp.all(randomized_scalar.smax == test_scalar_3.smax)

        # Randomization changes value within limits.
        assert jnp.all(randomized_scalar.s <= randomized_scalar.smax)
        assert jnp.all(randomized_scalar.s >= randomized_scalar.smin)


def test_bool_merge():
    test_bool_1 = Bool(jnp.array([True, False]))
    test_bool_2 = Bool(jnp.array([False, False, True]))
    assert test_bool_1.merge(test_bool_2) == Bool(jnp.array([True, False, False, False, True]))
    assert test_bool_2.merge(test_bool_1) == Bool(jnp.array([False, False, True, True, False]))
    test_bool_single = Bool(jnp.array(True))
    assert test_bool_1.merge(test_bool_single) == Bool(jnp.array([True, False, True]))
    assert test_bool_2.merge(test_bool_single) == Bool(jnp.array([False, False, True, True]))
    assert test_bool_single.merge(test_bool_1) == Bool(jnp.array([True, True, False]))
    assert test_bool_single.merge(test_bool_2) == Bool(jnp.array([True, False, False, True]))


def test_bool_perturbation():
    test_bool = Bool(jnp.array([True, False] * 500))
    perturbed_bool = test_bool.perturbed(scale=0.2, key=jax.random.key(345))
    assert test_bool.num_rows() == 1000 == perturbed_bool.num_rows()
    assert test_bool != perturbed_bool
    # ~half should be True, ~half should be False.
    assert jnp.isclose(jnp.sum(perturbed_bool.value()), 500, rtol=0.1)
    # ~20% of values should get flipped since scale is 0.2.
    flipped = test_bool.value() != perturbed_bool.value()
    assert jnp.isclose(jnp.sum(flipped), 200, atol=20, rtol=0)


def test_bool_randomization():
    test_bool = Bool(jnp.array([True, False] * 500))
    randomized_bool = test_bool.randomized(key=jax.random.key(345))
    assert test_bool.num_rows() == 1000 == randomized_bool.num_rows()
    assert test_bool != randomized_bool
    # ~half should be True, ~half should be False.
    assert jnp.isclose(jnp.sum(randomized_bool.value()), 500, atol=20, rtol=0)
    # Randomization should result in ~50% matches.
    flipped = test_bool.value() != randomized_bool.value()
    assert jnp.isclose(jnp.sum(flipped), 500, atol=20, rtol=0)


def test_categorical_merge():
    test_cat_1 = Categorical(jnp.array([1, 0]), num_categories=3)
    test_cat_2 = Categorical(jnp.array([0, 0, 2]), num_categories=3)
    assert test_cat_1.merge(test_cat_2) == Categorical(jnp.array([1, 0, 0, 0, 2]), 3)
    assert test_cat_2.merge(test_cat_1) == Categorical(jnp.array([0, 0, 2, 1, 0]), 3)
    test_cat_single = Categorical(jnp.array(2), num_categories=3)
    assert test_cat_1.merge(test_cat_single) == Categorical(jnp.array([1, 0, 2]), 3)
    assert test_cat_2.merge(test_cat_single) == Categorical(jnp.array([0, 0, 2, 2]), 3)
    assert test_cat_single.merge(test_cat_1) == Categorical(jnp.array([2, 1, 0]), 3)
    assert test_cat_single.merge(test_cat_2) == Categorical(jnp.array([2, 0, 0, 2]), 3)


def test_categorical_perturbation():
    test_categorical = Categorical(jnp.array([0, 1, 2, 3, 4] * 200), num_categories=5)
    perturbed_categorical = test_categorical.perturbed(scale=0.2, key=jax.random.key(345))

    assert test_categorical.num_rows() == 1000 == perturbed_categorical.num_rows()
    assert test_categorical != perturbed_categorical

    # Should be roughly evenly distributed between categories.
    _, counts = jnp.unique_counts(perturbed_categorical.value())
    assert jnp.allclose(counts, 200, atol=20)

    # ~20% of values should get flipped since scale is 0.2.
    flipped = test_categorical.value() != perturbed_categorical.value()
    assert jnp.isclose(jnp.sum(flipped), 200, atol=20, rtol=0)


def test_categorical_randomization():
    test_categorical = Categorical(jnp.array([0, 1, 2, 3, 4] * 200), num_categories=5)
    randomized_categorical = test_categorical.randomized(key=jax.random.key(345))

    assert test_categorical.num_rows() == 1000 == randomized_categorical.num_rows()
    assert test_categorical != randomized_categorical

    # Should be roughly evenly distributed between categories.
    _, counts = jnp.unique_counts(randomized_categorical.value())
    assert jnp.allclose(counts, 200, atol=30)

    # ~80% of values should get flipped since there are 5 categories.
    flipped = test_categorical.value() != randomized_categorical.value()
    assert jnp.isclose(jnp.sum(flipped), 800, atol=20, rtol=0)
