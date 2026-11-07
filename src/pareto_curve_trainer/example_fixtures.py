"""Tools and examples for use in testing."""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from pareto_curve_trainer.component_scalar import Scalar
from pareto_curve_trainer.component_catmap import Catmap, InternalCatmap
from pareto_curve_trainer.component_pwlcurve import StructPwlcurve, Pwlcurve
from pareto_curve_trainer.component_utils import Mono
from pareto_curve_trainer import population, metrics


def random_unit_struct_pwlcurve(num_points, seed):
    """Generates a piecewise-linear curve with random xs and ys in [0, 1]."""
    x_key, y_key = jax.random.split(jax.random.key(seed))
    curve_xs = jnp.sort(jax.random.uniform(shape=num_points, key=x_key))
    curve_ys = jax.random.uniform(shape=num_points, key=y_key)
    return StructPwlcurve(xs=curve_xs, ys=curve_ys)


def random_unit_pwlcurve(num_points, key, xmin=0, xmax=1, ymin=0, ymax=1, mono=Mono.NON_MONO):
    """Generates a piecewise-linear curve with random xs and ys."""
    curve_component = Pwlcurve(
        xs=jnp.array([jnp.linspace(xmin, xmax, num=num_points)]),
        ys=jnp.array([jnp.linspace(ymin, ymax, num=num_points)]),
        xmin=jnp.array([xmin], dtype=float),
        xmax=jnp.array([xmax], dtype=float),
        ymin=jnp.array([ymin], dtype=float),
        ymax=jnp.array([ymax], dtype=float),
        mono=jnp.array([mono]),
    ).randomized(key=key)
    return curve_component


def trainable_pwlcurve_with_defaults(xs, ys, xmin=0, xmax=1, ymin=0, ymax=1, mono=Mono.NON_MONO):
    """Generates a vectorized component of piecewise-linear curves."""
    xs, ys = jnp.array(xs, dtype=float), jnp.array(ys, dtype=float)
    assert xs.shape == ys.shape
    if jnp.ndim(xs) == 1:
        xs = xs[None, :]
        ys = ys[None, :]
    size = len(xs)
    xmin = jnp.resize(xmin, size).astype(float)
    xmax = jnp.resize(xmax, size).astype(float)
    ymin = jnp.resize(ymin, size).astype(float)
    ymax = jnp.resize(ymax, size).astype(float)
    mono = jnp.resize(mono, size).astype(int)
    return Pwlcurve(xs=xs, ys=ys,
                    xmin=xmin, xmax=xmax,
                    ymin=ymin, ymax=ymax,
                    mono=mono)


def wrapped_scalar(s):
    """A convenient function to generate a Scalar component."""
    return Scalar(s=jnp.array(s), smin=jnp.zeros_like(s), smax=jnp.ones_like(s))


test_target_components = {
    'curve': trainable_pwlcurve_with_defaults(
        xs=[0.1, 0.5, 0.8, 1.0], ys=[0.4, 0.5, 0.9, 0.6]).as_struct_pwlcurves()[0],
    'scalar_1': wrapped_scalar(s=0.3),
    'scalar_2': wrapped_scalar(s=0.8),
    'catmap': InternalCatmap(jnp.array([0.2, 0.9, 0.5])),
}

# A second target function for multiobjective work.
# Some pieces are identical to test_target_components,
# but some differ.
test_target_components_2 = {
    'curve': trainable_pwlcurve_with_defaults(
        xs=[0.1, 0.5, 0.8, 1.0], ys=[0.7, 0.1, 0.9, 0.6]).as_struct_pwlcurves()[0],
    'scalar_1': wrapped_scalar(s=0.5),
    'scalar_2': wrapped_scalar(s=0.8),
    'catmap': InternalCatmap(jnp.array([0.2, 0.6, 0.7])),
}


def example_model_fn(components, xs):
    """A simple example of a model_fn excercising each component."""
    curve = components['curve'].eval(xs['curve_x'])
    scalar_1 = components['scalar_1'].value() * xs['scalar_x1']
    scalar_2 = components['scalar_2'].value() * xs['scalar_x2']
    # To prevent creating parallel equivalent models where curve or cat_x absorb
    # a constant shift in y, we'll make sure the Catmap's output gets multiplied
    # by unique x-values.
    cat_x = components['catmap'].eval(xs['cat_x']) * xs['cat_multiplier_x']
    return scalar_1 + scalar_2 + cat_x + curve


def generate_xs(num_training_points, random_seed=102394):
    """Generate xs suitable for the model in example_fixtures.py"""
    keys = jax.random.split(jax.random.key(random_seed), num=6)
    # For group tasks, let's make the groups 20 points long on average.
    num_groups = num_training_points // 20
    xs = {
        'curve_x':  jax.random.uniform(keys[0], shape=num_training_points),
        'scalar_x1': jax.random.uniform(keys[1], shape=num_training_points),
        'scalar_x2': jax.random.uniform(keys[2], shape=num_training_points),
        'cat_x': jax.random.randint(keys[3], shape=num_training_points, minval=0, maxval=3),
        'cat_multiplier_x': jax.random.uniform(keys[4], shape=num_training_points),
        'groups': jax.random.randint(
            keys[5], shape=num_training_points, minval=0, maxval=num_groups)
    }
    # Labels belong in the xs. The metrics are responsible for extracting them for use.
    xs['label'] = example_model_fn(test_target_components, xs)
    xs['label_2'] = example_model_fn(test_target_components_2, xs)
    return xs


NUM_TRAINING_POINTS = 5_000
test_xs = generate_xs(NUM_TRAINING_POINTS)

test_components = {
    'curve': random_unit_pwlcurve(num_points=4, key=jax.random.key(2)),
    'scalar_1': Scalar(jnp.array([.7]), smin=jnp.array([0.0]), smax=jnp.array([1.0])),
    'scalar_2': Scalar(jnp.array([.2]), smin=jnp.array([0.0]), smax=jnp.array([1.0])),
    'catmap': Catmap(InternalCatmap(jnp.array([[.4, .4, .1]])), ymin=0.0, ymax=1.0),
}


def example_metric_fn(labels, predictions) -> jnp.ndarray:
    """A very simple metric function that just emits MSE against the labels."""
    return metrics.mse_fn(labels, predictions)


# Eventually we'll want the metrics_fn to return a dictionary, I think?
# So that we can label the metrics for posterity.
def example_metrics_fn(xs, predictions):
    """A simple two-metric function for multiobjective training."""
    # predictions should be (population_size, data_size).
    # Metrics like mse_fn are built to handle that.
    mse_1 = metrics.mse_fn(xs['label'], predictions)
    mse_2 = metrics.mse_fn(xs['label_2'], predictions)
    return mse_1, mse_2


class ExampleRankedMetrics(NamedTuple):
    """A Pytree version of metrics_fn so that we can contain arrays."""
    ncdg_metric_1: metrics.AnchoredOrderMetric
    ncdg_metric_2: metrics.AnchoredOrderMetric

    def __call__(self, xs, predictions):
        groups = xs['groups']
        # This sort is the slowest part of training.
        order = metrics.group_rank(predictions, groups)
        ndcg_score_1 = self.ncdg_metric_1.eval(xs['label'][order])
        ndcg_score_2 = self.ncdg_metric_2.eval(xs['label_2'][order])
        return ndcg_score_1, ndcg_score_2


def make_example_ranked_metrics_fn(xs):
    """A factory function for an example with two nDCG ranking functions."""
    anchored_ndcg_1 = metrics.anchored_ndcg_metric(
        labels=xs['label'], groups=xs['groups'], depth=5)
    anchored_ndcg_2 = metrics.anchored_ndcg_metric(
        labels=xs['label_2'], groups=xs['groups'], depth=5)
    return ExampleRankedMetrics(anchored_ndcg_1, anchored_ndcg_2)


NUM_METRICS = 2
test_population = population.from_components(test_components)
