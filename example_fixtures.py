"""Tools and examples for use in testing."""
import jax
import jax.numpy as jnp

from component_scalar import Scalar
from component_catmap import Catmap, InternalCatmap
from component_pwlcurve import StructPwlcurve, Pwlcurve
import component_utils as cu
from component_utils import Mono

# @jax.jit
def mse_fn(pred: jnp.ndarray, label: jnp.ndarray):
    """Vectorized MSE function."""
    # Predictions should be (#trials, #data).
    # labels should be (#data,).
    # Broadcasting works without needing to reshape.
    if jnp.ndim(pred) == 1:
        # The metric is just one number.
        return -jnp.mean(jnp.square(pred - label))

    # Else, the metric is one number per trial.
    return -jnp.mean(jnp.square(pred - label), axis=-1)

_pred = jnp.array([(1., 2, 3), (4, 5, 6)])
_label = jnp.array([1.2, 3.2, 5.2])
print(mse_fn(_pred[0], _label))
print(mse_fn(_pred, _label))


def RandomUnitStructPWLCurve(num_points, seed):
    """Generates a piecewise-linear curve with random xs and ys in [0, 1]."""
    x_key, y_key = jax.random.split(jax.random.key(seed))
    curve_xs = jnp.sort(jax.random.uniform(shape=num_points, key=x_key))
    curve_ys = jax.random.uniform(shape=num_points, key=y_key)
    return StructPwlcurve(xs=curve_xs, ys=curve_ys)


def RandomUnitPWLCurve(num_points, key, xmin=0, xmax=1, ymin=0, ymax=1, mono=Mono.NON_MONO):
    """Generates a piecewise-linear curve with random xs and ys."""
    curve_component, _ = Pwlcurve(
        xs=jnp.array([jnp.linspace(xmin, xmax, num=num_points)]),
        ys=jnp.array([jnp.linspace(ymin, ymax, num=num_points)]),
        xmin=jnp.array([xmin], dtype=float),
        xmax=jnp.array([xmax], dtype=float),
        ymin=jnp.array([ymin], dtype=float),
        ymax=jnp.array([ymax], dtype=float),
        mono=jnp.array([mono]),
    ).randomized(key=key)
    return curve_component


def TraininablePWLCurveWithDefaults(xs, ys, xmin=0, xmax=1, ymin=0, ymax=1, mono=Mono.NON_MONO):
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


def WrappedScalar(s):
    """A convenient function to generate a Scalar component."""
    return Scalar(s=jnp.array(s), smin=jnp.zeros_like(s), smax=jnp.ones_like(s))


NUM_TRAINING_POINTS = 5_000

test_xs = {
    'curve_x':  jax.random.uniform(jax.random.key(11), shape=NUM_TRAINING_POINTS),
    'scalar_x1': jax.random.uniform(jax.random.key(12), shape=NUM_TRAINING_POINTS),
    'scalar_x2': jax.random.uniform(jax.random.key(13), shape=NUM_TRAINING_POINTS),
    'cat_x': jax.random.randint(
        jax.random.key(14), shape=NUM_TRAINING_POINTS, minval=0, maxval=3),
    # To motivate uniqueness, we'll throw a multiplier against the Catmap's output.
    'cat_multiplier_x': jax.random.uniform(
        jax.random.key(15), shape=NUM_TRAINING_POINTS, minval=1.0, maxval=2.0),
}

test_target_components = {
    'curve': TraininablePWLCurveWithDefaults(
        xs=[0.1, 0.5, 0.8, 1.0], ys=[0.4, 0.5, 0.9, 0.6]).as_struct_pwlcurves()[0],
    # RandomUnitPWLCurve(num_points=4, key=jax.random.key(1)),
    # 'curve': RandomUnitStructPWLCurve(num_points=4, seed=1),
    'scalar_1': WrappedScalar(s=0.3),
    'scalar_2': WrappedScalar(s=0.8),
    'catmap': InternalCatmap(jnp.array([0.2, 0.9, 0.5])), #{0: 0.2, 1: 0.9, 2: 0.5},
}

# A second target function for multiobjective work.
# Some pieces are identical to test_target_components,
# but some differ.
test_target_components_2 = {
    'curve': TraininablePWLCurveWithDefaults(
        xs=[0.1, 0.5, 0.8, 1.0], ys=[0.7, 0.1, 0.9, 0.6]).as_struct_pwlcurves()[0],
    'scalar_1': WrappedScalar(s=0.5),
    'scalar_2': WrappedScalar(s=0.8),
    'catmap': InternalCatmap(jnp.array([0.2, 0.6, 0.7])),
}

# TODO: find an elegant way to let the trainable scalar behave more like a scalar.
# Maybe it could descend from float & NamedTuple both?
def test_model_fn(components, xs):
    """A simple example of a model_fn excercising each component."""
    curve = components['curve'].eval(xs['curve_x'])
    scalar_1 = components['scalar_1'].value() * xs['scalar_x1']
    scalar_2 = components['scalar_2'].value() * xs['scalar_x2']
    # To prevent creating parallel equivalent models where curve or cat_x absorb
    # a constant shift in y, we'll make sure the Catmap's output gets multiplied
    # by unique x-values.
    cat_x = components['catmap'].eval(xs['cat_x']) * xs['cat_multiplier_x']
    return scalar_1 + scalar_2 + cat_x + curve


test_labels = test_model_fn(test_target_components, test_xs)
test_labels_2 = test_model_fn(test_target_components_2, test_xs)
# Labels belong in the xs. The metrics are responsible for extracting them.
test_xs['label'] = test_model_fn(test_target_components, test_xs)
test_xs['label_2'] = test_model_fn(test_target_components_2, test_xs)

test_components = {
    'curve': RandomUnitPWLCurve(num_points=4, key=jax.random.key(2)),
    'scalar_1': Scalar(.7, smin=0.0, smax=1.0),
    'scalar_2': Scalar(.2, smin=0.0, smax=1.0),
    'catmap': Catmap(InternalCatmap(jnp.array([[.4, .4, .1]])), ymin=0.0, ymax=1.0),
}


def test_metric_fn(predictions, labels) -> jnp.ndarray:
    """A very simple metric function that just emits MSE against the labels."""
    return mse_fn(predictions, labels)


print("Metric eval:")
print(test_metric_fn(test_model_fn(test_target_components, test_xs), test_labels))
print(test_metric_fn(test_model_fn(test_components, test_xs), test_labels))
# test_labels should really be 1D, shouldn't it?


