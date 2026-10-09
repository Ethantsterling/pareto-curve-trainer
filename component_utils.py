from enum import IntEnum
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
import jax
import jax.numpy as jnp


class Mono(IntEnum):
    NON_MONO = 0
    UP = 1
    DOWN = 2


# Utilities for forcing dimensionality or stacking components.
def _1D(a: jax.typing.ArrayLike):  # pylint: disable=C0103:invalid-name
    return jnp.reshape(a, (-1,))


def _1D_merge(a1: jax.typing.ArrayLike, a2: jax.typing.ArrayLike):  # pylint: disable=C0103:invalid-name
    return jnp.concatenate([_1D(a1), _1D(a2)])


def _2D(a: jax.typing.ArrayLike):  # pylint: disable=C0103:invalid-name
    """Make sure the array is 2D."""
    if jnp.ndim(a) == 1:
        return jnp.reshape(a, (1, -1))
    return a


def _2D_merge(a1: jax.typing.ArrayLike, a2: jax.typing.ArrayLike):  # pylint: disable=C0103:invalid-name
    return jnp.concatenate([_2D(a1), _2D(a2)])


# When training, we'll often mutate parameters by adding a random step to the
# current parameter value. Sometimes the resulting sum will exceed the parameter's
# constraints. In that case, we'll need to constrain the parameter back to the
# original constraints.

# The easiest way is to clip the parameters like jnp.clip(x, xmin, xmax) so that
# x < xmin --> xmin and x > xmax --> xmax. That approach greatly increases the
# chance of a parameter reaching its exact min or max value. If this is desirable,
# then just use jnp.clip.

# However, when x is an array, there's a significant chance that multiple values
# in x will be clipped to the same endpoint. That's inconvenient for components
# such as pwlcurves where duplicate parameters can cause problems. (A pwlcurve's
# x-values are supposed to be unique. Duplicate xs cause errors or wasted points
# or discontinuities, depending on the implementation.) If we want to keep params
# unique, then a different constraint enforcement is preferred.
def constrain_by_wrapping(
    x: jax.typing.ArrayLike, xmin: jax.typing.ArrayLike, xmax: jax.typing.ArrayLike
) -> jax.typing.ArrayLike:
    """Impose the xmin & xmax constraints by wrapping around when x exceeds the
    min or the max. For example, a value slightly over xmax becomes a value
    slightly over xmin to fit the constraints.

    This approach makes it difficult for model training to learn parameter values
    near xmin or xmax, because random updates are likely to flip the value to the
    opposite extreme.
    """
    return (x - xmin) % (xmax - xmin) + xmin


def constrain_by_bouncing(
    x: jax.typing.ArrayLike, xmin: jax.typing.ArrayLike, xmax: jax.typing.ArrayLike
) -> jax.typing.ArrayLike:
    """Impose the xmin & xmax constraints by rebounding off the xmin and xmax.
    For example, a value slightly over xmax becomes a value slightly under xmax
    to fit the constraints.
    """
    diff = xmax - xmin
    # constrained(x) is equivalent to constrained(x + 2 * diff) since two diffs
    # will bounce off both bounds and return to the starting position.
    x = (x - xmin) % (2 * diff) + xmin

    # This expression requires x to be within one diff of the bounds, which the
    # previous expression guarantees.
    return jnp.where(x < xmin, xmin - x, jnp.where(x > xmax, 2 * xmax - x, x))


def perturb_array(
    a: jax.typing.ArrayLike,
    min_val: jax.typing.ArrayLike,
    max_val: jax.typing.ArrayLike,
    scale: float,
    key: jax.Array,
):
    """Returns `a` randomly shifted by up to scale frac of max-min.

    `a` should be 0, 1, or 2D, and min&max should be 0 or 1D. If min&max are 1D,
    then `a` must be an array whose first dimension is the same size as min&max.
    """
    key, subkey = jax.random.split(key)
    dtype = jax.dtypes.result_type(a)

    # If a is (N, M) and min&max are (N), then default broadcasting will match
    # the wrong axis unless we expand min and max to (N, 1).
    # In all other cases, the default broadcasting is as intended.
    if jnp.ndim(a) == 2 and jnp.ndim(min_val) == 1:
        min_val = jnp.expand_dims(min_val, -1)
        max_val = jnp.expand_dims(max_val, -1)

    max_shift = scale * (max_val - min_val)
    shifts = jax.random.uniform(
        key, shape=jnp.shape(a), dtype=dtype, minval=-max_shift, maxval=max_shift
    )

    new_a = constrain_by_bouncing(a + shifts, min_val, max_val)
    return new_a, subkey


def randomize_array(
    a: jax.typing.ArrayLike,    # Array of Floats
    min_val: jax.typing.ArrayLike,
    max_val: jax.typing.ArrayLike,
    key: jax.Array,
):
    """Returns `a` randomized between max and min.

    `a` should be 0, 1, or 2D, and min&max should be 0 or 1D. If min&max are 1D,
    then `a` must be an array whose first dimension is the same size as min&max.
    """
    key, subkey = jax.random.split(key)
    # If a is (N, M) and min&max are (N), then default broadcasting will match
    # the wrong axis unless we expand min and max to (N, 1).
    # In all other cases, the default broadcasting is as intended.
    if jnp.ndim(a) == 2 and jnp.ndim(min_val) == 1:
        min_val = jnp.expand_dims(min_val, -1)
        max_val = jnp.expand_dims(max_val, -1)

    new_a = jax.random.uniform(key=key, shape=jnp.shape(a), minval=min_val, maxval=max_val)
    return new_a, subkey