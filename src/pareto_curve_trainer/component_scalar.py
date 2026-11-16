"""A trainable scalar
Training wrappers around 0D, 1D, and 2D functions that include training
constraints such as bounds for parameters, monotonicity, etc.

This one represents a vector of variations of one scalar.
"""

from dataclasses import dataclass
from typing import NamedTuple, Self

import jax
import jax.numpy as jnp

import pareto_curve_trainer.component_utils as cu


class Scalar(NamedTuple):
    """A vectorized, trainable component representing variants of a scalar parameter."""

    s: jax.typing.ArrayLike
    smin: jax.typing.ArrayLike
    smax: jax.typing.ArrayLike

    def perturbed(self, scale: float, key: jax.Array) -> Self:
        """Returns itself shifted by a random step in [-scale, +scale]."""
        new_s = cu.perturb_array(self.s, self.smin, self.smax, scale, key)
        return self._replace(s=new_s)

    def randomized(self, key: jax.Array) -> Self:
        """Returns itself with the scalar randomized."""
        new_s = cu.randomize_array(self.s, self.smin, self.smax, key)
        return self._replace(s=new_s)

    def value(self) -> jnp.ndarray:
        """Convert to a 1 x n array to support operations against 1D arrays."""
        if jnp.ndim(self.s) == 1:
            return jnp.reshape(self.s, shape=(-1, 1))
        return self.s

    def merge(self, other) -> Self:
        """Concatenates this `scalar` with another `scalar`."""
        s = jnp.concatenate([cu._1D(self.s), cu._1D(other.s)])
        smin = jnp.concatenate([cu._1D(self.smin), cu._1D(other.smin)])
        smax = jnp.concatenate([cu._1D(self.smax), cu._1D(other.smax)])
        return Scalar(s=s, smin=smin, smax=smax)

    # NamedTuple uses __len__ so we can't override that.
    def num_rows(self) -> int:
        """Returns the number of scalar values s in this `scalar`."""
        return len(self.s) if jnp.ndim(self.s) else 1

    def constraints_equal(self, other):
        """Returns whether the constraints match."""
        return (
            (jnp.shape(self.smin) == jnp.shape(other.smin))
            and (jnp.shape(self.smax) == jnp.shape(other.smax))
            and jnp.all(self.smin == other.smin)
            and jnp.all(self.smax == other.smax)
        )

    def __eq__(self, other: Self):
        s_identical = (jnp.shape(self.s) == jnp.shape(other.s)) and jnp.all(self.s == other.s)
        return s_identical & self.constraints_equal(other)

    # Normally __ne__ falls back to __eq__, but NamedTuple has its own __ne__ defined.
    # To avoid falling back to NamedTuple.__ne__, we have to specify our own __ne__.
    def __ne__(self, other):
        return not self.__eq__(other)


class Bool(NamedTuple):
    """A vectorized, trainable component representing variants of a boolean parameter."""

    s: jax.typing.ArrayLike

    def perturbed(self, scale: float, key: jax.Array) -> Self:
        """Returns itself with a `scale` chance of each row switching True<-->False."""
        new_s = cu.perturb_bool_array(self.s, scale, key)
        return self._replace(s=new_s)

    def randomized(self, key: jax.Array) -> Self:
        """Returns itself with the scalar randomized."""
        new_s = cu.randomize_bool_array(self.s, key)
        return self._replace(s=new_s)

    def value(self) -> jnp.ndarray:
        """Convert to a 1 x n array to support operations against 1D arrays."""
        if jnp.ndim(self.s) == 1:
            return jnp.reshape(self.s, shape=(-1, 1))
        return self.s

    def merge(self, other) -> Self:
        """Concatenates this `scalar` with another `scalar`."""
        s = jnp.concatenate([cu._1D(self.s), cu._1D(other.s)])
        return Bool(s=s)

    # NamedTuple uses __len__ so we can't override that.
    def num_rows(self) -> int:
        """Returns the number of scalar values s in this `scalar`."""
        return len(self.s) if jnp.ndim(self.s) else 1

    def __eq__(self, other: Self):
        return (jnp.shape(self.s) == jnp.shape(other.s)) and jnp.all(self.s == other.s)

    # Normally __ne__ falls back to __eq__, but NamedTuple has its own __ne__ defined.
    # To avoid falling back to NamedTuple.__ne__, we have to specify our own __ne__.
    def __ne__(self, other):
        return not self.__eq__(other)


@jax.tree_util.register_dataclass
@dataclass
class Categorical:
    """A vectorized, trainable component representing variants of a scalar parameter."""

    s: jax.typing.ArrayLike
    num_categories: int = jax.tree.static()

    def perturbed(self, scale: float, key: jax.Array) -> Self:
        """Returns itself shifted by a random step in [-scale, +scale]."""
        new_s = cu.perturb_categorical_array(self.s, self.num_categories, scale, key)
        return self.__replace__(s=new_s)

    def randomized(self, key: jax.Array) -> Self:
        """Returns itself with the scalar randomized."""
        new_s = cu.randomize_categorical_array(self.s, self.num_categories, key)
        return self.__replace__(s=new_s)

    def value(self) -> jnp.ndarray:
        """Convert to a 1 x n array to support operations against 1D arrays."""
        if jnp.ndim(self.s) == 1:
            return jnp.reshape(self.s, shape=(-1, 1))
        return self.s

    def merge(self, other) -> Self:
        """Concatenates this `scalar` with another `scalar`."""
        s = jnp.concatenate([cu._1D(self.s), cu._1D(other.s)])
        # Merging categoricals with different numbers of categories is suspicious but possible.
        num_categories = max(self.num_categories, other.num_categories)
        return Categorical(s=s, num_categories=num_categories)

    def num_rows(self) -> int:
        """Returns the number of scalar values s in this `scalar`."""
        return len(self.s) if jnp.ndim(self.s) else 1

    def constraints_equal(self, other):
        """Returns whether the constraints match."""
        return self.num_categories == other.num_categories

    def __eq__(self, other: Self):
        s_identical = (jnp.shape(self.s) == jnp.shape(other.s)) and jnp.all(self.s == other.s)
        return s_identical & self.constraints_equal(other)
