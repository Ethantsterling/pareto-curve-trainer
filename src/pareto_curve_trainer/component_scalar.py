"""A trainable scalar 
Training wrappers around 0D, 1D, and 2D functions that include training
constraints such as bounds for parameters, monotonicity, etc.

This one represents a vector of variations of one scalar.
"""
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
            return jnp.reshape(self.s, shape=(-1,1))
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
        """Returns whether the constraints """
        return ((jnp.shape(self.smin) == jnp.shape(other.smin)) &
                (jnp.shape(self.smax) == jnp.shape(other.smax)) &
                jnp.all(self.smin == other.smin) &
                jnp.all(self.smax == other.smax))

    def __eq__(self, other: Self):
        s_identical = (jnp.shape(self.s) == jnp.shape(other.s)) & jnp.all(self.s == other.s)
        return s_identical & self.constraints_equal(other)
