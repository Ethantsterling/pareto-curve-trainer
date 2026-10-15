"""A trainable vectorized piecewise-linear-curve component and supporting functions."""
import jax
import jax.numpy as jnp

from functools import partial
from typing import Callable, List, NamedTuple, Self

import pareto_curve_trainer.component_utils as cu
from pareto_curve_trainer.component_utils import Mono


def jax_curve_slopes(
    curve_xs: jnp.ndarray,
    curve_ys: jnp.ndarray) -> jnp.ndarray:
    """The slopes for a single curve."""
    slopes = ((curve_ys[1:] - curve_ys[:-1]) /
              (curve_xs[1:] - curve_xs[:-1]))
    return jnp.concatenate([jnp.array([0]), slopes])


@partial(jax.jit, static_argnames=['fx'])
def jax_curve_eval(
    curve_xs: jnp.ndarray,
    curve_ys: jnp.ndarray,
    xs: jnp.ndarray,
    fx: Callable[[jnp.ndarray], jnp.ndarray] | None = None) -> jnp.ndarray:
    """Functional style of curve eval, forgoing curve objects.

      JAX expects a functional style whereas PWLCurve is written object-oriented,
      so this is a functional evaluator for pwlcurves. Presumably we'd call this
      or a jitted function like it from the object-oriented version's .eval().

      TODO: add an fx transform. We can do that with a static argument. That'll
      force us to jit multiple times, one for each fx argument, but that's fine.
      There aren't many different fx's that we use anyway.
      use to re-jit every time we change the fx argument, but that's fine.
      It's not like we'll use that many different fx's.

      Args:
        xs: 1D JAX array of data to evaluate the curve on.

      Returns:
        1D JAX array of the PWLCurve's value at each x in xs.
      """
    xs = jnp.clip(xs, curve_xs[0], curve_xs[-1])

    if fx is not None:
        xs = fx(xs)
        curve_xs = fx(curve_xs)
    slopes = jax_curve_slopes(curve_xs, curve_ys)

    # Index of the first knot >= x. Intuitively, it's the knot we 'round up' to.
    indices = curve_xs.searchsorted(xs)
    knot_y = curve_ys[indices]
    knot_x = curve_xs[indices]
    slope = slopes[indices]  # Slope leading up to the knot.

    xgap = knot_x - xs
    return knot_y - xgap * slope


def jax_curves_slopes(curves_xs: jnp.ndarray,
                      curves_ys: jnp.ndarray) -> jnp.ndarray:
    """Calculate the slopes of a vector of curves.

    # Suppose we have a curve with 3 knots. Searchsorted can return 4 positions:
      # 0 (x <= cxs[0])
      # 1 (cxs[0] < x <= cxs[1])
      # 2 (cxs[1] < x <= cxs[2])
      # 3 (cxs[2] < x)
    
    Args:
      curve_xs: (num_curves, curve_size) 2D array of knot-x floats.
      curve_ys: (num_curves, curve_size) 2D array of knot-y floats.

    Returns:
      (num_curves, curve_size + 1) 2D array where [i][j] is the slope leading
      into the jth knot of the ith curve. PWLCurves are always flat before their
      leftmost knot and after their rightmost knot, so [i][0] and [i][-1] are
      always 0.
    """
    assert jnp.ndim(curves_xs) == 2 == jnp.ndim(curves_ys)
    # The slopes between knots.
    slopes = ((curves_ys[:, 1:] - curves_ys[:, :-1]) /
              (curves_xs[:, 1:] - curves_xs[:, :-1]))
    # Set the preceding slope and trailing slope to zero.
    return jnp.pad(slopes, ((0, 0), (1, 1)))


# @partial(jax.jit, static_argnames=['fx'])
def jax_curves_eval(
    curves_xs: jnp.ndarray,  # 2D, num_curves x num_points_per_curve
    curves_ys: jnp.ndarray,  # 2D, num_curves x num_points_per_curve
    xs: jnp.ndarray,  # 1D
    fx: Callable[[jnp.ndarray], jnp.ndarray] | None = None) -> jnp.ndarray:
    """Functional style of curve eval, forgoing curve objects.

      JAX expects a functional style whereas PWLCurve is written object-oriented,
      so this is a functional evaluator for pwlcurves. Presumably we'd call this
      or a jitted function like it from the object-oriented version's .eval().

      TODO: add an fx transform. We can do that with a static argument. That'll
      force us to jit multiple times, one for each fx argument, but that's fine.
      There aren't many different fx's that we use anyway.
      use to re-jit every time we change the fx argument, but that's fine.
      It's not like we'll use that many different fx's.

      Args:
        xs: Data to evaluate the curve on. 1D JAX array.

      Returns:
        1D JAX array of the PWLCurve's value at each x in xs.
    """
    if fx is not None:
        xs = fx(xs)
        curves_xs = fx(curves_xs)
    # Index of the first knot >= x. Intuitively, it's the knot we 'round up' to.

    # 2D num_curves x num_xs array of the sorting index into each curve of each x.
    indices = jax.vmap(lambda curve_xs: curve_xs.searchsorted(xs))(curves_xs)
    # Note: when x > max(curve_xs), index will be len(curve_xs).
    # We can still do curve_xs[index], because JAX automatically clips indices
    # into an array. If a = jnp.array([1,2,3]), then a[2] == a[3] == a[50].
    # Note: if x > max(curve_xs) for a curve, then the index will be len(curve_xs).
    # Then we'd think curve_xs[index] would brea
    # indices = indices.clip(0, curve_size - 1)  # In case x > max curve x.
    row_select = jnp.arange(curves_xs.shape[0])[:, None]

    # knot_[i][j] = curves_[i][indices[i][j]].
    knot_y = curves_ys[row_select, indices]
    knot_x = curves_xs[row_select, indices]
    slopes = jax_curves_slopes(curves_xs, curves_ys)
    slope = slopes[row_select, indices]  # Slope leading up to the knot.
    xgap = knot_x - xs
    # TODO: Deal with the case where some xs are infinite,
    # which causes xgap = +-inf and slope=0, resulting in a 0 * inf = nan output.
    # To avoid nans, we'll need to deal with the case where xs (and therefore xgap)
    # are infinite. That can happen with a trouble transform.
    return knot_y - xgap * slope


class StructPwlcurve(NamedTuple):
    """A single PWLCurve. Not vectorized."""
    xs: jnp.ndarray
    ys: jnp.ndarray
    fx: Callable[[jnp.ndarray], jnp.ndarray] | None = None

    def eval(self, xs):
        """Evaluates the curve on a vector of data points."""
        return jax_curve_eval(self.xs, self.ys, xs, fx=self.fx)

    def __str__(self):
        points = list(zip(self.xs.tolist(), self.ys.tolist()))
        if self.fx is None or self.fx.__name__ == 'identity':
            return f"PWLCurve({points})"
        return f"PWLCurve({points}, fx={self.fx.__name__})"


# @jax.jit
class Pwlcurve(NamedTuple):
    """A vectorized, trainable component representing variants of a pwlcurve."""
    xs: jnp.ndarray    # 2D num_curves x curve_size floats
    ys: jnp.ndarray    # 2D num_curves x curve_size floats
    xmin: float  # jnp.ndarray  # 1D floats
    xmax: float  # jnp.ndarray  # 1D floats
    ymin: float  # jnp.ndarray  # 1D floats
    ymax: float  # jnp.ndarray  # 1D floats
    mono: Mono   # jnp.ndarray  # 1D array of ints representing the Mono enum values.
    # For now, let's omit fx.

    def eval(self, data: jnp.ndarray) -> jnp.ndarray:
        """Calculates each pwlcurve on each data point."""
        if jnp.ndim(self.xs) == 1:
            # This allows us to do eval in a jax map/vmap if we want.
            return jax_curve_eval(self.xs, self.ys, data, fx=None)
        return jax_curves_eval(self.xs, self.ys, data, fx=None)

    def perturbed(self, scale: float, key: jax.Array) -> Self:
        """Randomly shifts each curve coordinate by up to a scaled fraction of its range."""
        key_x, key_y = jax.random.split(key)
        xs = cu.perturb_array(self.xs, self.xmin, self.xmax, scale, key_x)
        ys = cu.perturb_array(self.ys, self.ymin, self.ymax, scale, key_y)
        # TODO: when we add fx transforms, do the perturbing in the transformed space.
        # This likely requires transforming xs & constraints, perturbing, and transforming
        # back, which also requires transforms be reversible.

        # Always sort xs. Follow the mono constraints for ys.
        xs = jnp.sort(xs)  # Sorts on rightmost axis by default, which is what we want.
        ys = cu.vectorized_conditional_mono_sort(ys, self.mono)
        new_trainable_curve = self._replace(xs=xs, ys=ys)
        return new_trainable_curve

    def randomized(self, key: jax.Array) -> Self:
        """Returns itself with randomized curve params."""
        return self.perturbed(scale=1.0, key=key)

    def merge(self, other: Self) -> Self:
        """Concatenate this vector of curves with another."""
        return self._replace(
            xs=jnp.concatenate([self.xs, other.xs]),
            ys=jnp.concatenate([self.ys, other.ys]),
            xmin=jnp.concatenate([self.xmin, other.xmin]),
            xmax=jnp.concatenate([self.xmax, other.xmax]),
            ymin=jnp.concatenate([self.ymin, other.ymin]),
            ymax=jnp.concatenate([self.ymax, other.ymax]),
            mono=jnp.concatenate([self.mono, other.mono]),
        )

    def as_struct_pwlcurves(self) -> List[StructPwlcurve]:
        """Returns a list of single pwlcurves without training constraints."""
        return [StructPwlcurve(xarray, yarray) for
                (xarray, yarray) in zip(self.xs, self.ys)]

    # NamedTuple uses __len__ so we can't override that.
    def num_rows(self) -> int:
        """Returns the number of pwlcurves represented by this component."""
        return len(self.xs)

    def constraints_equal(self, other):
        """Check if this component's constraints match another component's."""
        return (jnp.all(self.xmin == other.xmin) &
                jnp.all(self.xmax == other.xmax) &
                jnp.all(self.ymin == other.ymin) &
                jnp.all(self.ymax == other.ymax) &
                jnp.all(self.mono == other.mono))

    def __eq__(self, other):
        return (jnp.all(self.xs == other.xs) &
                jnp.all(self.ys == other.ys) &
                self.constraints_equal(other))
