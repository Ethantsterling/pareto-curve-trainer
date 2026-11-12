"""Transforms """
from abc import ABC, abstractmethod, ABCMeta
import math
from typing import Callable, Literal, Tuple
import jax
import jax.numpy as jnp


class Transform(ABCMeta):
    """Abstract class that couples a fn:R-->R with its inverse."""

    # Overrides __init__ so we can use a Transform like a 
    def __call__(cls, x: jnp.ndarray) -> jnp.ndarray:
        """Forward transform."""
        return cls.__call__(x)

    @staticmethod
    @abstractmethod
    def inverse(x: jnp.ndarray) -> jnp.ndarray:
        """Inverse transform. Inverts .__call__."""


class Identity(metaclass=Transform):
    """The identity transform. Maps x to x."""

    @staticmethod
    def __call__(x: jnp.ndarray):
        """No-op. Returns the identity of x, which is x."""
        return x

    @staticmethod
    def inverse(x: jnp.ndarray):
        """No-op. Returns the inverse of the identity of x, which is x."""
        return x


class Symlog1p(metaclass=Transform):
    """A symmetric extension of log1p(x >= 0) to all real x, with an attached inverse."""

    @staticmethod
    def __call__(x: jnp.ndarray):
        """A symmetric extension of log1p(x >= 0) to all real x."""
        return jnp.sign(x) * jnp.log1p(jnp.abs(x))

    @staticmethod
    def inverse(x: jnp.ndarray):
        """The inverse of the symmetric extension of log1p(x >= 0) to all real x."""
        return jnp.sign(x) * jnp.expm1(jnp.abs(x))


class Log(metaclass=Transform):
    """The log transform extended to all real x by clipping inputs, with an attached inverse."""

    @staticmethod
    def __call__(x: jnp.ndarray):
        """An extension of log(x) to the reals. Negative values are mapped to -inf."""
        return jnp.log(jnp.clip(x, min=0))

    @staticmethod
    def inverse(x: jnp.ndarray):
        """The inverse of log, which is exp."""
        return jnp.exp(x)


def uniformity_divergence_score(sorted_xs: jnp.ndarray) -> float:
    """A score that's higher for less uniform distributions."""
    # Normalizes xs into [0, 1].
    normed_xs = (sorted_xs - sorted_xs[0]) / (sorted_xs[-1] - sorted_xs[0])
    uniform = jnp.linspace(0, 1, len(normed_xs))
    score = jnp.mean(jnp.abs(normed_xs - uniform))
    return score


def best_transform(xs: jnp.ndarray,
                   xmin: float = -math.inf,
                   xmax: float = math.inf,
                   transforms: Tuple[Transform] = (Identity, Log, Symlog1p),
                   divergence_fn=uniformity_divergence_score,
                   ):
    """Choose the transform that makes the distribution most uniform.

    PWLCurves work best when the input xs are distributed ~evenly over the domain.
    Many features have long-tailed distributions that span many orders of
    magnitude, and benefit from a log-like space. For example, most YouTube videos
    have only a few views, but some have billions. A linear treatment of #views
    is going to suffer. A piecewise-linear treatment can be better, but for best
    results we'd really rather transform the input feature first.

    This function will automatically filter out transforms that return nan or inf
    for any of the xs.

    Note that it may be desirable to bias in favor of the identify, because the
    identity is simpler and faster to compute. However, that would be complicated
    to calibrate, so for the moment we won't.

    args:
      xs: The input domain. Ideally, this is representative training data.
      xmin: Optional lower bound. Useful if we don't care to distinguish data
          below the given cutoff.
      xmax: Optional upper bound. Useful if we don't care to distinguish data
          above the given cutoff.
      transforms: The transforms to choose between. log elegantly swaps
          exponential space into linear space, but only works for positive values.
          symlog1p is less elegant but more robust, and is ~linear near 0.
      divergence_fn: A function to measure divergence from uniformity. Lower
          numbers are better.
    """
    if transforms is None:
        return None
    if isinstance(transforms, Transform):
        return transforms
    xs = xs.clip(xmin, xmax)
    xs = xs[jnp.isfinite(xs)]  # Missing/infinite values aren't relevant.
    xs = xs.sort()
    if xs[0] == xs[-1]:
        return transforms.Identity  # The data is useless here. Unclear what we should do.

    scores = []
    valid_fxs = []
    for fx in transforms:
        # Skip a transform if it would produce invalid values.
        transformed_xs = fx(xs)
        if not jnp.all(jnp.isfinite(transformed_xs)):
            continue

        scores.append(divergence_fn(transformed_xs))
        valid_fxs.append(fx)

    return valid_fxs[scores.index(min(scores))]
