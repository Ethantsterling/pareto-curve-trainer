"""Tools for analyzing a dataset before and after training."""

import altair as alt
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pareto_curve_trainer import component_catmap, component_pwlcurve, component_scalar, transforms


def plot_2d(
    df,
    x: str,
    y: str,
    tooltip: list[str] | None = None,
    category: None | str = None,
    opacity_bool: None | str = None,
) -> alt.Chart:
    """2D interactive plot intended for pareto frontiers.

    After training a pareto frontier of models, it's helpful to display the metric tradeoffs
    visually to help us select a favorite tradeoff. A 2D plot is most effective when there are
    precisely two metrics, or when we can reduce results from 3+ metrics down to two metrics.
    (For example, by converting the other metrics into thresholds. Say we have metrics A, B, C.
    We could filter the dataset to C >= chosen_threshold, and then plot a frontier comparing A and
    B, satistfied that we're only consider cases where C is acceptably good.)

    Args:
      df: A pandas dataframe containing all features used in the plot.
      x: Feature name such that df[x] is the x-coordinate.
      y: Feature name such that df[y] is the y-coordinate.
      tooltip: Additional feature names within df to display when hovering over a point.
      category: Optional feature name to visually distinguish different kinds of points. Ideal for
        comparing the results of different model training runs or different model architectures.
      opacity_bool: Optional feature name to highlight True points and fade False points.
        Normally used to highlight the pareto frontier and de-emphasize dominated points.
    """
    tooltip = [] if tooltip is None else tooltip
    # Always include these as tooltip labels.
    for label in [x, y, category, opacity_bool]:
        if label is not None and label not in tooltip:
            tooltip.append(label)
    params = {
        "tooltip": tooltip,
        "x": alt.X(x, scale=alt.Scale(zero=False)),
        "y": alt.Y(y, scale=alt.Scale(zero=False)),
    }
    if opacity_bool is not None:
        params["opacity"] = alt.Opacity(opacity_bool, scale=alt.Scale(range=[0.2, 0.8]))
    if category:
        params["color"] = alt.Color(category, scale=alt.Scale(scheme="dark2"))

    return alt.Chart(df).mark_circle(size=60).encode(**params).interactive()


def shuffle_importance(
    xs: dict[jnp.array],
    components: dict,
    model_fn,
    metrics_fn,
    features: dict,
    key=jax.random.key(8345),
) -> dict[jnp.array]:
    """Measure the metric loss from shuffling features."""
    baseline = jnp.array(metrics_fn(xs, model_fn(components, xs)))
    keys = jax.random.split(key, num=len(features))

    metric_diffs = {}
    for fkey, feature in zip(keys, features):
        feature_x = xs[feature]
        shuffled_feature_x = jax.random.permutation(fkey, feature_x)
        # Calculate metrics with the one feature ablated through shuffling.
        xs[feature] = shuffled_feature_x
        shuffled_metrics_x = jnp.array(metrics_fn(xs, model_fn(components, xs)))
        metric_diffs[feature] = shuffled_metrics_x - baseline
        # Restore xs.
        xs[feature] = feature_x

    return metric_diffs


def plot_pwlcurve(pwlcurve: component_pwlcurve.Pwlcurve, name: str):
    """Plot a learned piecewise-linear curve."""
    xs, ys, fx = pwlcurve.xs, pwlcurve.ys, pwlcurve.fx
    if jnp.ndim(xs) == 2:
        xs = xs[0]
        ys = ys[0]

    ax = plt.axes()
    plt.plot(xs, ys)
    plt.scatter(xs, ys)
    if fx is not None and fx is not transforms.Identity:
        fx = fx.as_numpy()  # For compatibility with matplotlib.
        ax.set_xscale("function", functions=(fx, fx.inverse))

    plt.legend(labels=[name])
    plt.show()


def plot_catmap(catmap: component_catmap.Catmap, name: str):
    """Plot a learned categorical map."""
    ys = catmap.map.ys
    if jnp.ndim(ys) == 2:
        ys = ys[0]
    categories = [str(y) for y in ys]
    plt.bar(categories, ys, label=name)
    plt.legend(labels=[name])
    plt.show()


def _get_scalar_value(s):
    value = s.s
    if jnp.ndim(value) == 1:
        return value[0]
    elif jnp.ndim(value) == 2:
        return value[0][0]
    else:
        return value


def plot_bool(s: component_scalar.Bool, name: str):
    """'Plot' a boolean constant by printing it out."""
    print(f"{name}: {_get_scalar_value(s)}")


def plot_categorical(s: component_scalar.Categorical, name: str):
    """'Plot' a categorical constant by printing it out."""
    print(f"{name}: {_get_scalar_value(s)} (categorical)")


def plot_scalar(s: component_scalar.Scalar, name: str):
    """'Plot' a scalar constant by printing it out."""
    print(f"{name}: {_get_scalar_value(s)}")


def plot_component(component, name):
    """Plot a learned component."""
    if isinstance(component, component_catmap.Catmap):
        plot_catmap(component, name)
    elif isinstance(component, component_pwlcurve.Pwlcurve):
        plot_pwlcurve(component, name)
    elif isinstance(component, component_scalar.Bool):
        plot_bool(component, name)
    elif isinstance(component, component_scalar.Categorical):
        plot_categorical(component, name)
    elif isinstance(component, component_scalar.Scalar):
        plot_scalar(component, name)
    else:
        print(f"Could not identify component {name}: {component}")


def plot_components(components):
    """Plot a dict of learned components."""
    for name, component in components.items():
        plot_component(component, name)
