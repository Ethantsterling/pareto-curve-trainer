"""Tools for analyzing a dataset before and after training."""

import altair as alt


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
