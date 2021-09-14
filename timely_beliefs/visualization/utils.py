from datetime import timedelta
from typing import List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd

from timely_beliefs.beliefs import classes  # noqa: F401
from timely_beliefs.beliefs.probabilistic_utils import (
    get_nth_percentile_belief,
    interpret_complete_cdf,
)
from timely_beliefs.visualization import graphs, selectors


def plot(
    bdf: "classes.BeliefsDataFrame",
    show_accuracy: bool = True,
    active_fixed_viewpoint_selector: bool = True,
    reference_source: "classes.BeliefSource" = None,
    ci: float = 0.9,
    intuitive_forecast_horizon: bool = True,
    interpolate: bool = True,
    event_value_range: Tuple[Optional[float], Optional[float]] = (None, None),
) -> alt.LayerChart:
    """Plot the BeliefsDataFrame with the Altair visualization library.

    :param bdf: The BeliefsDataFrame to visualize
    :param show_accuracy: If true, show additional graphs with accuracy over time and horizon
    :param active_fixed_viewpoint_selector: If true, fixed viewpoint beliefs can be selected
    :param reference_source: The BeliefSource serving as a reference for accuracy calculations
    :param ci: The confidence interval to highlight in the time series graph
    :param intuitive_forecast_horizon: If true, horizons are shown with respect to event start rather than knowledge time
    :param interpolate: If True, the time series chart shows a user-friendly interpolated line rather than more accurate stripes indicating average values
    :param event_value_range: Optionally set explicit limits on the range of event values (for axis scaling).
    :return: Altair LayerChart
    """

    # Validate input
    if reference_source is None:
        raise ValueError("Must set reference source.")

    # Set up data source
    bdf = bdf.copy()
    bdf["event_value"] = bdf["event_value"].astype(float)
    sensor_name = bdf.sensor.name
    sensor_unit = bdf.sensor.unit if bdf.sensor.unit != "" else "a.u."  # arbitrary unit
    plottable_df, belief_horizon_unit = prepare_df_for_plotting(
        bdf,
        ci=ci,
        show_accuracy=show_accuracy,
        reference_source=reference_source,
        intuitive_forecast_horizon=intuitive_forecast_horizon,
    )
    unique_belief_horizons = plottable_df["belief_horizon"].unique()

    # Set range of event values
    if None in event_value_range:
        event_value_range = list(event_value_range)
        if event_value_range[0] is None:
            # Infer minimum from data
            event_value_range[0] = bdf["event_value"].min()
        if event_value_range[-1] is None:
            # Infer maximum from data
            event_value_range[-1] = bdf["event_value"].max()
        event_value_range = tuple(event_value_range)

    plottable_df = plottable_df.groupby(
        ["event_start", "source"], group_keys=False
    ).apply(
        lambda x: align_belief_horizons(x, unique_belief_horizons)
    )  # Propagate beliefs so that each event has the same set of unique belief horizons
    max_absolute_error = plottable_df["mae"].max() if show_accuracy is True else None

    # Construct base chart
    base = graphs.time_series_base_chart(
        plottable_df, belief_horizon_unit, intuitive_forecast_horizon
    )

    # Construct selectors
    time_window_selector = selectors.time_window_selector(base, interpolate)
    if show_accuracy is True:
        horizon_selection_brush = selectors.horizon_selection_brush()
        horizon_selector = selectors.horizon_selector(
            base.properties(width=1300),
            horizon_selection_brush,
            belief_horizon_unit,
            intuitive_forecast_horizon,
            unique_belief_horizons,
        )

    # Add generic filters to base chart
    base = base.transform_filter(selectors.source_selection_brush).transform_filter(
        selectors.time_selection_brush
    )
    if show_accuracy is True:
        filtered_base = base.transform_filter(horizon_selection_brush)
    else:
        filtered_base = base

    # Construct charts
    ts_chart = graphs.value_vs_time_chart(
        filtered_base,
        active_fixed_viewpoint_selector,
        sensor_name,
        sensor_unit,
        belief_horizon_unit,
        intuitive_forecast_horizon,
        interpolate,
        ci,
        event_value_range,
    )
    # Initialize belief time to (roughly) the middle of the event time window
    event_starts = bdf.event_starts
    init_value = event_starts[len(event_starts) // 2] if len(event_starts) > 0 else None
    if show_accuracy is True:
        ha_chart = graphs.accuracy_vs_horizon_chart(
            base.properties(width=1300),
            horizon_selection_brush,
            belief_horizon_unit,
            intuitive_forecast_horizon,
            unique_belief_horizons,
        )
        hd_chart = graphs.source_vs_hour_chart(
            filtered_base.properties(height=290), sensor_unit, max_absolute_error
        )
        return (
            (
                (
                    (time_window_selector | selectors.source_selector(plottable_df))
                    & selectors.fixed_viewpoint_selector(
                        base,
                        active_fixed_viewpoint_selector=active_fixed_viewpoint_selector,
                        init_value=init_value,
                    )
                    + ts_chart
                    | hd_chart
                )
                & (horizon_selector & ha_chart)
            )
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
        )
    else:
        return (
            (
                (
                    time_window_selector
                    & selectors.fixed_viewpoint_selector(base, init_value=init_value)
                    + ts_chart
                )
                | selectors.source_selector(plottable_df)
            )
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
        )


def timedelta_to_human_range(s: pd.Series) -> Tuple[pd.Series, str]:
    """Convert a pandas Series of timedeltas to a pandas Series of floats,
    and derive a time unit (a string such as "years" or "minutes") that gives a nice human readable range of floats.
    For example:

    >>> import timely_beliefs as tb
    >>> horizons = tb.examples.get_temperature_df().convert_index_from_belief_time_to_horizon().reset_index()["belief_horizon"].drop_duplicates()
    >>> horizons  # This is going to look awkward as tick labels
    <<< 0     0 days 00:00:00
        1     0 days 01:00:00
        4     0 days 02:00:00
        7     0 days 03:00:00
        10    0 days 04:00:00
                    ...
        205   2 days 21:00:00
        208   2 days 22:00:00
        211   2 days 23:00:00
        214   3 days 00:00:00
        217   3 days 01:00:00
        Length: 74, dtype: timedelta64[ns]
    >>> from timely_beliefs.visualization.utils import timedelta_to_human_range
    >>> timedelta_to_human_range(horizons)  # This is human readable range, though
    <<< (0       0.0
        1       1.0
        4       2.0
        7       3.0
        10      4.0
               ...
        205    69.0
        208    70.0
        211    71.0
        214    72.0
        217    73.0
        Length: 74, dtype: float64, 'hours')
    """
    timedelta_span = max(s) - min(s)
    if timedelta_span >= timedelta(days=4 * 365.2425):
        s = s.apply(
            lambda x: x.days / 365.2425
            + (x.seconds + x.microseconds / 10 ** 6) / (365.2425 * 24 * 60 * 60)
        )
        time_unit = "years"
    elif timedelta_span >= timedelta(days=4):
        s = s.apply(
            lambda x: x.days + (x.seconds + x.microseconds / 10 ** 6) / (24 * 60 * 60)
        )
        time_unit = "days"
    elif timedelta_span >= timedelta(hours=4):
        s = s.apply(
            lambda x: x.days * 24 + (x.seconds + x.microseconds / 10 ** 6) / (60 * 60)
        )
        time_unit = "hours"
    elif timedelta_span >= timedelta(minutes=4):
        s = s.apply(
            lambda x: x.days * 24 * 60 + (x.seconds + x.microseconds / 10 ** 6) / 60
        )
        time_unit = "minutes"
    elif timedelta_span >= timedelta(seconds=4):
        s = s.apply(
            lambda x: x.days * 24 * 60 * 60 + x.seconds + x.microseconds / 10 ** 6
        )
        time_unit = "seconds"
    else:
        s = s.apply(lambda x: x.days * 24 * 60 * 60 * 10 ** 6 + x.microseconds)
        time_unit = "microseconds"
    return s, time_unit


def prepare_df_for_plotting(
    df: "classes.BeliefsDataFrame",
    ci: float = 0.95,
    show_accuracy: bool = False,
    reference_source: "classes.BeliefSource" = None,
    intuitive_forecast_horizon: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Convert a probabilistic BeliefsDataFrame to a Pandas DataFrame, and calculate accuracy metrics if needed.

    :param df: BeliefsDataFrame to visualize
    :param ci: Confidence interval (default is 95%)
    :param show_accuracy: If true, calculate accuracy for each (probabilistic or deterministic) belief
    :param reference_source: BeliefSource to indicate that the accuracy should be determined with respect to the beliefs held by the given source
    :param intuitive_forecast_horizon: if True, anchor horizons to event_start rather than knowledge_time
    :return: Two things:
    - a DataFrame with probabilistic beliefs mapped to three columns (lower_value, expected_value and upper_value),
    and horizons expressed as a (float) number of time units (e.g. days, hours or minutes), sorted by event_start,
    belief_horizon and source.
    - a string representing the time unit for the horizons
    """
    event_resolution = df.event_resolution
    df = df.convert_index_from_belief_horizon_to_time()
    if show_accuracy is True:
        reference_df = df.groupby(level="event_start").apply(
            lambda x: x.accuracy(reference_source=reference_source, lite_metrics=True)
        )
    else:
        reference_df = (
            df.set_reference_values(
                reference_source=reference_source, return_expected_value=True
            )
            .drop(columns="event_value")
            .for_each_belief(get_nth_percentile_belief, 50)
            .convert_index_from_belief_time_to_horizon()
            .droplevel("cumulative_probability")
        )
    df["belief_horizon"] = df.knowledge_times - df.belief_times
    if df.lineage.percentage_of_probabilistic_beliefs == 0:
        df = df.droplevel("cumulative_probability")
        df["expected_value"] = df["event_value"]
        df["upper_value"] = df["event_value"]
        df["lower_value"] = df["event_value"]
    else:
        df_ci0 = (
            df.for_each_belief(get_nth_percentile_belief, n=(1 - ci) * 100 / 2)
            .rename(columns={"event_value": "lower_value"})
            .droplevel("cumulative_probability")
            .drop("belief_horizon", axis=1)
        )
        df_exp = (
            df.for_each_belief(get_nth_percentile_belief, n=50)
            .rename(columns={"event_value": "expected_value"})
            .droplevel("cumulative_probability")
        )
        df_ci1 = (
            df.for_each_belief(get_nth_percentile_belief, n=100 - (1 - ci) * 100 / 2)
            .rename(columns={"event_value": "upper_value"})
            .droplevel("cumulative_probability")
            .drop("belief_horizon", axis=1)
        )
        df = pd.concat([df_ci0, df_exp, df_ci1], axis=1)
    df = df.reset_index().set_index(["event_start", "belief_horizon", "source"])
    df = pd.concat([df, reference_df], axis=1)
    df = df.reset_index().sort_values(
        ["event_start", "belief_horizon", "source"], ascending=[True, True, True]
    )
    if intuitive_forecast_horizon is True:
        df["belief_horizon"] = df["event_start"] - df["belief_time"]
    df["belief_horizon"], belief_horizon_unit = timedelta_to_human_range(
        df["belief_horizon"]
    )

    df["event_end"] = df["event_start"] + event_resolution
    df["source"] = df["source"].apply(lambda x: x.name)

    return df, belief_horizon_unit


def align_belief_horizons(
    df: pd.DataFrame, unique_belief_horizons: List[float]
) -> pd.DataFrame:
    applicable_horizons = sorted(
        [h for h in unique_belief_horizons if h <= max(df["belief_horizon"])],
        reverse=True,
    )
    # Build up input data for new BeliefsDataFrame
    data = []
    previous_slice_with_existing_belief_horizon = None
    for ubh in applicable_horizons:

        # Check if the unique belief horizon (ubh) is already in the DataFrame
        if ubh not in df["belief_horizon"].values:

            # If not already present, create a new row with the most recent belief
            if previous_slice_with_existing_belief_horizon is not None:
                previous_slice_with_existing_belief_horizon[
                    "belief_horizon"
                ] = ubh  # Update belief horizon to reflect propagation of beliefs over time
                data.extend(previous_slice_with_existing_belief_horizon.values.tolist())
        else:
            # If already present, copy the row (may be multiple rows in case of a probabilistic belief)
            slice_with_existing_belief_horizon = df.loc[df["belief_horizon"] == ubh]
            data.extend(slice_with_existing_belief_horizon.values.tolist())
            previous_slice_with_existing_belief_horizon = (
                slice_with_existing_belief_horizon
            )
    df2 = df.copy().iloc[0:0]
    df2 = df2.append(pd.DataFrame(data, columns=df2.columns))
    return df2


def ridgeline_plot(
    bdf,
    fixed_viewpoint: bool = False,
    distribution: str = "uniform",
    event_value_window: Tuple[float, float] = None,
) -> alt.FacetChart:
    """
    Creates ridgeline plot

    :param bdf: BeliefsDataFrame
    :param fixed_viewpoint: boolean, if true create fixed viewpoint plot
    :param distribution: string, distribution name to use (discrete, normal or uniform)
    :param event_value_window: optional tuple specifying an event value window for the x-axis
           (e.g. plot temperatures between -1 and 21 degrees Celsius)
    """
    df = interpret_and_sample_distribution_long_form(
        bdf, distribution=distribution, event_value_window=event_value_window
    ).reset_index()
    df["belief_horizon"], belief_horizon_unit = timedelta_to_human_range(
        df["belief_horizon"]
    )

    # Set defaults for timely-beliefs ridgeline plots
    step = 10
    overlap = 50
    probability_scale_range = (step, -step * overlap)
    if belief_horizon_unit == "hours":
        y_label_once_every_n_horizon_steps = 6
    elif belief_horizon_unit == "days":
        y_label_once_every_n_horizon_steps = 7
    else:
        y_label_once_every_n_horizon_steps = 5

    deterministic_chart = graphs.deterministic_chart(probability_scale_range)
    probabilistic_chart = graphs.probabilistic_chart(
        probability_scale_range,
        belief_horizon_unit=belief_horizon_unit,
        sensor_name=bdf.sensor.name,
        sensor_unit=bdf.sensor.unit,
    )
    ridgeline_chart = (
        alt.layer(
            probabilistic_chart,
            deterministic_chart,
            selectors.ridgeline_selector(probability_scale_range, belief_horizon_unit),
            data=df,
        )
        .properties(height=step)
        .facet(
            row=alt.Row(
                "belief_horizon:N",
                sort="descending",
                title=("Upcoming " if fixed_viewpoint else "Previous ")
                + belief_horizon_unit,
                header=alt.Header(
                    labelAngle=0,
                    labelAlign="left",
                    labelExpr=f"datum.value % {y_label_once_every_n_horizon_steps} == 0 ? datum.value : ''",
                ),
            )
        )
        .properties(bounds="flush")
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
        .configure_title(anchor="end")
    )

    return ridgeline_chart


def interpret_and_sample_distribution_long_form(
    df: "classes.BeliefsDataFrame",
    distribution: str = "uniform",
    event_value_window: Tuple[float, float] = None,
) -> pd.DataFrame:
    """Interpret each probabilistic belief as a continuous or discrete distribution of possible outcomes (an openturns distribution),
    collect a sample of points that is adequate to draw its PDF (using the drawPDF attribute on openturns distributions),
    and concatenate those points for each belief to return a pandas DataFrame (long form) with the following columns:
    "event_value", "probability" and "belief_horizon".
    """
    frame = pd.DataFrame()
    for _group_index, df in df.for_each_belief():

        # Interpret CDF
        dist = interpret_complete_cdf(
            cdfs_p=[df.index.get_level_values("cumulative_probability").values],
            cdfs_v=[df["event_value"].values],
            distribution=distribution,
        )

        # Draw PDF
        graph = (
            dist[0].drawPDF(event_value_window[0], event_value_window[1])
            if event_value_window is not None
            else dist[0].drawPDF()
        )
        new_frame = pd.DataFrame(
            np.array(graph.getDrawable(0).getData()),
            columns=["event_value", "probability"],
        ).set_index("event_value")

        new_frame["belief_horizon"] = df.lineage.belief_horizons[0]

        frame = pd.concat([frame, new_frame]) if not frame.empty else new_frame

    return frame
