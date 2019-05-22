from datetime import timedelta
from typing import List, Tuple

import altair as alt
import pandas as pd

from timely_beliefs.beliefs import classes  # noqa: F401
from timely_beliefs.beliefs.probabilistic_utils import get_nth_percentile_belief
from timely_beliefs.visualization import graphs, selectors


def plot(
    bdf: "classes.BeliefsDataFrame",
    show_accuracy: bool = True,
    active_fixed_viewpoint_selector: bool = True,
    reference_source: "classes.BeliefSource" = None,
    ci: float = 0.9,
    intuitive_forecast_horizon: bool = True,
    interpolate: bool = True,
    plottable_df: Tuple[pd.DataFrame, str, str, Tuple[float, float]] = None,
) -> alt.LayerChart:
    """Plot the BeliefsDataFrame with the Altair visualization library.

    :param bdf: The BeliefsDataFrame to visualize
    :param show_accuracy: If true, show additional graphs with accuracy over time and horizon
    :param active_fixed_viewpoint_selector: If true, fixed viewpoint beliefs can be selected
    :param reference_source: The BeliefSource serving as a reference for accuracy calculations
    :param ci: The confidence interval to highlight in the time series graph
    :param intuitive_forecast_horizon: If true, horizons are shown with respect to event start rather than knowledge time
    :param interpolate: If True, the time series chart shows a user-friendly interpolated line rather than more accurate stripes indicating average values
    :param plottable_df: Optionally, specify as plottable DataFrame directly together with a sensor name, a sensor unit and a y-axis value range for the event values (if None, we create it)
    :return: Altair LayerChart
    """

    # Validate input
    if reference_source is None:
        raise ValueError("Must set reference source.")

    # Set up data source
    if plottable_df is None:
        bdf = bdf.copy()
        sensor_name = bdf.sensor.name
        sensor_unit = (
            bdf.sensor.unit if bdf.sensor.unit != "" else "a.u."
        )  # arbitrary unit
        plottable_df, belief_horizon_unit = prepare_df_for_plotting(
            bdf,
            ci=ci,
            show_accuracy=show_accuracy,
            reference_source=reference_source,
            intuitive_forecast_horizon=intuitive_forecast_horizon,
        )
        unique_belief_horizons = plottable_df["belief_horizon"].unique()
        event_value_range = (bdf.min()[0], bdf.max()[0])
        plottable_df = plottable_df.groupby(
            ["event_start", "source"], group_keys=False
        ).apply(
            lambda x: align_belief_horizons(x, unique_belief_horizons)
        )  # Propagate beliefs so that each event has the same set of unique belief horizons
    else:
        sensor_name = plottable_df[1]
        sensor_unit = plottable_df[2]
        event_value_range = plottable_df[3]
        plottable_df = plottable_df[0]
        unique_belief_horizons = plottable_df["belief_horizon"].unique()
    max_absolute_error = plottable_df["mae"].max() if show_accuracy is True else None

    # Construct base chart
    base = graphs.time_series_base_chart(
        plottable_df, belief_horizon_unit, intuitive_forecast_horizon
    )

    # Construct selectors
    time_window_selector = selectors.time_window_selector(base, interpolate)
    if show_accuracy is True:
        horizon_selection_brush = selectors.horizon_selection_brush(
            init_belief_horizon=unique_belief_horizons[0]
        )
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
                    & selectors.fixed_viewpoint_selector(base) + ts_chart
                )
                | selectors.source_selector(plottable_df)
            )
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
        )


def timedelta_to_human_range(
    df: "classes.BeliefsDataFrame"
) -> Tuple["classes.BeliefsDataFrame", str]:
    timedelta_span = max(df["belief_horizon"]) - min(df["belief_horizon"])
    if timedelta_span >= timedelta(days=4 * 365.2425):
        df["belief_horizon"] = df["belief_horizon"].apply(
            lambda x: x.days / 365.2425
            + (x.seconds + x.microseconds / 10 ** 6) / (365.2425 * 24 * 60 * 60)
        )
        time_unit = "years"
    elif timedelta_span >= timedelta(days=4):
        df["belief_horizon"] = df["belief_horizon"].apply(
            lambda x: x.days + (x.seconds + x.microseconds / 10 ** 6) / (24 * 60 * 60)
        )
        time_unit = "days"
    elif timedelta_span >= timedelta(hours=4):
        df["belief_horizon"] = df["belief_horizon"].apply(
            lambda x: x.days * 24 + (x.seconds + x.microseconds / 10 ** 6) / (60 * 60)
        )
        time_unit = "hours"
    elif timedelta_span >= timedelta(minutes=4):
        df["belief_horizon"] = df["belief_horizon"].apply(
            lambda x: x.days * 24 * 60 + (x.seconds + x.microseconds / 10 ** 6) / 60
        )
        time_unit = "minutes"
    elif timedelta_span >= timedelta(seconds=4):
        df["belief_horizon"] = df["belief_horizon"].apply(
            lambda x: x.days * 24 * 60 * 60 + x.seconds + x.microseconds / 10 ** 6
        )
        time_unit = "seconds"
    else:
        df["belief_horizon"] = df["belief_horizon"].apply(
            lambda x: x.days * 24 * 60 * 60 * 10 ** 6 + x.microseconds
        )
        time_unit = "microseconds"
    return df, time_unit


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
            df.for_each_belief(get_nth_percentile_belief, n=(1 - ci) * 100 / 2, df=df)
            .rename(columns={"event_value": "lower_value"})
            .droplevel("cumulative_probability")
            .drop("belief_horizon", axis=1)
        )
        df_exp = (
            df.for_each_belief(get_nth_percentile_belief, n=50, df=df)
            .rename(columns={"event_value": "expected_value"})
            .droplevel("cumulative_probability")
        )
        df_ci1 = (
            df.for_each_belief(
                get_nth_percentile_belief, n=100 - (1 - ci) * 100 / 2, df=df
            )
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
    df, belief_horizon_unit = timedelta_to_human_range(df)

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
                ] = (
                    ubh
                )  # Update belief horizon to reflect propagation of beliefs over time
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
