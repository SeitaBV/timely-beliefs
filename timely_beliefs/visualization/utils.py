from datetime import timedelta
from typing import List, Tuple

import altair as alt
import pandas as pd

from timely_beliefs.beliefs import classes  # noqa: F401
from timely_beliefs.beliefs.probabilistic_utils import get_nth_percentile_belief
from timely_beliefs.visualization import graphs, selectors


def plot(
    df: "classes.BeliefsDataFrame",
    show_accuracy: bool = True,
    reference_source: "classes.BeliefSource" = None,
    ci: float = 0.9,
    intuitive_forecast_horizon: bool = True,
) -> alt.LayerChart:
    """Plot the BeliefsDataFrame with the Altair visualization library.

    :param df: The BeliefsDataFrame to visualize
    :param show_accuracy: If true, show additional graphs with accuracy over time and horizon
    :param reference_source: The BeliefSource serving as a reference for accuracy calculations
    :param ci: The confidence interval to highlight in the time series graph
    :param intuitive_forecast_horizon: If true, horizons are shown with respect to event start rather than knowledge time
    :return: Altair LayerChart
    """

    # Validate input
    if show_accuracy is True and reference_source is None:
        raise ValueError("Must set reference source to calculate accuracy metrics.")

    # Set up data source
    sensor_name = df.sensor.name
    sensor_unit = df.sensor.unit if df.sensor.unit != "" else "a.u."  # arbitrary unit
    df, belief_horizon_unit = prepare_df_for_plotting(
        df,
        ci=ci,
        show_accuracy=show_accuracy,
        reference_source=reference_source,
        intuitive_forecast_horizon=intuitive_forecast_horizon,
    )
    unique_belief_horizons = df["belief_horizon"].unique()
    df = df.groupby(["event_start", "source"], group_keys=False).apply(
        lambda x: align_belief_horizons(x, unique_belief_horizons)
    )  # Propagate beliefs so that each event has the same set of unique belief horizons

    # Construct base chart
    base = graphs.base_chart(df, belief_horizon_unit)

    # Construct selectors
    time_window_selector = selectors.time_window_selector(base)
    if show_accuracy is True:
        horizon_selection_brush = selectors.horizon_selection_brush(
            init_belief_horizon=unique_belief_horizons[0]
        )
        horizon_selector = selectors.horizon_selector(
            base,
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
        filtered_base = base.transform_filter(
            selectors.horizon_hover_brush | horizon_selection_brush
        )
    else:
        filtered_base = base

    # Construct charts
    ts_chart = graphs.time_series_chart(
        filtered_base,
        show_accuracy,
        sensor_name,
        sensor_unit,
        belief_horizon_unit,
        intuitive_forecast_horizon,
        ci,
    )
    if show_accuracy is True:
        ha_chart = graphs.horizon_accuracy_chart(
            base,
            horizon_selection_brush,
            belief_horizon_unit,
            intuitive_forecast_horizon,
            unique_belief_horizons,
        )
        hd_chart = graphs.hour_date_chart(filtered_base)
        return (
            (
                (
                    (
                        (
                            time_window_selector
                            & selectors.fixed_viewpoint_selector(base, idle=True)
                            + ts_chart
                        )
                        | selectors.source_selector(df)
                    )
                    | (horizon_selector & ha_chart)
                )
                & hd_chart
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
                | selectors.source_selector(df)
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
        reference_df = df.groupby(level="event_start").apply(
            lambda x: x.set_reference_values(reference_source=reference_source)
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
