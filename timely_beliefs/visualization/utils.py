from datetime import timedelta
from typing import List, Tuple

import altair as alt
import pandas as pd

from timely_beliefs.beliefs import classes  # noqa: F401
from timely_beliefs.beliefs.probabilistic_utils import get_nth_percentile_belief
from timely_beliefs.visualization import graphs, selectors


def plot(
    df: "classes.BeliefsDataFrame",
    reference_source: "classes.BeliefSource",
    ci: float = 0.9,
) -> alt.LayerChart:

    # Set up data source
    df, belief_horizon_unit = prepare_df_for_plotting(
        df, ci=ci, reference_source=reference_source
    )
    unique_belief_horizons = df["belief_horizon"].unique()
    df = df.groupby(["event_start", "source"], group_keys=False).apply(
        lambda x: align_belief_horizons(x, unique_belief_horizons)
    )  # Propagate beliefs so that each event has the same set of unique belief horizons
    source = pd.DataFrame(df)

    base = graphs.base_chart(source, belief_horizon_unit)

    # Construct selectors
    time_window_selector = selectors.time_window_selector(base)
    horizon_selector = selectors.horizon_selector(
        base, belief_horizon_unit, unique_belief_horizons
    )

    # Construct charts
    ts_chart = graphs.time_series_chart(base, belief_horizon_unit, ci)
    ha_chart = graphs.horizon_accuracy_chart(
        base, belief_horizon_unit, unique_belief_horizons
    )
    hd_chart = graphs.hour_date_chart(base)

    return (
        (time_window_selector & ts_chart) | (horizon_selector & ha_chart)
    ) & hd_chart


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
    reference_source: "classes.BeliefSource" = None,
) -> Tuple[pd.DataFrame, str]:
    """

    :param df:
    :param ci: Confidence interval (default is 95%)
    :return:
    """
    accuracy_df = df.groupby(level="event_start").apply(
        lambda x: x.accuracy(
            reference_source=reference_source, keep_reference_observation=True
        )
    )
    df["belief_horizon"] = df.knowledge_times - df.belief_times
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
        df.for_each_belief(get_nth_percentile_belief, n=100 - (1 - ci) * 100 / 2, df=df)
        .rename(columns={"event_value": "upper_value"})
        .droplevel("cumulative_probability")
        .drop("belief_horizon", axis=1)
    )
    df = pd.concat([df_ci0, df_exp, df_ci1], axis=1)
    df = pd.concat(
        [
            df.reset_index()
            .set_index(["event_start", "belief_horizon", "source"])
            .sort_index(),
            accuracy_df,
        ],
        axis=1,
        sort=True,
    )
    df = df.reset_index()
    df, belief_horizon_unit = timedelta_to_human_range(df)

    df["source"] = df["source"].apply(lambda x: x.name)
    df["event_start_copy"] = df["event_start"]

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
