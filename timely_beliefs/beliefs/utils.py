from typing import List, Optional
from datetime import timedelta

import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
from pandas.core.groupby import DataFrameGroupBy

from timely_beliefs.beliefs import classes
from timely_beliefs.beliefs.probabilistic_utils import (
    calculate_crps,
    probabilistic_nan_mean,
    set_truth,
)
from timely_beliefs import BeliefSource, Sensor
from timely_beliefs import utils as tb_utils


def select_most_recent_belief(
    df: "classes.BeliefsDataFrame"
) -> "classes.BeliefsDataFrame":

    # Remember original index levels
    indices = df.index.names

    # Convert index levels to columns
    df = df.reset_index()

    # Drop all but most recent belief
    if "belief_horizon" in indices:
        df = (
            df.sort_values(by=["belief_horizon"], ascending=True)
            .drop_duplicates(
                subset=["event_start", "source", "cumulative_probability"], keep="first"
            )
            .sort_values(by=["event_start"])
        )
    elif "belief_time" in indices:
        df = (
            df.sort_values(by=["belief_time"], ascending=True)
            .drop_duplicates(
                subset=["event_start", "source", "cumulative_probability"], keep="last"
            )
            .sort_values(by=["event_start"])
        )
    else:
        raise KeyError(
            "No belief_horizon or belief_time index level found in DataFrame."
        )

    # Convert columns to index levels (only columns that represent index levels)
    return df.set_index(indices).sort_index()


def upsample_event_start(
    df,
    output_resolution: timedelta,
    input_resolution: timedelta,
    fill_method: Optional[str] = "pad",
):
    """Upsample event_start (which must be the first index level) from input_resolution to output_resolution."""

    # Check input
    if output_resolution > input_resolution:
        raise ValueError(
            "Cannot use an upsampling policy to downsample from %s to %s."
            % (input_resolution, output_resolution)
        )
    if df.empty:
        return df
    if df.index.names[0] != "event_start":
        raise KeyError(
            "Upsampling only works if the event_start is the first index level."
        )

    lvl0 = pd.date_range(
        start=df.index.get_level_values(0)[0],
        periods=input_resolution // output_resolution,
        freq=to_offset(output_resolution).freqstr,
    )
    new_index_values = [lvl0]
    if df.index.nlevels > 0:
        new_index_values.extend(
            [[df.index.get_level_values(i)[0]] for i in range(1, df.index.nlevels)]
        )
    mux = pd.MultiIndex.from_product(new_index_values, names=df.index.names)

    # Todo: allow customisation for de-aggregating event values
    if fill_method is None:
        return df.reindex(mux)
    elif fill_method == "pad":
        return df.reindex(mux).fillna(
            method="pad"
        )  # pad is the reverse of mean downsampling
    else:
        raise NotImplementedError("Unknown upsample method.")


def respect_event_resolution(grouper: DataFrameGroupBy, resolution):
    """Resample to make sure the df slice contains events with the same frequency as the given resolution.
    The input BeliefsDataFrame (see below) should represent beliefs about sequential sub-events formed by a single source
    at a single unique belief time.
    Extra beliefs are added with nan values.

    :Example:

    >>> df = df.groupby([pd.Grouper(freq="1D", level="event_start"), "belief_time", "source"]).pipe(respect_event_resolution, timedelta(hours=1))

    So don't pass a BeliefsDataFrame directly, but pipe it so that we receive a DataFrameGroupBy object, which we can
    iterate over to obtain a BeliefsDataFrame slice for a unique belief time, source and (in our example) day of
    events. We then make sure an event is stated explicitly for (in our example) each hour.
    """

    # We need to loop over each belief time in this slice, and reindex such that each subslice has rows for each event. Then recombine.

    # Get a list of n groups, one group for each belief_time with info about how we sliced and the actual slice
    groups = list(grouper.__iter__())

    # Describe the event_start bin for the slices (we take the first, because the slices share the same event_start bin)
    bin_size = grouper.keys[0].freq
    bin_start = groups[0][0][0]
    bin_end = bin_start + bin_size

    # Build up our new BeliefsDataFrame (by copying over and emptying the rows, the metadata should be copied over)
    df = groups[0][1].copy().iloc[0:0]
    for (
        group
    ) in (
        groups
    ):  # Loop over the groups (we grouped by unique belief time and unique source)

        # Get the BeliefsDataFrame for a unique belief time and source
        df_slice = group[1]
        if not df_slice.empty:
            lvl0 = pd.date_range(
                start=bin_start,
                end=bin_end,
                freq=to_offset(resolution).freqstr,
                closed="left",
                name="event_start",
            )
            df = df.append(
                tb_utils.replace_multi_index_level(
                    df_slice, level="event_start", index=lvl0, intersection=True
                )
            )

    return df


def align_belief_times(
    slice: "classes.BeliefsDataFrame", unique_belief_times
) -> "classes.BeliefsDataFrame":
    """Align belief times such that each event has the same set of unique belief times. We do this by assuming beliefs
    propagate over time (ceteris paribus, you still believe what you believed before).
    The most recent belief about an event is valid until a new belief is formed.
    If no previous belief has been formed, a row is still explicitly included with a nan value.
    The input BeliefsDataFrame should represent beliefs about a single event formed by a single source.
    """

    # Check input
    if not slice.lineage.number_of_events == 1:
        raise ValueError("BeliefsDataFrame slice must describe a single event.")
    if not slice.lineage.number_of_sources == 1:
        raise ValueError(
            "BeliefsDataFrame slice must describe beliefs by a single source"
        )

    # Get unique source for this slice
    assert slice.lineage.number_of_sources == 1
    source = slice.lineage.sources[0]

    # Get unique event start for this slice
    event_start = slice.index.get_level_values(level="event_start").unique()
    assert len(event_start) == 1
    event_start = event_start[0]

    # Build up input data for new BeliefsDataFrame
    data = []
    previous_slice_with_existing_belief_time = None
    for ubt in unique_belief_times:

        # Check if the unique belief time (ubt) is already in the DataFrame
        if ubt not in slice.index.get_level_values("belief_time"):

            # If not already present, create a new row with the most recent belief (or nan if no previous exists)
            if previous_slice_with_existing_belief_time is not None:
                ps = previous_slice_with_existing_belief_time.reset_index()
                ps[
                    "belief_time"
                ] = (
                    ubt
                )  # Update belief time to reflect propagation of beliefs over time
                data.extend(ps.values.tolist())
            else:
                data.append([event_start, ubt, source, np.nan, np.nan])
        else:
            # If already present, copy the row (may be multiple rows in case of a probabilistic belief)
            slice_with_existing_belief_time = slice.xs(
                ubt, level="belief_time", drop_level=False
            )
            data.extend(slice_with_existing_belief_time.reset_index().values.tolist())
            previous_slice_with_existing_belief_time = slice_with_existing_belief_time

    # Create new BeliefsDataFrame
    df = slice.copy().reset_index().iloc[0:0]
    sensor = df.sensor
    df = df.append(
        pd.DataFrame(
            data,
            columns=[
                "event_start",
                "belief_time",
                "source",
                "cumulative_probability",
                "event_value",
            ],
        )
    )
    df.sensor = sensor
    df = df.set_index(
        ["event_start", "belief_time", "source", "cumulative_probability"]
    )

    return df


def join_beliefs(
    slice: "classes.BeliefsDataFrame",
    output_resolution: timedelta,
    input_resolution: timedelta,
    distribution: Optional[str] = None,
) -> "classes.BeliefsDataFrame":
    """
    Determine the joint belief about the time-aggregated event.
    The input BeliefsDataFrame slice should represent beliefs about sequential sub-events formed by a single source
    at a single unique belief time (in case of downsampling), or about a single super-event (in case of upsampling).
    The slice may contain deterministic beliefs (which require a single row) and probabilistic beliefs (which require
    multiple rows).
    """

    # Check input
    if not slice.lineage.number_of_belief_times == 1:
        raise ValueError(
            "BeliefsDataFrame slice must describe beliefs formed at the exact same time."
        )
    if not slice.lineage.number_of_sources == 1:
        raise ValueError(
            "BeliefsDataFrame slice must describe beliefs by a single source"
        )

    if output_resolution > input_resolution:

        # Create new BeliefsDataFrame with downsampled event_start
        df = slice.groupby(
            [
                pd.Grouper(
                    freq=to_offset(output_resolution).freqstr, level="event_start"
                ),
                "belief_time",
                "source",
            ],
            group_keys=False,
        ).apply(
            lambda x: probabilistic_nan_mean(
                x, output_resolution, input_resolution, distribution=distribution
            )
        )  # Todo: allow customisation for aggregating event values
    else:
        # Create new BeliefsDataFrame with upsampled event_start
        if input_resolution % output_resolution != timedelta():
            raise NotImplementedError(
                "Cannot upsample from resolution %s to %s."
                % (input_resolution, output_resolution)
            )
        df = slice.groupby(
            [
                pd.Grouper(
                    freq=to_offset(output_resolution).freqstr, level="event_start"
                ),
                "belief_time",
                "source",
                "cumulative_probability",
            ],
            group_keys=False,
        ).apply(lambda x: upsample_event_start(x, output_resolution, input_resolution))
    return df


def resample_event_start(
    df: "classes.BeliefsDataFrame",
    output_resolution: timedelta,
    input_resolution: timedelta,
    distribution: Optional[str] = None,
) -> "classes.BeliefsDataFrame":
    """For a unique source."""

    # Determine unique set of belief times
    unique_belief_times = np.sort(
        df.reset_index()["belief_time"].unique()
    )  # Sorted from past to present

    # Propagate beliefs so that each event has the same set of unique belief times
    df = df.groupby(["event_start"], group_keys=False).apply(
        lambda x: align_belief_times(x, unique_belief_times)
    )

    # Resample to make sure the df slice contains events with the same frequency as the input_resolution
    # (make nan rows if you have to)
    # Todo: this is only necessary when the resampling policy for the event value needs to take into account nan values within the slice, so move it closer to the join_beliefs() method
    # df = df.groupby(
    #     [pd.Grouper(freq=to_offset(output_resolution).freqstr, level="event_start"), "belief_time", "source"]
    # ).pipe(respect_event_resolution, input_resolution)

    # For each unique belief time, determine the joint belief about the time-aggregated event
    df = df.groupby(["belief_time"], group_keys=False).apply(
        lambda x: join_beliefs(
            x, output_resolution, input_resolution, distribution=distribution
        )
    )

    return df


def load_time_series(
    event_value_series: pd.Series,
    sensor: Sensor,
    source: BeliefSource,
    belief_horizon: timedelta,
) -> List["classes.TimedBelief"]:
    """Turn series entries into TimedBelief objects.
       TODO: enable to add probability data."""
    beliefs = []
    for time, value in event_value_series.iteritems():
        beliefs.append(
            classes.TimedBelief(
                sensor=sensor,
                source=source,
                value=value,
                event_time=time,
                belief_horizon=belief_horizon,
            )
        )
    return beliefs


def compute_accuracy_scores(
    df_forecast: "classes.BeliefsDataFrame",
    df_observation: "classes.BeliefsDataFrame",
    reference_source: "classes.BeliefSource" = None,
    keep_reference_observation: bool = False,
) -> "classes.BeliefsDataFrame":
    """ Compute the following accuracy scores:
    - mean absolute error (mae)
    - mean absolute percentage error (mape)
    - weighted absolute percentage error (wape)

    For probabilistic forecasts, the MAE is computed as the Continuous Ranked Probability Score (CRPS),
    which is a generalisation of the MAE. Metrics similar to MAPE and WAPE are obtained by dividing the CRPS over
    the reference observations or the average reference observation, respectively.
    For your convenience, hopefully, we left the column names unchanged.
    For probabilistic reference observations, the CRPS takes into account all possible outcomes.
    However, the MAPE and WAPE use the expected observation (cp=0.5) as their denominator.

    :param keep_reference_observation: Set to True to return the reference observation used to calculate mape and wape as a DataFrame column
    """

    # Check input
    if not df_forecast.lineage.unique_beliefs_per_event_per_source:
        raise ValueError(
            "BeliefsDataFrame slice with forecasts must describe a single belief per source per event."
        )
    if not df_observation.lineage.unique_beliefs_per_event_per_source:
        raise ValueError(
            "BeliefsDataFrame slice with observations must describe a single belief per source per event."
        )

    # If applicable, decide which source provides the observations that are considered to be true when we compute scores
    if reference_source is not None:
        df_observation = df_observation.groupby(
            level=["event_start"], group_keys=False
        ).apply(lambda x: x.groupby(level=["source"]).pipe(set_truth, reference_source))

    # Combine the forecasts and observations into one DataFrame
    df_forecast.index = df_forecast.index.droplevel(
        "belief_horizon"
        if "belief_horizon" in df_forecast.index.names
        else "belief_time"
    )
    df_observation.index = df_observation.index.droplevel(
        "belief_horizon"
        if "belief_horizon" in df_observation.index.names
        else "belief_time"
    )
    df_forecast.columns = ["forecast"]
    df_observation.columns = ["observation"]
    df = pd.concat([df_forecast, df_observation], axis=1)

    # Calculate the continuous ranked probability score
    df_scores = df.groupby(level=["event_start", "source"], group_keys=False).apply(
        lambda x: calculate_crps(x)
    )

    # Rename to mae, calculate mape and wape
    df_scores = df_scores.rename(
        columns={"crps": "mae"}
    )  # Technically, we have yet to take the mean
    df_scores["mape"] = (df_scores["mae"] / df_scores["observation"]).replace(
        [np.inf, -np.inf], np.nan
    )
    df_scores = df_scores.groupby(level=["source"], group_keys=False).mean()
    df_scores["wape"] = (df_scores["mae"] / df_scores["observation"]).replace(
        [np.inf, -np.inf], np.nan
    )

    # Drop the reference observations by default (only keep the scores)
    if keep_reference_observation is False:
        df_scores = df_scores.drop(columns=["observation"])
    else:
        df_scores = df_scores.rename(columns={"observation": "reference_value"})

    return df_scores
