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
from timely_beliefs import Sensor
from timely_beliefs import BeliefSource


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
    return df.set_index(indices)


def replace_multi_index_level(
    df: "classes.BeliefsDataFrame",
    level: str,
    index: pd.Index,
    intersection: bool = False,
) -> "classes.BeliefsDataFrame":
    """Replace one of the index levels of the multi-indexed DataFrame. Returns a new dataframe object.
    :param: df: a BeliefsDataFrame (or just a multi-indexed DataFrame).
    :param: level: the name of the index level to replace.
    :param: index: the new index.
    :param: intersection: policy for replacing the index level.
    If intersection is False then simply replace (note that the new index should have the same length as the old index).
    If intersection is True then add indices not contained in the old index and delete indices not contained in the new
    index. New rows have nan columns values and copies of the first row for other index levels (note that the resulting
    index is usually longer and contains values that were both in the old and new index, i.e. the intersection).
    """
    # Todo: check whether timezone information is copied over correctly

    if intersection is False and len(index) != len(df.index):
        raise ValueError(
            "Cannot simply replace multi-index level with an index of different length than the original. "
            "Use intersection instead?"
        )
    if index.name is None:
        index.name = level

    new_index_values = []
    new_index_names = []
    if intersection is True:
        contained_in_old = index.isin(df.index.get_level_values(level))
        new_index_not_in_old = index[~contained_in_old]
        contained_in_new = df.index.get_level_values(level).isin(index)
        for i in df.index.names:
            if i == level:  # For the index level that should be replaced
                # Copy old values that the new index contains, and add new values that the old index does not contain
                new_index_values.append(
                    df.index.get_level_values(i)[contained_in_new].append(
                        new_index_not_in_old
                    )
                )
                new_index_names.append(index.name)
            else:  # For the other index levels
                # Copy old values that the new index contains, and add the first value to the new rows
                new_row_values = pd.Index(
                    [df.index.get_level_values(i)[0]] * len(new_index_not_in_old)
                )
                new_index_values.append(
                    df.index.get_level_values(i)[contained_in_new].append(
                        new_row_values
                    )
                )
                new_index_names.append(i)
    else:
        for i in df.index.names:
            if i == level:  # For the index level that should be replaced
                # Replace with new index
                new_index_values.append(index)
                new_index_names.append(index.name)
            else:  # For the other index levels
                # Copy all old values
                new_index_values.append(df.index.get_level_values(i))
                new_index_names.append(i)

    # Construct new MultiIndex
    mux = pd.MultiIndex.from_arrays(new_index_values, names=new_index_names)

    df = df.copy(deep=True)
    # Apply new MultiIndex
    if intersection is True:
        # Reindex such that new rows get nan column values
        df = df.reindex(mux)
    else:
        # Replace the index
        df.index = mux
    return df.sort_index()


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
                replace_multi_index_level(
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


def compute_scores(
    df_forecast: "classes.BeliefsDataFrame",
    df_observation: "classes.BeliefsDataFrame",
    source_anchor: "classes.BeliefSource" = None,
) -> "classes.BeliefsDataFrame":
    """

    If df_true contains probabilistic beliefs, scores are determined with respect to the expected value (with cp=0.5).

    References
    ----------
    Hans Hersbach. Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems
        in Weather and Forecasting, Volume 15, No. 5, pages 559-570, 2000.
        https://journals.ametsoc.org/doi/pdf/10.1175/1520-0434%282000%29015%3C0559%3ADOTCRP%3E2.0.CO%3B2
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
    if source_anchor is not None:
        df_observation = df_observation.groupby(
            level=["event_start"], group_keys=False
        ).apply(lambda x: x.groupby(level=["source"]).pipe(set_truth, source_anchor))

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

    # Rename to mae, calculate mape and wape, and drop the true values (only keep the scores)
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
    df_scores = df_scores.drop(columns=["observation"])

    return df_scores
