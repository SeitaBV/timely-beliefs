from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta
from typing import Union

import numpy as np
import pandas as pd
import pytz
from packaging import version
from pandas.core.groupby import DataFrameGroupBy
from pandas.tseries.frequencies import to_offset

from timely_beliefs import BeliefSource, Sensor
from timely_beliefs import utils as tb_utils
from timely_beliefs.beliefs import classes
from timely_beliefs.beliefs.probabilistic_utils import (
    calculate_crps,
    get_expected_belief,
    get_median_belief,
    probabilistic_nan_mean,
)
from timely_beliefs.sources import utils as source_utils

TimedeltaLike = Union[timedelta, str, pd.Timedelta]


def select_most_recent_belief(
    df: "classes.BeliefsDataFrame",
) -> "classes.BeliefsDataFrame":
    """Drop all but most recent (non-NaN) belief."""

    if df.empty:
        return df

    # Drop NaN beliefs before selecting the most recent
    df = df.for_each_belief(
        lambda x: x.dropna() if x.isnull().all()["event_value"] else x
    )
    if df.empty:
        return df

    if "belief_horizon" in df.index.names:
        return df.groupby(level=["event_start", "source"], group_keys=False).apply(
            lambda x: x.xs(
                min(x.lineage.belief_horizons),
                level="belief_horizon",
                drop_level=False,
            )
        )
    elif "belief_time" in df.index.names:
        return df.groupby(level=["event_start", "source"], group_keys=False).apply(
            lambda x: x.xs(
                max(x.lineage.belief_times), level="belief_time", drop_level=False
            )
        )
    else:
        raise KeyError(
            "No belief_horizon or belief_time index level found in DataFrame."
        )


def upsample_event_start(
    df,
    output_resolution: timedelta,
    input_resolution: timedelta,
    fill_method: str | None = "ffill",
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
        freq=output_resolution,
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
    elif fill_method == "ffill":
        return df.reindex(mux).fillna(
            method="ffill"
        )  # ffill (formerly 'pad') is the reverse of mean downsampling
    else:
        raise NotImplementedError("Unknown upsample method.")


def respect_event_resolution(grouper: DataFrameGroupBy, resolution):
    """Resample to make sure the df slice contains events with the same frequency as the given resolution.
    The input BeliefsDataFrame (see below) should represent beliefs about sequential sub-events formed by a single source
    at a single unique belief time.
    Extra beliefs are added with nan values.

    :Example:

    >>> df = df.groupby([pd.Grouper(freq="1D", level="event_start"), "belief_time", "source"], group_keys=False).pipe(respect_event_resolution, timedelta(hours=1))

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
            lvl0 = initialize_index(
                start=bin_start,
                end=bin_end,
                resolution=resolution,
            )
            df = pd.concat(
                [
                    df,
                    tb_utils.replace_multi_index_level(
                        df_slice, level="event_start", index=lvl0, intersection=True
                    ),
                ],
                axis=0,
            )

    return df


def propagate_beliefs(
    df: "classes.BeliefsDataFrame",
) -> "classes.BeliefsDataFrame":
    """Propagate beliefs over time by filling NaN values.

    We do this by assuming beliefs propagate over time (ceteris paribus, you still believe what you believed before).
    That is, the most recent belief about an event is valid until a new belief is formed.
    If no previous belief has been formed for a certain event, the original NaN valued row will be kept.
    Requires deterministic data.

    For example:

                                                                                          temp_air  wind_speed  cloud_cover
    event_start               belief_time               source    cumulative_probability
    2022-07-01 02:00:00+02:00 2022-06-30 15:10:30+02:00 simulator 0.5                          NaN         NaN         0.94
                              2022-07-01 01:10:41+02:00 simulator 0.5                        13.61         NaN          NaN
                              2022-07-01 02:10:32+02:00 simulator 0.5                          NaN        2.88          NaN

    Becomes:

                                                                                          temp_air  wind_speed  cloud_cover
    event_start               belief_time               source    cumulative_probability
    2022-07-01 02:00:00+02:00 2022-06-30 15:10:30+02:00 simulator 0.5                          NaN         NaN         0.94
                              2022-07-01 01:10:41+02:00 simulator 0.5                        13.61         NaN         0.94
                              2022-07-01 02:10:32+02:00 simulator 0.5                        13.61        2.88         0.94
    """
    if df.lineage.probabilistic_depth != 1:
        raise NotImplementedError(
            "Propagating probabilistic beliefs is not yet implemented. Please file a GitHub issue."
        )
    return df.groupby(level=["event_start", "source"], group_keys=False).ffill()


def align_belief_times(
    slice: "classes.BeliefsDataFrame", unique_belief_times
) -> "classes.BeliefsDataFrame":
    """Align belief times such that each event has the same set of unique belief times.
    We do this by assuming beliefs propagate over time (ceteris paribus, you still believe what you believed before).
    That is, the most recent belief about an event is valid until a new belief is formed.
    If no previous belief has been formed, a row is still explicitly included with a NaN value.
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
                ] = ubt  # Update belief time to reflect propagation of beliefs over time
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
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                data,
                columns=[
                    "event_start",
                    "belief_time",
                    "source",
                    "cumulative_probability",
                    "event_value",
                ],
            ),
        ],
        axis=0,
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
    distribution: str | None = None,
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
        if input_resolution == timedelta(
            0
        ) or output_resolution % input_resolution != timedelta(0):
            raise NotImplementedError(
                "Cannot downsample from resolution %s to %s."
                % (input_resolution, output_resolution)
            )
        df = slice.groupby(
            [
                pd.Grouper(freq=output_resolution, level="event_start"),
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
                pd.Grouper(freq=output_resolution, level="event_start"),
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
    distribution: str | None = None,
    keep_only_most_recent_belief: bool = False,
) -> "classes.BeliefsDataFrame":
    """For a unique source. Also assumes belief_time is one of the index levels."""

    if input_resolution == output_resolution:
        return df

    # Determine unique set of belief times
    unique_belief_times = np.sort(
        df.reset_index()["belief_time"].unique()
    )  # Sorted from past to present

    if keep_only_most_recent_belief:
        # faster
        df = tb_utils.replace_multi_index_level(
            df, "belief_time", pd.Index([unique_belief_times[-1]] * len(df))
        )
    else:
        # slower
        # Propagate beliefs so that each event has the same set of unique belief times
        df = df.groupby(["event_start"], group_keys=False).apply(
            lambda x: align_belief_times(x, unique_belief_times)
        )

    # Resample to make sure the df slice contains events with the same frequency as the input_resolution
    # (make nan rows if you have to)
    # Todo: this is only necessary when the resampling policy for the event value needs to take into account nan values within the slice,
    #  so move it closer to the join_beliefs() method
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
    source: BeliefSource | pd.Series,
    belief_horizon: timedelta | pd.Series,
    cumulative_probability: float = 0.5,
) -> list["classes.TimedBelief"]:
    """Turn series entries into TimedBelief objects."""
    beliefs = []
    if isinstance(belief_horizon, timedelta):
        belief_horizon_series = pd.Series(
            belief_horizon, index=event_value_series.index
        )
    else:
        belief_horizon_series = belief_horizon
    if isinstance(source, BeliefSource):
        source_series = pd.Series(BeliefSource, index=event_value_series.index)
    else:
        source_series = source
    for time, value, h, s in zip(
        pd.to_datetime(event_value_series.index),
        event_value_series.values,
        belief_horizon_series.values,
        source_series.values,
    ):
        beliefs.append(
            classes.TimedBelief(
                sensor=sensor,
                source=s,
                value=value,
                event_start=time,
                belief_horizon=h,
                cumulative_probability=cumulative_probability,
            )
        )
    return beliefs


def compute_accuracy_scores(
    df: "classes.BeliefsDataFrame", lite_metrics: bool = False
) -> "classes.BeliefsDataFrame":
    """Compute the following accuracy scores:
    - mean absolute error (mae)
    - mean absolute percentage error (mape)
    - weighted absolute percentage error (wape)

    For probabilistic forecasts, the MAE is computed as the Continuous Ranked Probability Score (CRPS),
    which is a generalisation of the MAE. Metrics similar to MAPE and WAPE are obtained by dividing the CRPS over
    the reference observations or the average reference observation, respectively.
    For your convenience, hopefully, we left the column names unchanged.
    For probabilistic reference observations, the CRPS takes into account all possible outcomes.
    However, the MAPE and WAPE use the middle observation (cp=0.5) as their denominator.
    """

    # Todo: put back checks when BeliefsDataFrame validation works after an index level becomes an attribute
    # # Check input
    # if not df_forecast.lineage.unique_beliefs_per_event_per_source:
    #     raise ValueError(
    #         "BeliefsDataFrame slice with forecasts must describe a single belief per source per event."
    #     )
    # if not df_observation.lineage.unique_beliefs_per_event_per_source:
    #     raise ValueError(
    #         "BeliefsDataFrame slice with observations must describe a single belief per source per event."
    #     )

    # Calculate the continuous ranked probability score
    df_scores = df.groupby(level=["event_start", "source"], group_keys=False).apply(
        lambda x: calculate_crps(x)
    )

    # Rename to mae, and calculate mape and wape if needed
    df_scores = df_scores.rename(
        columns={"crps": "mae"}
    )  # Technically, we have yet to take the mean
    if lite_metrics is False:
        df_scores["mape"] = (df_scores["mae"] / df_scores["reference_value"]).replace(
            [np.inf, -np.inf], np.nan
        )
    df_scores = df_scores.groupby(level=["source"], group_keys=False).mean()
    if lite_metrics is False:
        df_scores["wape"] = (df_scores["mae"] / df_scores["reference_value"]).replace(
            [np.inf, -np.inf], np.nan
        )

    return df_scores


def set_reference(
    df: "classes.BeliefsDataFrame",
    reference_belief_time: datetime,
    reference_belief_horizon: timedelta,
    reference_source: BeliefSource,
    return_reference_type: str = "full",
) -> "classes.BeliefsDataFrame":
    # If applicable, decide which horizon or time provides the beliefs to serve as the reference
    if reference_belief_time is None:
        if reference_belief_horizon is None:
            reference_df = select_most_recent_belief(df)
        else:
            reference_df = select_most_recent_belief(
                df[
                    df.index.get_level_values("belief_horizon")
                    >= reference_belief_horizon
                ]
            )
    else:
        if reference_belief_horizon is None:
            df = df.convert_index_from_belief_horizon_to_time()
            reference_df = select_most_recent_belief(
                df[df.index.get_level_values("belief_time") <= reference_belief_time]
            )
        else:
            raise ValueError(
                "Cannot pass both a reference belief time and a reference belief horizon."
            )

    # If applicable, decide which source provides the beliefs that serve as the reference
    if reference_source is not None:
        reference_df = reference_df.set_event_value_from_source(reference_source)

    # Take a deterministic value of the beliefs as the reference value
    if return_reference_type == "mean":
        reference_df = reference_df.for_each_belief(get_expected_belief)
    elif return_reference_type == "median":
        reference_df = reference_df.for_each_belief(get_median_belief)
    elif return_reference_type != "full":
        raise ValueError(
            f"Unknown return_reference_type {return_reference_type}: use 'full', 'mean' or 'median'."
        )

    belief_timing_col = (
        "belief_time" if "belief_time" in reference_df.index.names else "belief_horizon"
    )
    reference_df = reference_df.droplevel(
        ["event_start", belief_timing_col, "cumulative_probability"]
        if return_reference_type != "full"
        else ["event_start", belief_timing_col]
    ).rename(columns={"event_value": "reference_value"})

    # Set the reference values for each belief
    return pd.concat(
        [reference_df] * df.lineage.number_of_belief_horizons,
        keys=df.lineage.belief_horizons,
        names=["belief_horizon", "source"]
        if return_reference_type != "full"
        else ["belief_horizon", "source", "cumulative_probability"],
    )


def read_csv(  # noqa C901
    path: str,
    sensor: "classes.Sensor",
    source: "classes.BeliefSource" = None,
    look_up_sources: list["classes.BeliefSource"] = None,
    belief_horizon: timedelta = None,
    belief_time: datetime = None,
    cumulative_probability: float = None,
    resample: bool | TimedeltaLike = False,
    timezone: str | None = None,
    filter_by_column: dict = None,
    event_ends_after: datetime = None,
    event_starts_before: datetime = None,
    datetime_column_split: str | None = None,
    transformations: list[dict] = None,
    **kwargs,
) -> "classes.BeliefsDataFrame":
    """Utility function to load a BeliefsDataFrame from a csv file or xls sheet (see example/temperature.csv).

    You still need to set the sensor and the source for the BeliefsDataFrame; the csv file only contains their names.
    In case the csv file contains multiple source names, you can pass a list of sources.
    Each source name will be replaced by the actual source.

    :param path:                    Path (or url) to the csv file.
    :param sensor:                  The sensor to which the data pertains.
    :param source:                  Optionally, set a specific source for the read-in data.
                                    If not set, the look_up_sources parameter must be used.
    :param look_up_sources:         Optionally, pass a list of sources used to look up source names from the csv file.
                                    If not set, the source parameter must be used.
    :param belief_horizon:          Optionally, set a specific belief horizon for the read-in data.
    :param belief_time:             Optionally, set a specific belief time for the read-in data.
    :param cumulative_probability:  Optionally, set a specific cumulative probability for the read-in data.
    :param resample:                Optionally, resample to the event resolution of the sensor.
                                    Set to True to infer the input resolution, or use a timedelta to set it explicitly.
                                    Only implemented for the special read case of 2-column data (see below).
    :param timezone:                Optionally, localize timezone naive datetimes to a specific timezone.
                                    Accepts IANA timezone names (e.g. UTC or Europe/Amsterdam).
                                    If not set and timezone naive datetimes are read in, the data is localized to UTC.
    :param filter_by_column:        Select a subset of rows by filtering on a specific value for a specific column.
                                    For example: {4: 1995} selects all rows where column 4 contains the value 1995.
    :param event_ends_after:        Optionally, keep only events that end after this datetime.
                                    Exclusive for non-instantaneous events, inclusive for instantaneous events.
                                    Note that the first event may transpire partially before this datetime.
    :param event_starts_before:     Optionally, keep only events that start before this datetime.
                                    Exclusive for non-instantaneous events, inclusive for instantaneous events.
                                    Note that the last event may transpire partially after this datetime.
    :param floor_event_start:       Whether to floor the event_start datetime to the sensor event_resolution.
    :param ceil_event_start:        Whether to ceil the event_start datetime to the sensor event_resolution.
    :param round_event_start:       Whether to round the event_start datetime to the sensor event_resolution.
    :param datetime_column_split:   Optionally, help parse the datetime column by splitting according to some string.
                                    For example:
                                            "1 jan 2022 00:00 - 1 jan 2022 01:00"
                                        with
                                            datetime_column_split = " - "
                                        becomes
                                            "1 jan 2022 00:00"
                                        which can then be parsed as a datetime.
    :param transformations:         Optionally, pass a transformations list to apply on the resulting BeliefsDataFrame.
                                    Each transformation defines a Pandas method along with args and kwargs.
                                    Examples:
                                        # Add 1
                                        {
                                            "func": "add",
                                            "args": [1],
                                        }
                                        # Multiply by 100, after filling NaN values with 1
                                        {
                                            "func": "multiply",
                                            "args": [100],
                                            "kwargs": {"fill_value": 1},
                                        }

    Also supports the case of a csv file with just 2 columns and 1 header row (a quite common time series format).
    In this case no special header names are required, but the first column has to contain the event starts,
    and the second column has to contain the event values.
    In case the event starts are local times (timezone naive), they must be localized to a timezone,
    by passing the relevant IANA timezone name (e.g. timezone='UTC' or timezone='Europe/Amsterdam').
    You also need to pass explicit values for the belief horizon/time and cumulative probability,
    in addition to the sensor and source.
    If needed, the time series may be resampled to the event resolution of the sensor, using resample=True.
    This is the only case that supports resampling.

    Also supports the case of a csv file with just 3 columns and 1 header row.
    In this case no special header names are required, but the first two columns have to contain the event starts
    and belief times, respectively, and the third column has to contain the event values.
    In case event starts and belief times are local times (timezone naive), they must be localized to a timezone,
    by passing the relevant IANA timezone name (e.g. timezone='UTC' or timezone='Europe/Amsterdam').

    Consult pandas documentation for which additional kwargs can be passed to pandas.read_csv or pandas.read_excel.
    Useful examples are parse_dates=True, infer_datetime_format=True (for read_csv)
    and sheet_name (sheet number or name, for read_excel).

    To write a BeliefsDataFrame to a csv file, just use the pandas way:

    >>> df.to_csv()

    """
    original_usecols = kwargs.get("usecols", []).copy()
    if filter_by_column:
        # Also read in any extra columns used to filter the read-in data
        kwargs["usecols"] += [
            col
            for col in filter_by_column.keys()
            if col not in kwargs.get("usecols", [])
        ]
    ext = find_out_extension(path)

    dayfirst = kwargs.pop("dayfirst", None)
    floor_event_start = kwargs.pop("floor_event_start", False)
    ceil_event_start = kwargs.pop("ceil_event_start", False)
    round_event_start = kwargs.pop("round_event_start", False)

    if ext.lower() == "csv":
        df = pd.read_csv(path, **kwargs)
    elif ext.lower() in ("xlsm", "xlsx", "xls"):
        df = pd.read_excel(path, **kwargs)  # requires openpyxl
    else:
        raise TypeError(
            f"Extension {ext} not recognized. Accepted file extensions are csv, xlsm, xlsx and xls."
        )

    if filter_by_column:
        # Filter the read-in data
        for col, val in filter_by_column.items():
            df = df[df[col] == val]
        # Remove the extra columns used to filter
        df = df.drop(
            columns=[
                col for col in filter_by_column.keys() if col not in original_usecols
            ]
        )

    # Preserve order of usecols
    if "usecols" in kwargs:
        df = df[[col for col in kwargs["usecols"] if col in df.columns]]

    # Special cases for simple time series
    df = interpret_special_read_cases(
        df, sensor, resample, timezone, dayfirst, split=datetime_column_split
    )

    # Exclude rows with NaN or NaT values
    keep_nan_values = kwargs.get("keep_default_na", True)
    if not keep_nan_values:
        df = df.dropna()

    if event_ends_after:
        if sensor.event_resolution == timedelta(0):
            df = df[df["event_start"] + sensor.event_resolution >= event_ends_after]
        else:
            df = df[df["event_start"] + sensor.event_resolution > event_ends_after]
    if event_starts_before:
        if sensor.event_resolution == timedelta(0):
            df = df[df["event_start"] <= event_starts_before]
        else:
            df = df[df["event_start"] < event_starts_before]

    if resample:
        resolution = resample if not isinstance(resample, bool) else None
        df = resample_events(
            df, sensor, keep_nan_values=keep_nan_values, resolution=resolution
        )

    # Apply optionally set belief timing
    if belief_horizon is not None and belief_time is not None:
        raise ValueError("Cannot set both a belief horizon and a belief time.")
    elif belief_horizon is not None:
        df["belief_horizon"] = belief_horizon
    elif belief_time is not None:
        df["belief_time"] = belief_time

    # Apply optionally set source, or look up sources
    df = fill_in_sources(df, source, look_up_sources, ext)

    # Apply optionally set cumulative probability
    if cumulative_probability is not None:
        df["cumulative_probability"] = cumulative_probability

    if ceil_event_start:
        df["event_start"] = df["event_start"].dt.ceil(sensor.event_resolution)
    elif floor_event_start:
        df["event_start"] = df["event_start"].dt.floor(sensor.event_resolution)
    elif round_event_start:
        df["event_start"] = df["event_start"].dt.round(sensor.event_resolution)

    # Construct BeliefsDataFrame
    bdf = classes.BeliefsDataFrame(df, sensor=sensor)

    # Apply transformations
    if transformations:
        bdf = apply_transformations(bdf, transformations)

    return bdf


def apply_transformations(
    bdf: classes.BeliefsDataFrame, transformations: list[dict]
) -> classes.BeliefsDataFrame:
    for transformation in transformations:
        bdf = getattr(bdf, transformation["func"])(
            *transformation.get("args", []), **transformation.get("kwargs", {})
        )
    return bdf


def find_out_extension(path: str):
    """Returns 'csv' unless another extension can be inferred from the path."""
    if isinstance(path, str):
        return path.split(".")[-1]
    elif hasattr(path, "filename"):
        # For example, supports werkzeug.datastructures.FileStorage objects
        return path.filename.split(".")[-1]
    # We'll let Pandas attempt to read the path as a CSV file
    return "csv"


def fill_in_sources(
    df: pd.DataFrame,
    source: "classes.BeliefSource" | None,
    look_up_sources: list["classes.BeliefSource"] | None,
    ext: str,
) -> pd.DataFrame:
    """Fill the 'source' column with BeliefSource objects.

    :param df:              DataFrame whose 'source' column (if there is one), only contains source names.
    :param source:          If set, the 'source' column is filled with this BeliefSource.
    :param look_up_sources: If set, the source names in the 'source' column are replaced
                            with the corresponding BeliefSource from this list.
    :param ext:             File extension as a string, used in warning or error messages.
    """
    if source is not None:
        df["source"] = source_utils.ensure_source_exists(source)
    elif "source" in df.columns:
        if look_up_sources is not None:
            source_names = df["source"].unique()
            look_up_source_names = [source.name for source in look_up_sources]
            for source_name in source_names:
                source = look_up_sources[look_up_source_names.index(source_name)]
                df["source"].replace(source_name, source, inplace=True)
        else:
            warnings.warn(
                f"Sources are stored in {ext} file by their name or id. Please specify a list of BeliefSources for looking them up."
            )
    else:
        raise Exception(f"No source specified in {ext} file, please set a source.")
    return df


def interpret_special_read_cases(
    df: pd.DataFrame,
    sensor: "classes.Sensor",
    resample: bool,
    timezone: str | None,
    dayfirst: bool | None,
    split: str | None = None,
) -> pd.DataFrame:
    """Interpret the read-in data, either as event starts and event values (2 cols),
    or as event starts, belief times and event values (3 cols).
    """
    if resample and len(df.columns) != 2:
        raise NotImplementedError("Resampling is not supported for this import case.")

    if len(df.columns) == 2:
        # datetime in 1st column and value in 2nd column
        df.columns = ["event_start", "event_value"]
        if split is not None:
            df["event_start"] = df["event_start"].str.split(split, expand=True)[0]
        if dayfirst:
            df["event_start"] = pd.to_datetime(
                df["event_start"], dayfirst=dayfirst
            ).dt.to_pydatetime()

        df["event_start"] = convert_to_timezone(
            df["event_start"],
            timezone_to_convert_to=sensor.timezone,
            timezone_to_localize_to=timezone,
        )
    elif len(df.columns) == 3:
        # datetimes in 1st and 2nd column, and value in 3rd column
        df.columns = ["event_start", "belief_time", "event_value"]
        if split is not None:
            df["event_start"] = df["event_start"].str.split(split, expand=True)[0]
        if dayfirst:
            df["event_start"] = pd.to_datetime(
                df["event_start"], dayfirst=dayfirst
            ).dt.to_pydatetime()
            df["belief_time"] = pd.to_datetime(
                df["belief_time"], dayfirst=dayfirst
            ).dt.to_pydatetime()
        df["event_start"] = convert_to_timezone(
            df["event_start"],
            timezone_to_convert_to=sensor.timezone,
            timezone_to_localize_to=timezone,
        )
        df["belief_time"] = convert_to_timezone(
            df["belief_time"],
            timezone_to_convert_to=sensor.timezone,
            timezone_to_localize_to=timezone,
        )
    return df


def resample_events(
    df: pd.DataFrame,
    sensor: "classes.Sensor",
    keep_nan_values: bool,
    resolution: TimedeltaLike | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.set_index("event_start")
    if resolution is not None:
        resolution = tb_utils.parse_timedelta_like(resolution)
        index = initialize_index(
            start=df.index[0], end=df.index[-1] + resolution, resolution=resolution
        )
        df = df.reindex(index)
    if df.index.freq is None and len(df) > 2:
        # Try to infer the event resolution from the event frequency
        df.index.freq = pd.infer_freq(df.index)
    if df.index.freq is None:
        raise NotImplementedError(
            "Resampling is not supported for data without a discernible frequency."
        )
    if df.index.freq > sensor.event_resolution:
        # Upsample by forward filling
        df.event_resolution = df.index.freq
        df = upsample_beliefs_data_frame(
            df, sensor.event_resolution, keep_nan_values=keep_nan_values
        )
    else:
        # Downsample by computing the mean event_value and max belief_time
        if "belief_time" in df.columns:
            df = df.resample(sensor.event_resolution).agg(
                {"event_value": np.mean, "belief_time": np.max}
            )
        else:
            df = df.resample(sensor.event_resolution).agg({"event_value": np.mean})
    return df.reset_index()


def convert_to_timezone(
    s: pd.Series, timezone_to_convert_to: str, timezone_to_localize_to: str | None
) -> pd.Series:
    """Convert the timezone of the series to the given timezone.

    In case the series contains naive datetimes, they are first localized using the 'timezone_to_localize_to' parameter.

    :param s:                           series with datetime representations
    :param timezone_to_convert_to:      timezone to which all datetimes are converted
    :param timezone_to_localize_to:     optional timezone for localizing timezone naive datetimes
    :raises:                            TypeError in case naive datetimes are passed without a timezone to localize to
    """
    # Convert to datetime (works for timezone naive datetimes, and timezone aware datetime with a shared offset)
    s = pd.to_datetime(s)
    if s.dtype == "object":
        # Reattempt conversion for timezone aware datetimes with a mixed offset
        s = pd.to_datetime(s, utc=True)
    if s.dt.tz is None:
        if timezone_to_localize_to is None:
            raise TypeError(
                f"The timely-beliefs package does not work with timezone-naive datetimes. Please specify a timezone to which to localize your data (e.g. the timezone of the sensor, which is '{timezone_to_convert_to}')."
            )
        elif timezone_to_localize_to != timezone_to_convert_to:
            warnings.warn(
                f"Converting the timezone of the data from {timezone_to_localize_to} to {timezone_to_convert_to}."
            )
        s = s.dt.tz_localize(timezone_to_localize_to, ambiguous="infer")
    return s.dt.tz_convert(timezone_to_convert_to)


# Supports updated function signature of pd.date_range.
# From pandas>=1.4.0, it is clear that 'closed' will be replaced by 'inclusive'.
if version.parse(pd.__version__) >= version.parse("1.4.0"):

    def initialize_index(
        start: datetime, end: datetime, resolution: timedelta, inclusive: str = "left"
    ) -> pd.DatetimeIndex:
        """Initialize DatetimeIndex for event starts."""
        return pd.date_range(
            start=start,
            end=end,
            freq=resolution,
            inclusive=inclusive,
            name="event_start",
        )

else:

    def initialize_index(
        start: datetime, end: datetime, resolution: timedelta, inclusive: str = "left"
    ) -> pd.DatetimeIndex:
        """Initialize DatetimeIndex for event starts."""
        return pd.date_range(
            start=start, end=end, freq=resolution, closed=inclusive, name="event_start"
        )


def is_pandas_structure(x):
    return isinstance(x, (pd.DataFrame, pd.Series))


def is_tb_structure(x):
    return isinstance(x, (classes.BeliefsDataFrame, classes.BeliefsSeries))


def extreme_timedeltas_not_equal(
    td_a: timedelta | pd.Timedelta,
    td_b: timedelta,
) -> bool:
    """Workaround for pd.Timedelta(...) != timedelta.max (or min)
    See pandas GH49021.
    """
    if isinstance(td_a, pd.Timedelta):
        td_a = td_a.to_pytimedelta()
    return td_a != td_b


def resample_instantaneous_events(
    df: pd.DataFrame | "classes.BeliefsDataFrame",
    resolution: timedelta,
    method: str | None = None,
    dropna: bool = True,
) -> pd.DataFrame | "classes.BeliefsDataFrame":
    """Resample data representing instantaneous events.

    Updates the event frequency of the resulting data frame, and possibly also its event resolution.
    The event resolution is only updated if the resampling method computes a characteristic of a period of events,
    like 'mean' or 'first'.

    Note that, for resolutions over 1 hour, the data frequency may not turn out to be constant per se.
    This is due to DST transitions:
    - The duration between events is typically longer for the fall DST transition.
    - The duration between events is typically shorter for the spring DST transition.
    This is done to keep the data frequency in step with midnight in the sensor's timezone.
    """

    # Default resampling method for instantaneous sensors
    if method is None:
        method = "asfreq"

    # Use event_start as the only index level
    index_names = df.index.names
    df = df.reset_index().set_index("event_start")

    # Resample the data in each unique fixed timezone offset that belongs to the given IANA timezone, then recombine
    unique_offsets = df.index.map(lambda x: x.utcoffset()).unique()
    resampled_df_offsets = []
    for offset in unique_offsets:
        df_offset = df.copy()
        # Convert all the data to given timezone offset
        df_offset.index = df.index.tz_convert(
            pytz.FixedOffset(offset.seconds // 60)
        )  # offset is max 1439 minutes, so we don't need to check offset.days
        # Resample all the data in the given timezone offset, using the given method
        resampled_df_offset = getattr(df_offset.resample(resolution), method)()
        # Convert back to the original timezone
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            resampled_df_timezone = resampled_df_offset.tz_convert(df.index.tz)
        elif isinstance(df, classes.BeliefsDataFrame):
            # As a backup, use the original timezone from the BeliefsDataFrame's sensor
            resampled_df_timezone = resampled_df_offset.tz_convert(df.sensor.timezone)
        else:
            ValueError("Missing original timezone.")
        # See which resampled rows still fall in the given offset, in this timezone
        resampled_df_timezone = resampled_df_timezone[
            resampled_df_timezone.index.map(lambda x: x.utcoffset()) == offset
        ]
        resampled_df_offsets.append(resampled_df_timezone)
    resampled_df = pd.concat(resampled_df_offsets).sort_index()

    # If possible, infer missing frequency
    if resampled_df.index.freq is None and len(resampled_df) > 2:
        resampled_df.index.freq = pd.infer_freq(resampled_df.index)

    # Restore the original index levels
    resampled_df = resampled_df.reset_index().set_index(index_names)

    if method in (
        "mean",
        "max",
        "min",
        "median",
        "count",
        "nunique",
        "first",
        "last",
        "ohlc",
        "prod",
        "size",
        "sem",
        "std",
        "sum",
        "var",
        "quantile",
    ):
        # These methods derive properties of a period of events.
        # Therefore, the event resolution is updated.
        # The methods are typically used for downsampling.
        resampled_df.event_resolution = resolution
    elif method in (
        "asfreq",
        "interpolate",
        "ffill",
        "bfill",
        "pad",
        "backfill",
        "nearest",
    ):
        # These methods derive intermediate events.
        # Therefore, the event resolution is unaffected.
        # The methods are typically used for upsampling.
        pass
    else:
        raise NotImplementedError(
            f"Please file a GitHub ticket for timely-beliefs to support the '{method}' method."
        )

    if dropna:
        return resampled_df.dropna()
    return resampled_df


def meta_repr(
    tb_structure: "classes.BeliefsDataFrame" | "classes.BeliefsSeries",
) -> str:
    """Returns a string representation of all metadata.

    For example:
    >>> from timely_beliefs.examples import get_example_df
    >>> df = get_example_df()
    >>> meta_repr(df)
    'sensor: <Sensor: weight>, event_resolution: 0:15:00'
    """
    return ", ".join(
        [
            ": ".join([attr, str(getattr(tb_structure, attr))])
            for attr in tb_structure._metadata
        ]
    )


def convert_to_instantaneous(
    df: "classes.BeliefsDataFrame",
    boundary_policy: str,
):
    """Convert non-instantaneous events to instantaneous events.

    Expects event_start as the sole index, and belief_time, source, cumulative_probability and event_value as columns.

    :param df:              frame to convert
    :param boundary_policy: 'min', 'max' or 'first'
    """
    df2 = df.copy()
    df2.index = df2.index + df.event_resolution
    df = df.reset_index().set_index(
        ["event_start", "belief_time", "source", "cumulative_probability"]
    )
    df2 = df2.reset_index().set_index(
        ["event_start", "belief_time", "source", "cumulative_probability"]
    )
    df = pd.concat([df, df2], axis=1)
    if boundary_policy == "first":
        s = df.fillna(method="bfill", axis=1).iloc[:, 0]
    else:
        s = getattr(df, boundary_policy)(axis=1).rename("event_value")
    df = s.sort_index().reset_index().set_index("event_start")
    df.event_resolution = timedelta(0)
    return df


def upsample_beliefs_data_frame(
    df: "classes.BeliefsDataFrame" | pd.DataFrame,
    event_resolution: timedelta,
    keep_nan_values: bool = False,
    boundary_policy: str = "first",
) -> "classes.BeliefsDataFrame":
    """Because simply doing df.resample().ffill() does not correctly resample the last event in the data frame.

    :param df:                  In case of a regular pd.DataFrame, make sure to set df.event_resolution before passing it to this function.
    :param event_resolution:    Resolution to upsample to.
    :param keep_nan_values:     If True, place back resampled NaN values. Drops NaN values by default.
    :param boundary_policy:     When upsampling to instantaneous events,
                                take the 'max', 'min' or 'first' value at event boundaries.
    """
    if df.empty:
        df.event_resolution = event_resolution
        return df
    if event_resolution == timedelta(0):
        return convert_to_instantaneous(
            df=df,
            boundary_policy=boundary_policy,
        )
    from_event_resolution = df.event_resolution
    if from_event_resolution == timedelta(0):
        raise NotImplementedError("Cannot upsample from zero event resolution.")
    resample_ratio = pd.to_timedelta(to_offset(from_event_resolution)) / pd.Timedelta(
        event_resolution
    )
    if keep_nan_values:
        # Back up NaN values.
        # We are flagging the positions of the NaN values in the original data with a unique number.
        # The unique number is the series' L1 norm + 1, which is guaranteed not to exist within the series.
        # For example, for x = [-2, 2, 0, 1, 0.5]  =>  y = L1 norm + 1 = 5.5 + 1 = 6.5
        # The ( + 1 ) is needed for the special case of a series with only a single non-zero value.
        # For example, for x = [0, 2, 0, 0, 0]  =>  y = L1 norm + 1 = 2 + 1 = 3
        unique_event_value_not_in_df = df["event_value"].abs().sum() + 1
        df = df.fillna(unique_event_value_not_in_df)
    if isinstance(df, classes.BeliefsDataFrame):
        start = df.event_starts[0]
        end = df.event_starts[-1] + from_event_resolution
    else:
        start = df.index[0]
        end = df.index[-1] + from_event_resolution
    new_index = initialize_index(
        start=start,
        end=end,
        resolution=event_resolution,
    )
    # Reindex to introduce NaN values, then forward fill by the number of steps
    # needed to have the new resolution cover the old resolution.
    # For example, when resampling from a resolution of 30 to 20 minutes (NB frequency is 1 hour):
    # event_start               event_value
    # 2020-03-29 10:00:00+02:00 1000.0
    # 2020-03-29 11:00:00+02:00 NaN
    # 2020-03-29 12:00:00+02:00 2000.0
    # After reindexing
    # event_start               event_value
    # 2020-03-29 10:00:00+02:00 1000.0
    # 2020-03-29 10:20:00+02:00 NaN
    # 2020-03-29 10:40:00+02:00 NaN
    # 2020-03-29 11:00:00+02:00 NaN
    # 2020-03-29 11:20:00+02:00 NaN
    # 2020-03-29 11:40:00+02:00 NaN
    # 2020-03-29 12:00:00+02:00 2000.0
    # 2020-03-29 12:20:00+02:00 NaN
    # 2020-03-29 12:40:00+02:00 NaN
    # After filling a limited number of NaN values (ceil(30/20)-1 == 1)
    # event_start               event_value
    # 2020-03-29 10:00:00+02:00 1000.0
    # 2020-03-29 10:20:00+02:00 1000.0
    # 2020-03-29 10:40:00+02:00 NaN
    # 2020-03-29 11:00:00+02:00 NaN
    # 2020-03-29 11:20:00+02:00 NaN
    # 2020-03-29 11:40:00+02:00 NaN
    # 2020-03-29 12:00:00+02:00 2000.0
    # 2020-03-29 12:20:00+02:00 2000.0
    # 2020-03-29 12:40:00+02:00 NaN
    if isinstance(df, classes.BeliefsDataFrame):
        index_levels = df.index.names
        df = df.reset_index().set_index("event_start")
    df = df.reindex(new_index)
    df = df.fillna(
        method="ffill",
        limit=math.ceil(resample_ratio) - 1 if resample_ratio > 1 else None,
    )
    df = df.dropna()
    if isinstance(df, classes.BeliefsDataFrame):
        df = df.reset_index().set_index(index_levels)
    if keep_nan_values:
        # place back original NaN values
        df = df.replace(unique_event_value_not_in_df, np.NaN)
    df.event_resolution = event_resolution
    return df
