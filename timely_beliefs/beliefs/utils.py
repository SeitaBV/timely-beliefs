import warnings
from datetime import datetime, timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from packaging import version
from pandas.core.groupby import DataFrameGroupBy

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
            lvl0 = initialize_index(
                start=bin_start,
                end=bin_end,
                resolution=resolution,
            )
            df = df.append(
                tb_utils.replace_multi_index_level(
                    df_slice, level="event_start", index=lvl0, intersection=True
                )
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
    return df.groupby(level=["event_start", "source"]).ffill()


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
        if output_resolution % input_resolution != timedelta(0):
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
    distribution: Optional[str] = None,
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
    source: Union[BeliefSource, pd.Series],
    belief_horizon: Union[timedelta, pd.Series],
    cumulative_probability: float = 0.5,
) -> List["classes.TimedBelief"]:
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


def read_csv(
    path: str,
    sensor: "classes.Sensor",
    source: "classes.BeliefSource" = None,
    look_up_sources: List["classes.BeliefSource"] = None,
    belief_horizon: timedelta = None,
    belief_time: datetime = None,
    cumulative_probability: float = None,
    resample: bool = False,
    timezone: Optional[str] = None,
    filter_by_column: dict = None,
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
                                    Only implemented for the special read case of 2-column data (see below).
    :param timezone:                Optionally, localize timezone naive datetimes to a specific timezone.
                                    Accepts IANA timezone names (e.g. UTC or Europe/Amsterdam).
                                    If not set and timezone naive datetimes are read in, the data is localized to UTC.
    :param filter_by_column:        Select a subset of rows by filtering on a specific value for a specific column.
                                    For example: {4: 1995} selects all rows where column 4 contains the value 1995.

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
    ext = path.split(".")[-1]
    if ext.lower() == "csv":
        df = pd.read_csv(path, **kwargs)
    elif ext.lower() in ("xlsx", "xls"):
        df = pd.read_excel(path, **kwargs)  # requires openpyxl
    else:
        raise TypeError(
            f"Extension {ext} not recognized. Accepted file extensions are csv, xlsx and xls."
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

    # Exclude rows with NaN or NaT values
    if not kwargs.get("keep_default_na", True):
        df = df.dropna()

    # Preserve order of usecols
    if "usecols" in kwargs:
        df = df[[col for col in kwargs["usecols"] if col in df.columns]]

    # Special cases for simple time series
    df = interpret_special_read_cases(df, sensor, resample, timezone)

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

    return classes.BeliefsDataFrame(df, sensor=sensor)


def fill_in_sources(
    df: pd.DataFrame,
    source: Union["classes.BeliefSource", None],
    look_up_sources: Union[List["classes.BeliefSource"], None],
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
    df: pd.DataFrame, sensor: "classes.Sensor", resample: bool, timezone: Optional[str]
) -> pd.DataFrame:
    """Interpret the read-in data, either as event starts and event values (2 cols),
    or as event starts, belief times and event values (3 cols).
    """
    if resample and len(df.columns) != 2:
        raise NotImplementedError("Resampling is not supported for this import case.")

    if len(df.columns) == 2:
        # datetime in 1st column and value in 2nd column
        df.columns = ["event_start", "event_value"]
        df["event_start"] = convert_to_timezone(
            df["event_start"],
            timezone_to_convert_to=sensor.timezone,
            timezone_to_localize_to=timezone,
        )
        if resample:
            df = (
                df.set_index("event_start")
                .resample(sensor.event_resolution)
                .mean()
                .reset_index()
            )
    elif len(df.columns) == 3:
        # datetimes in 1st and 2nd column, and value in 3rd column
        df.columns = ["event_start", "belief_time", "event_value"]
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


def convert_to_timezone(
    s: pd.Series, timezone_to_convert_to: str, timezone_to_localize_to: Optional[str]
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


def initialize_index(
    start: datetime, end: datetime, resolution: timedelta, inclusive: str = "left"
) -> pd.DatetimeIndex:
    """Initialize DatetimeIndex for event starts.

    Supports updated function signature of pd.date_range.
    From pandas>=1.4.0, it is clear that 'closed' will be replaced by 'inclusive'.
    """
    if version.parse(pd.__version__) >= version.parse("1.4.0"):
        return pd.date_range(
            start=start,
            end=end,
            freq=resolution,
            inclusive=inclusive,
            name="event_start",
        )
    else:
        return pd.date_range(
            start=start, end=end, freq=resolution, closed=inclusive, name="event_start"
        )


def is_pandas_structure(x):
    return isinstance(x, (pd.DataFrame, pd.Series))


def is_tb_structure(x):
    return isinstance(x, (classes.BeliefsDataFrame, classes.BeliefsSeries))


def extreme_timedeltas_not_equal(
    td_a: Union[timedelta, pd.Timedelta],
    td_b: timedelta,
) -> bool:
    """Workaround for pd.Timedelta(...) != timedelta.max (or min)
    See pandas GH49021.
    """
    if isinstance(td_a, pd.Timedelta):
        td_a = td_a.to_pytimedelta()
    return td_a != td_b
