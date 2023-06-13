import numpy as np
import pandas as pd
import pytest

from timely_beliefs import BeliefsDataFrame, Sensor
from timely_beliefs.beliefs.probabilistic_utils import get_median_belief
from timely_beliefs.beliefs.utils import (
    propagate_beliefs,
    resample_instantaneous_events,
)
from timely_beliefs.examples import get_example_df
from timely_beliefs.tests.utils import equal_lists


def test_propagate_metadata_on_empty_frame():
    """Check that calling these functions, which use groupby().apply(), on an empty frame retains the frame's metadata."""
    df = BeliefsDataFrame(sensor=Sensor("test", unit="kW"))
    df = df.for_each_belief(get_median_belief)
    assert df.sensor.name == "test"
    assert df.sensor.unit == "kW"
    df = df.groupby(level=["event_start"], group_keys=False).apply(lambda x: x.head(1))
    assert df.sensor.name == "test"
    assert df.sensor.unit == "kW"


def test_propagate_multi_sourced_deterministic_beliefs():
    # Start with a deterministic example frame (4 events, 2 sources and 2 belief times)
    df = get_example_df().for_each_belief(get_median_belief)

    # Set the four later beliefs to an unknown value
    df[
        df.index.get_level_values("belief_time")
        == pd.Timestamp("2000-01-01 01:00:00+00:00")
    ] = None
    assert df["event_value"].isnull().sum() == 8

    # Propagate the four earlier beliefs
    df = propagate_beliefs(df)
    assert df["event_value"].isnull().sum() == 0

    # After propagating, the four later beliefs should be equal to the four earlier beliefs
    pd.testing.assert_frame_equal(
        df[
            df.index.get_level_values("belief_time")
            == pd.Timestamp("2000-01-01 00:00:00+00:00")
        ].droplevel("belief_time"),
        df[
            df.index.get_level_values("belief_time")
            == pd.Timestamp("2000-01-01 01:00:00+00:00")
        ].droplevel("belief_time"),
    )


@pytest.mark.parametrize(
    ("start", "periods", "frequency", "method", "exp_event_values"),
    [
        # ----------------- downsample cases -----------------
        (
            "2022-01-01 22:00+01",
            7,
            "PT2H",
            None,
            [1, 3, 5, 7],
        ),  # Downsample from 1h to 2h frequency
        (
            "2022-01-01 23:00+01",
            8,
            "PT2H",
            None,
            [np.nan, 2, 4, 6, 8],
        ),  # Downsample from 1h to 2h frequency (no value at 10 PM)
        (
            "2022-01-01 23:00+01",
            8,
            "PT2H",
            "first",
            [1, 2, 4, 6, 8],
        ),  # Downsample from 1h to 2h frequency (includes the 'first' value between 10 PM and midnight, indexed at 10 PM)
        (
            "2022-01-01 23:00+01",
            8,
            "PT2H",
            "max",
            [1, 3, 5, 7, 8],
        ),  # Downsample from 1h to 2h frequency (computes the 'max' value for each 2h event)
        # ----------------- upsample cases -----------------
        (
            "2022-01-01 22:00+01",
            7,
            "PT30M",
            None,
            [1, np.nan, 2, np.nan, 3, np.nan, 4, np.nan, 5, np.nan, 6, np.nan, 7],
        ),  # Upsample from 1h to 30m frequency (no new values at half past the hour)
        (
            "2022-01-01 22:00+01",
            7,
            "PT30M",
            "interpolate",
            [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7],
        ),  # Upsample from 1h to 30m frequency (linear interpolation)
        (
            "2022-01-01 22:00+01",
            7,
            "PT30M",
            "ffill",
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        ),  # Upsample from 1h to 30m frequency (forward filled values)
        # ----------------- spring forward DST cases -----------------
        (
            "2022-03-27 00:00+01",
            8,
            "PT2H",
            None,
            [1, 4, 6, 8],
        ),  # Downsample across 'spring forward' (+01 to +02) DST transition (2 AM does not exist, so we get an expanded duration between events)
        (
            "2022-03-27 01:00+01",
            8,
            "PT2H",
            None,
            [np.nan, 3, 5, 7],
        ),  # Downsample across 'spring forward' (+01 to +02) DST transition (no value at midnight, and 2 AM does not exist)
        (
            "2022-03-27 01:00+01",
            7,
            "PT2H",
            "first",
            [1, 3, 5, 7],
        ),  # Downsample across 'spring forward' (+01 to +02) DST transition (includes the 'first' value between midnight and 3 AM +02, indexed at midnight)
        (
            "2022-03-27 00:00+01",
            5,
            "PT30M",
            "ffill",
            [1, 1, 2, 2, 3, 3, 4, 4, 5],
        ),  # Upsample across 'spring forward' (+01 to +02) DST transition (events will be spaced 30 minutes apart, no hour is skipped)
        (
            "2022-03-27 00:00+01",
            8,
            "PT1H30M",
            "first",
            [1, 3, 3, 5, 6, 8],
        ),  # Downsample across 'spring forward' (+01 to +02) DST transition (there are only 30 minutes between 1.30 AM +01 and 3 AM +02, so we get a contracted duration between events)
        (
            "2022-03-27 00:00+01",
            8,
            "PT3H",
            "first",
            [1, 3, 6],
        ),  # Downsample across 'spring forward' (+01 to +02) DST transition (there are only 2 hours between midnight +01 and 3 AM +02, so we get a contracted duration between events)
        (
            "2022-03-26 00:00+01",
            3 * 24,
            "PT12H",
            "first",
            [1, 13, 25, 36, 48, 60, 72],
        ),  # Downsample across 'spring forward' (+01 to +02) DST transition (there are only 11 hours between midnight +01 and noon +02, so we get a contracted duration between events)
        (
            "2022-03-26 00:00+01",
            3 * 24,
            "P1D",
            "first",
            [1, 25, 48, 72],
        ),  # Downsample across 'spring forward' (+01 to +02) DST transition (there are only 23 hours on 27 March, so we get a contracted duration between events)
        (
            "2022-03-26 01:00+01",
            23 + 23 + 23,
            "PT24H",
            None,
            [np.nan, 24, 47],
        ),  # Downsample across 'spring forward' (+01 to +02) DST transition (missing value for midnight of 26 March)
        (
            "2022-03-26 01:00+01",
            23 + 23 + 23,
            "P1D",
            None,
            [np.nan, 24, 47],
        ),  # Downsample across 'spring forward' (+01 to +02) DST transition (no difference between P1D and PT24H)
        # ----------------- Fall back DST cases -----------------
        (
            "2022-10-30 00:00+02",
            7,
            "PT2H",
            None,
            [1, 3, 4, 6],
        ),  # Downsample across 'fall back' (+02 to +01) DST transition (2 AM exists in both offsets, so we get a contracted duration between events)
        (
            "2022-10-30 01:00+02",
            7,
            "PT2H",
            None,
            [np.nan, 2, 3, 5, 7],
        ),  # Downsample across 'fall back' (+02 to +01) DST transition (no value at midnight, and 2 AM exists in both offsets)
        (
            "2022-10-30 01:00+02",
            7,
            "PT2H",
            "first",
            [1, 2, 3, 5, 7],
        ),  # Downsample across 'fall back' (+02 to +01) DST transition (includes the 'first' value between midnight and 2 AM +02, indexed at midnight)
        (
            "2022-10-30 00:00+02",
            5,
            "PT30M",
            "ffill",
            [1, 1, 2, 2, 3, 3, 4, 4, 5],
        ),  # Upsample across 'fall back' (+02 to +01) DST transition (events will be spaced 30 minutes apart, no hour is skipped)
        (
            "2022-10-30 00:00+02",
            8,
            "PT1H30M",
            "first",
            [1, 3, 5, 7, 8],
        ),  # Downsample across 'fall back' (+02 to +01) DST transition (there are 2.5 hours between 1.30 AM +02 and 3 AM +01, so we get an extended duration between events)
        (
            "2022-10-30 00:00+02",
            8,
            "PT3H",
            "first",
            [1, 5, 8],
        ),  # Downsample across 'fall back' (+02 to +01) DST transition (there are 4 hours between midnight +02 and 3 AM +01, so we get an extended duration between events)
        (
            "2022-10-29 00:00+02",
            3 * 24,
            "PT12H",
            "first",
            [1, 13, 25, 38, 50, 62],
        ),  # Downsample across 'fall back' (+02 to +01) DST transition (there are 13 hours between midnight +02 and noon +01, so we get an extended duration between events)
        (
            "2022-10-29 00:00+02",
            3 * 24,
            "P1D",
            "first",
            [1, 25, 50],
        ),  # Downsample across 'fall back' (+02 to +01) DST transition (there are 25 hours on 30 October, so we get an extended duration between events)
        (
            "2022-10-29 01:00+02",
            23 + 25 + 23,
            "PT24H",
            None,
            [np.nan, 24, 49],
        ),  # Downsample across 'fall back' (+02 to +01) DST transition (missing value for midnight of 29 October)
        (
            "2022-10-29 01:00+02",
            23 + 25 + 23,
            "P1D",
            None,
            [np.nan, 24, 49],
        ),  # Downsample across 'fall back' (+02 to +01) DST transition (no difference between P1D and PT24H)
    ],
)
def test_resample_instantaneous_events(
    start, periods, frequency, method, exp_event_values
):
    """Enumerate the events and check whether downsampling returns the expected events."""
    index = pd.date_range(
        start, periods=periods, freq="1H", name="event_start"
    ).tz_convert("Europe/Amsterdam")
    df = pd.DataFrame(list(range(1, periods + 1)), index=index, columns=["event_value"])
    print(df)
    ds_df = resample_instantaneous_events(
        df, pd.Timedelta(frequency), method, dropna=False
    )
    print(ds_df)
    assert equal_lists(ds_df.values, exp_event_values)
