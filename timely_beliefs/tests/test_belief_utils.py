import pandas as pd
import pytest

from timely_beliefs.beliefs.probabilistic_utils import get_median_belief
from timely_beliefs.beliefs.utils import downsample_first, propagate_beliefs
from timely_beliefs.examples import get_example_df
from timely_beliefs.tests.utils import equal_lists


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
    ("start", "periods", "resolution", "exp_event_values"),
    [
        (
            "2022-03-27 01:00+01",
            7,
            "PT2H",
            [2, 3, 5, 7],
        ),  # DST transition from +01 to +02 (spring forward, contracted event)
        (
            "2022-10-30 01:00+02",
            7,
            "PT2H",
            [2, 5, 7],
        ),  # DST transition from +02 to +01 (fall back -> extended event)
        (
            "2022-03-26 01:00+01",
            23 + 23 + 23,
            "PT24H",
            [24, 47],
        ),  # midnight of 1 full (contracted) day, plus the following midnight of 1 partial day
        (
            "2022-03-26 01:00+01",
            23 + 23 + 23,
            "P1D",
            [24, 47],
        ),  # midnight of 1 full (contracted) day, plus the following midnight of 1 partial day
        (
            "2022-10-29 01:00+02",
            23 + 25 + 23,
            "PT24H",
            [24, 49],
        ),  # midnight of 1 full (extended) day, plus the following midnight of 1 partial day
        (
            "2022-10-29 01:00+02",
            23 + 25 + 24 + 23,
            "P1D",
            [24, 49, 73],
        ),  # midnight of 1 full (extended) day and 1 full (regular) day, plus the following midnight of 1 partial day
    ],
)
def test_downsample_first(start, periods, resolution, exp_event_values):
    """Enumerate the events and check whether downsampling returns the expected events."""
    index = pd.date_range(start, periods=periods, freq="1H").tz_convert(
        "Europe/Amsterdam"
    )
    df = pd.DataFrame(list(range(1, periods + 1)), index=index)
    ds_df = downsample_first(df, pd.Timedelta(resolution))
    print(ds_df)
    assert equal_lists(ds_df.values, exp_event_values)
