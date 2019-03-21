import pytest
from datetime import datetime, timedelta

from pytz import utc
import pandas as pd

from timely_beliefs import BeliefsDataFrame, BeliefSource, Sensor, TimedBelief
from timely_beliefs.beliefs.utils import replace_multi_index_level


@pytest.fixture(scope="module", autouse=True)
def df_4323(time_slot_sensor: Sensor, test_source_a: BeliefSource, test_source_b: BeliefSource) -> BeliefsDataFrame:
    """Convenient BeliefsDataFrame to run tests on.
    For a single sensor, it contains 4 events, 3 beliefs, 2 sources and 3 probabilistic values.
    Note that the event resolution of the sensor is 15 minutes.
    """

    sources = [test_source_a, test_source_b]
    cps = [0.1587, 0.5, 0.8413]

    # Build up a BeliefsDataFrame with various events, beliefs, sources and probabilistic accuracy (for a single sensor)
    beliefs = [TimedBelief(
            source=sources[s],
            sensor=time_slot_sensor,
            value=1000*e+100*b+10*s+p,
            belief_time=datetime(2000, 1, 1, tzinfo=utc) + timedelta(hours=b),
            event_start=datetime(2000, 1, 3, 9, tzinfo=utc) + timedelta(hours=e),
            belief_percentile=cps[p],
        )
        for p in range(3)  # 3 cumulative probabilities
        for s in range(2)  # 2 sources
        for b in range(3)  # 3 beliefs
        for e in range(4)  # 4 events
    ]
    return BeliefsDataFrame(sensor=time_slot_sensor, beliefs=beliefs).sortlevel()


def test_replace_index_level_with_intersect(df_4323):
    """Test replacing an index level.
    First test deterministic beliefs, then probabilistic beliefs."""

    df = df_4323.xs(0.5, level="belief_percentile", drop_level=False)
    df = replace_multi_index_level(df, "event_start", pd.date_range(
        start=df.index.get_level_values(0)[0], periods=1, freq=df.sensor.event_resolution
    ), intersection=True)
    assert len(df.index) == 6  # 2 sources each having 3 deterministic beliefs
    df = replace_multi_index_level(df, "event_start", pd.Index([]), intersection=True)
    assert len(df.index) == 0

    # Todo: uncomment below to test probabilistic beliefs
    df = df_4323
    df = replace_multi_index_level(df, "event_start", pd.date_range(
        start=df.index.get_level_values(0)[0], periods=1, freq=df.sensor.event_resolution
    ), intersection=True)
    assert len(df.index) == 18  # 2 sources each having 3 probabilistic beliefs with 3 probabilistic values
    df = replace_multi_index_level(df, "event_start", pd.Index([]), intersection=True)
    assert len(df.index) == 0


def test_downsample_twice_upsample_once(df_4323):
    """Test piping of resampling methods.
    First resample to daily values, then two-daily values, then back to daily values.
    Even with all original values falling within a single day,
    the final result should have separate (and equal) values for two days."""
    df = df_4323.xs(0.5, level="belief_percentile", drop_level=False)
    df = df.resample_events(timedelta(days=1))
    assert df.event_resolution == timedelta(days=1)
    assert df.index.get_level_values(level="event_start").nunique() == 1  # All events fall within the same day
    assert pd.Timestamp(df.index.get_level_values(level="event_start").values.tolist()[0]) == pd.Timestamp(datetime(2000, 1, 3, 9))
    assert len(df.index) == 6  # 2 sources each having 3 deterministic beliefs about 1 event
    assert df["event_value"].values.tolist() == [1501+100*b+10*s for s in range(2) for b in range(3)]

    df = df.resample_events(timedelta(days=2))
    assert df.event_resolution == timedelta(days=2)
    assert df.index.get_level_values(level="event_start").nunique() == 1  # All events fall within the same 2 days
    assert len(df.index) == 6  # 2 sources each having 3 deterministic beliefs about 1 event
    assert df["event_value"].values.tolist() == [1501 + 100 * b + 10 * s for s in range(2) for b in range(3)]

    df = df.resample_events(timedelta(days=1))
    assert df.event_resolution == timedelta(days=1)
    assert df.index.get_level_values(level="event_start").nunique() == 2  # We have events for both days now
    assert len(df.index) == 12  # 2 sources each having 3 deterministic beliefs about 2 events
    assert df["event_value"].values.tolist() == [1501 + 100 * b + 10 * s + 0*e for s in range(2) for b in range(3) for e in range(2)]


def test_upsample_probabilistic(df_4323):
    """Test upsampling probabilistic beliefs."""
    df = df_4323
    df = df.resample_events(timedelta(minutes=5))
    assert df.event_resolution == timedelta(minutes=5)
    assert df.index.get_level_values(level="event_start").nunique() == 3*4  # We have 3 events per quarterhour now
    assert df["event_value"].values.tolist()[0:9] == [0, 0, 0, 1, 1, 1, 2, 2, 2]


# def test_downsample_probabilistic(df_4323):
#     """Test downsampling probabilistic beliefs."""
#     df = df_4323
#     df = df.resample_events(timedelta(hours=2))
#     assert df.event_resolution == timedelta(hours=2)
#     print(df)
#     assert 1 == 2
