import math
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest
import pytz
from pytest import approx

from timely_beliefs import BeliefsDataFrame, BeliefSource, Sensor, TimedBelief
from timely_beliefs.beliefs.utils import resample_instantaneous_events
from timely_beliefs.examples.beliefs_data_frames import sixteen_probabilistic_beliefs
from timely_beliefs.utils import replace_multi_index_level


@pytest.fixture(scope="function", autouse=True)
def df_wxyz(
    test_source_a: BeliefSource, test_source_b: BeliefSource
) -> Callable[[Sensor, int, int, int, int, Optional[datetime]], BeliefsDataFrame]:
    """Convenient BeliefsDataFrame to run tests on.
    For a single sensor, it contains w events, for each of which x beliefs by y sources each (max 2),
    described by z probabilistic values (max 3).
    """

    sources = [test_source_a, test_source_b]  # expand to increase max y
    cps = [0.1587, 0.5, 0.8413]  # expand to increase max z

    def f(
        sensor: Sensor, w: int, x: int, y: int, z: int, start: Optional[datetime] = None
    ):
        if start is None:
            start = datetime(2000, 1, 3, 9, tzinfo=pytz.utc)

        # Build up a BeliefsDataFrame with various events, beliefs, sources and probabilistic accuracy (for a single sensor)
        beliefs = [
            TimedBelief(
                source=sources[s],
                sensor=sensor,
                event_value=1000 * e + 100 * b + 10 * s + p,
                belief_time=datetime(2000, 1, 1, tzinfo=pytz.utc) + timedelta(hours=b),
                event_start=start + timedelta(hours=e),
                cumulative_probability=cps[p],
            )
            for e in range(w)  # w events
            for b in range(x)  # x beliefs
            for s in range(y)  # y sources
            for p in range(z)  # z cumulative probabilities
        ]
        return BeliefsDataFrame(sensor=sensor, beliefs=beliefs)

    return f


@pytest.fixture(scope="function", autouse=True)
def df_4323(
    time_slot_sensor: Sensor,
    test_source_a: BeliefSource,
    test_source_b: BeliefSource,
    df_wxyz: Callable[
        [Sensor, int, int, int, int, Optional[datetime]], BeliefsDataFrame
    ],
) -> BeliefsDataFrame:
    """Convenient BeliefsDataFrame to run tests on.
    For a single sensor, it contains 4 events, for each of which 3 beliefs by 2 sources each, described by 3
    probabilistic values.
    Note that the event resolution of the sensor is 15 minutes.
    """
    start = pytz.timezone("utc").localize(datetime(2000, 1, 3, 9))
    return df_wxyz(time_slot_sensor, 4, 3, 2, 3, start)


@pytest.fixture(scope="function", autouse=True)
def df_4111(
    time_slot_sensor: Sensor,
    test_source_a: BeliefSource,
    test_source_b: BeliefSource,
    df_wxyz: Callable[
        [Sensor, int, int, int, int, Optional[datetime]], BeliefsDataFrame
    ],
) -> BeliefsDataFrame:
    """Convenient BeliefsDataFrame to run tests on.
    For a single sensor, it contains 4 events, for each of which 1 belief by 1 source, described by 1
    deterministic value.
    Note that the event resolution of the sensor is 15 minutes, while the frequency of the events is hourly.
    """
    start = pytz.timezone("utc").localize(datetime(2000, 1, 3, 9))
    return df_wxyz(time_slot_sensor, 4, 1, 1, 1, start)


@pytest.fixture(scope="function", autouse=True)
def df_instantaneous_8111(
    instantaneous_sensor: Sensor,
    test_source_a: BeliefSource,
    test_source_b: BeliefSource,
    df_wxyz: Callable[
        [Sensor, int, int, int, int, Optional[datetime]], BeliefsDataFrame
    ],
) -> BeliefsDataFrame:
    """Convenient BeliefsDataFrame to run tests on.
    For a single sensor, it contains 8 events, for each of which 1 beliefs by 1 source each, described by 1
    probabilistic value.
    Note that the event resolution of the sensor is 15 minutes, while the event frequency is 1 hour.
    """
    start = pytz.timezone("utc").localize(datetime(2000, 1, 3, 9))
    return df_wxyz(instantaneous_sensor, 8, 1, 1, 1, start)


def test_replace_index_level_with_intersect(df_4323):
    """Test replacing an index level.
    First test deterministic beliefs, then probabilistic beliefs."""

    df = df_4323.xs(0.5, level="cumulative_probability", drop_level=False)
    df = replace_multi_index_level(
        df,
        "event_start",
        pd.date_range(
            start=df.index.get_level_values(0)[0],
            periods=1,
            freq=df.sensor.event_resolution,
        ),
        intersection=True,
    )
    assert len(df.index) == 6  # 2 sources each having 3 deterministic beliefs
    df = replace_multi_index_level(df, "event_start", pd.Index([]), intersection=True)
    assert len(df.index) == 0

    df = df_4323
    df = replace_multi_index_level(
        df,
        "event_start",
        pd.date_range(
            start=df.index.get_level_values(0)[0],
            periods=1,
            freq=df.sensor.event_resolution,
        ),
        intersection=True,
    )
    assert (
        len(df.index) == 18
    )  # 2 sources each having 3 probabilistic beliefs with 3 probabilistic values
    df = replace_multi_index_level(df, "event_start", pd.Index([]), intersection=True)
    assert len(df.index) == 0


def test_downsample_twice_upsample_once(df_4323):
    """Test piping of resampling methods.
    First resample to daily values, then two-daily values, then back to daily values.
    Even with all original values falling within a single day,
    the final result should have separate (and equal) values for two days."""
    df = df_4323.xs(0.5, level="cumulative_probability", drop_level=False)
    df = df.resample_events(timedelta(days=1))
    assert df.event_resolution == timedelta(days=1)
    assert (
        df.index.get_level_values(level="event_start").nunique() == 1
    )  # All events fall within the same day
    assert pd.Timestamp(
        df.index.get_level_values(level="event_start").values.tolist()[0]
    ) == pd.Timestamp(datetime(2000, 1, 3, 9))
    assert (
        len(df.index) == 6
    )  # 2 sources each having 3 deterministic beliefs about 1 event
    assert df["event_value"].values.tolist() == [
        1501 + 100 * b + 10 * s for b in range(3) for s in range(2)
    ]
    assert len(df.knowledge_times.unique()) == 1
    assert df.knowledge_times.unique()[0] == pd.Timestamp(
        "2000-01-04 09:00", tzinfo=pytz.utc
    )

    df = df.resample_events(timedelta(days=2))
    assert df.event_resolution == timedelta(days=2)
    assert (
        df.index.get_level_values(level="event_start").nunique() == 1
    )  # All events fall within the same 2 days
    assert (
        len(df.index) == 6
    )  # 2 sources each having 3 deterministic beliefs about 1 event
    assert df["event_value"].values.tolist() == [
        1501 + 100 * b + 10 * s for b in range(3) for s in range(2)
    ]
    assert len(df.knowledge_times.unique()) == 1
    assert df.knowledge_times.unique()[0] == pd.Timestamp(
        "2000-01-05 09:00", tzinfo=pytz.utc
    )

    df = df.resample_events(timedelta(days=1))
    assert df.event_resolution == timedelta(days=1)
    assert (
        df.index.get_level_values(level="event_start").nunique() == 2
    )  # We have events for both days now
    assert (
        len(df.index) == 12
    )  # 2 sources each having 3 deterministic beliefs about 2 events
    assert df["event_value"].values.tolist() == [
        1501 + 100 * b + 10 * s + 0 * e
        for e in range(2)
        for b in range(3)
        for s in range(2)
    ]
    assert len(df.knowledge_times.unique()) == 2
    assert df.knowledge_times.unique()[0] == pd.Timestamp(
        "2000-01-04 09:00", tzinfo=pytz.utc
    )
    assert df.knowledge_times.unique()[1] == pd.Timestamp(
        "2000-01-05 09:00", tzinfo=pytz.utc
    )


def test_upsample_probabilistic(df_4323, test_source_a: BeliefSource):
    """Test upsampling probabilistic beliefs."""
    df = df_4323
    df = df.resample_events(timedelta(minutes=5))
    assert df.event_resolution == timedelta(minutes=5)
    assert (
        df.index.get_level_values(level="event_start").nunique() == 3 * 4
    )  # We have 3 events per quarterhour now
    assert df.xs(datetime(2000, 1, 1, tzinfo=pytz.utc), level="belief_time").xs(
        test_source_a, level="source"
    )["event_value"].values.tolist()[0:9] == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_downsample_probabilistic(df_4323, test_source_a: BeliefSource):
    """Test downsampling probabilistic beliefs."""
    df = df_4323
    df = df.resample_events(timedelta(hours=2))
    assert df.event_resolution == timedelta(hours=2)
    # Half of the events are binned together, with two 3-valued probabilistic beliefs turned into one 5-valued belief
    assert len(df) == 72 / 2 + 72 / 2 * 5 / 6
    cdf = (
        df.xs(datetime(2000, 1, 3, 10, tzinfo=pytz.utc), level="event_start")
        .xs(datetime(2000, 1, 1, tzinfo=pytz.utc), level="belief_time")
        .xs(test_source_a, level="source")
    )
    cdf_p = cdf.index.get_level_values(level="cumulative_probability")
    assert cdf_p[0] == approx(
        0.1587**2
    )  # 1 combination yields the 1st unique possible outcome
    assert cdf_p[1] - cdf_p[0] == approx(
        (0.1587 * (0.5 - 0.1587)) * 2
    )  # 2 combinations yield the 2nd outcome
    assert cdf_p[2] - cdf_p[1] == approx(
        0.1587 * (1.0 - 0.5) * 2 + (0.5 - 0.1587) ** 2
    )  # 3 for the 3rd
    assert cdf_p[3] - cdf_p[2] == approx(
        ((0.5 - 0.1587) * (1.0 - 0.5)) * 2
    )  # 2 for the 4th
    assert cdf_p[4] - cdf_p[3] == approx((1.0 - 0.5) ** 2)  # 1 for the 5th


# def test_downsample_probabilistic_with_autocorrelation(df_4323):
#     """Test downsampling probabilistic beliefs with autocorrelation (apply a copula)."""
#     df = df_4323
#     df.compute_autocorrelation()
#     df = df.resample_events(timedelta(hours=2))
#     assert df.event_resolution == timedelta(hours=2)
#     # Half of the events are binned together, with two 3-valued probabilistic beliefs turned into one 5-valued belief
#     assert len(df) == 72/2 + 72/2*5/6


def test_rolling_horizon_probabilistic(df_4323):
    """Test whether probabilistic beliefs stay probabilistic when selecting a rolling horizon."""
    df = df_4323.rolling_viewpoint(belief_horizon=timedelta(days=2))
    assert (
        len(df) == 4 * 1 * 2 * 3
    )  # 4 events, 1 belief, 2 sources and 3 probabilistic values


def test_percentages_and_accuracy_of_probabilistic_model(df_4323: BeliefsDataFrame):
    df = df_4323
    assert df.lineage.number_of_probabilistic_beliefs == 24
    assert df.lineage.percentage_of_probabilistic_beliefs == 1
    assert df.lineage.percentage_of_deterministic_beliefs == 0
    assert df.lineage.probabilistic_depth == 3
    assert df.lineage.probabilistic_depth == df.probabilistic_depth_per_belief.mean()

    df = sixteen_probabilistic_beliefs()
    assert df.lineage.number_of_probabilistic_beliefs == 16
    assert df.lineage.percentage_of_probabilistic_beliefs == 1
    assert df.lineage.percentage_of_deterministic_beliefs == 0
    assert df.lineage.probabilistic_depth == (8 * 3 + 8 * 2) / 16
    assert df.lineage.probabilistic_depth == df.probabilistic_depth_per_belief.mean()

    # Check lineage is updated even in case of an inplace operation
    df.drop(df.head(20).index, inplace=True)
    assert df.lineage.number_of_probabilistic_beliefs == 8
    assert df.lineage.percentage_of_probabilistic_beliefs == 1
    assert df.lineage.percentage_of_deterministic_beliefs == 0
    assert df.lineage.probabilistic_depth == (4 * 3 + 4 * 2) / 8
    assert df.lineage.probabilistic_depth == df.probabilistic_depth_per_belief.mean()


def test_downsample_once_upsample_once_around_dst(
    time_slot_sensor: Sensor,
    df_wxyz: Callable[
        [Sensor, int, int, int, int, Optional[datetime]], BeliefsDataFrame
    ],
):
    """Fast track resampling is enabled because the data contains 1 deterministic belief per event and a unique belief time and source."""
    downsampled_event_resolution = timedelta(hours=24)
    upsampled_event_resolution = timedelta(minutes=10)
    start = pytz.timezone("Europe/Amsterdam").localize(datetime(2020, 3, 29, 0))
    df = df_wxyz(
        time_slot_sensor, 25, 1, 1, 1, start
    )  # 1 deterministic belief per event
    df.iloc[0] = np.nan  # introduce 1 NaN value
    print(df)

    # Downsample the original frame
    df_resampled_1 = df.resample_events(downsampled_event_resolution)
    print(df_resampled_1)
    # todo: uncomment if this is ever fixed: https://github.com/pandas-dev/pandas/issues/35248
    # assert df_resampled_1.index.get_level_values("event_start")[1] == pd.Timestamp(start) + event_resolution_1
    assert len(df_resampled_1) == 2  # the data falls in 2 days
    assert df_resampled_1.sensor == df.sensor
    assert df_resampled_1.event_resolution == downsampled_event_resolution
    assert df_resampled_1["event_value"].isnull().sum() == 0

    # Upsample the original frame
    df_resampled_2 = df.resample_events(
        upsampled_event_resolution, keep_nan_values=True
    )
    pd.set_option("display.max_rows", None)
    print(df_resampled_2)
    assert len(df_resampled_2) == len(df) * math.ceil(
        df.event_resolution / df_resampled_2.event_resolution
    )
    # The number of NaN values has increased by a factor equal to the resample ratio
    assert df_resampled_2["event_value"].isnull().sum() == df[
        "event_value"
    ].isnull().sum() * math.ceil(df.event_resolution / df_resampled_2.event_resolution)

    # Upsample the downsampled frame
    df_resampled_3 = df_resampled_1.resample_events(upsampled_event_resolution)
    print(df_resampled_3)
    assert (
        len(df_resampled_3)
        == len(df_resampled_1)
        * df_resampled_1.event_resolution
        / df_resampled_3.event_resolution
        - timedelta(hours=1) / upsampled_event_resolution
    )  # 12*24*2 - 12 (for DST transition) -> if this fails, https://github.com/pandas-dev/pandas/issues/35248 might have been fixed. If so, remove the - 12 to fix the test
    assert (
        df_resampled_3.index.get_level_values("event_start")[1]
        == pd.Timestamp(start) + upsampled_event_resolution
    )
    assert df_resampled_3.sensor == df.sensor
    assert df_resampled_3.event_resolution == upsampled_event_resolution
    assert df_resampled_3["event_value"].isnull().sum() == 0


def test_resample_with_belief_horizon(df_4323: BeliefsDataFrame):
    # GH 25
    df = df_4323.convert_index_from_belief_time_to_horizon()
    assert "belief_horizon" in df.index.names
    df = df.resample_events(timedelta(hours=1))
    assert df.sensor == df_4323.sensor
    assert "belief_horizon" in df.index.names


def test_groupby_preserves_metadata(df_4323: BeliefsDataFrame):
    df = df_4323
    grouper = df.groupby("event_start")
    groups = list(grouper.__iter__())
    slice_0 = groups[0][1]
    assert slice_0.sensor == df.sensor
    df_2 = grouper.apply(lambda x: x.head(1))
    assert df_2.sensor == df.sensor


def test_downsample_instantaneous(df_instantaneous_8111):
    """Check resampling instantaneous events from hourly readings to 2-hourly readings.

    Given data for 9, 10, 11, 12, 13, 14, 15 and 16 o'clock,
    we expect to get out only data for 10, 12, 14 and 16 o'clock, with the original event resolution.

    Then try again with the resampling method 'first', which takes the first reading within 2-hour periods.
    We then expect to get out data for 8, 10, 12, 14 and 16 o'clock, with an updated event resolution.
    """
    pd.set_option("display.max_rows", None)
    print(df_instantaneous_8111)
    # Downsample the original frame
    downsampled_event_resolution = timedelta(hours=2)
    df_resampled_1 = df_instantaneous_8111.resample_events(downsampled_event_resolution)
    print(df_resampled_1)
    df_expected_1 = df_instantaneous_8111[
        df_instantaneous_8111.index.get_level_values("event_start").isin(
            [
                "2000-01-03 10:00:00+00:00",
                "2000-01-03 12:00:00+00:00",
                "2000-01-03 14:00:00+00:00",
                "2000-01-03 16:00:00+00:00",
            ]
        )
    ]
    pd.testing.assert_frame_equal(
        df_resampled_1, df_expected_1, check_dtype=False
    )  # dtype is converted to float, due to introduction of np.nan values
    # original resolution kept
    assert df_resampled_1.event_resolution == df_instantaneous_8111.event_resolution
    # frequency updated
    assert df_resampled_1.event_frequency == downsampled_event_resolution

    df_resampled_2 = resample_instantaneous_events(
        df_instantaneous_8111, downsampled_event_resolution, method="first"
    )
    print(df_resampled_2)
    df_expected_2 = df_instantaneous_8111[
        df_instantaneous_8111.index.get_level_values("event_start").isin(
            [
                "2000-01-03 9:00:00+00:00",
                "2000-01-03 10:00:00+00:00",
                "2000-01-03 12:00:00+00:00",
                "2000-01-03 14:00:00+00:00",
                "2000-01-03 16:00:00+00:00",
            ]
        )
    ]
    # replace 9 AM with 8 AM
    index_levels = df_expected_2.index.names
    df_expected_2 = df_expected_2.reset_index()
    df_expected_2.loc[
        df_expected_2["event_start"] == "2000-01-03 9:00:00+00:00", "event_start"
    ] = pd.Timestamp("2000-01-03 8:00:00+00:00")
    df_expected_2 = df_expected_2.set_index(index_levels)

    pd.testing.assert_frame_equal(df_resampled_2, df_expected_2, check_dtype=True)
    # resolution updated
    assert df_resampled_2.event_resolution == downsampled_event_resolution
    # frequency updated
    assert df_resampled_2.event_frequency == downsampled_event_resolution


def test_upsample_to_instantaneous(df_4111, test_source_a: BeliefSource):
    """Test upsampling deterministic beliefs about time slot event to instantaneous events."""
    df = df_4111
    df = df.resample_events(timedelta(minutes=0))
    assert df.event_resolution == timedelta(minutes=0)
    expected_values = [0, 0, 1000, 1000, 2000, 2000, 3000, 3000]
    expected_event_starts = [
        pd.Timestamp("2000-01-03T09:00+00"),
        pd.Timestamp("2000-01-03T09:15+00"),
        pd.Timestamp("2000-01-03T10:00+00"),
        pd.Timestamp("2000-01-03T10:15+00"),
        pd.Timestamp("2000-01-03T11:00+00"),
        pd.Timestamp("2000-01-03T11:15+00"),
        pd.Timestamp("2000-01-03T12:00+00"),
        pd.Timestamp("2000-01-03T12:15+00"),
    ]
    pd.testing.assert_index_equal(
        df.index.get_level_values(level="event_start"),
        pd.DatetimeIndex(expected_event_starts, name="event_start"),
    )
    assert df["event_value"].values.tolist() == expected_values
