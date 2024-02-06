from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pytz import utc
from sqlalchemy import select

import timely_beliefs.beliefs.queries as query_utils
import timely_beliefs.beliefs.utils as belief_utils
from timely_beliefs import (
    BeliefsDataFrame,
    DBBeliefSource,
    DBSensor,
    DBTimedBelief,
    TimedBelief,
)
from timely_beliefs.tests import session


@pytest.fixture(scope="function")
def beliefs_recorded_at_unique_knowledge_time(
    unique_knowledge_time_sensor: DBSensor, test_source_a: DBBeliefSource
) -> list[DBTimedBelief]:
    """Define beliefs about a future event at its unique knowledge time (e.g. a publication date)."""
    beliefs = [
        DBTimedBelief(
            source=test_source_a,
            sensor=unique_knowledge_time_sensor,
            event_value=10 + i,
            belief_time=datetime(1990, 5, 10, 0, tzinfo=utc),
            event_start=datetime(1990, 6, 1 + i, 0, tzinfo=utc),
        )
        for i in range(2)
    ]
    session.add_all(beliefs)
    return beliefs


def test_query_belief_for_sensor_with_unique_knowledge_time(
    unique_knowledge_time_sensor: DBSensor,
    beliefs_recorded_at_unique_knowledge_time: list[DBTimedBelief],
):
    """Test query of sensor with a unique knowledge time, in combination with a belief time window."""
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=unique_knowledge_time_sensor,
        beliefs_after=pd.Timestamp("1990-04-01 00:00Z"),
        beliefs_before=pd.Timestamp("1990-06-01 00:00Z"),
    ).convert_index_from_belief_time_to_horizon()
    assert belief_df.belief_horizons[0] == timedelta(0)
    assert belief_df.belief_horizons[1] == timedelta(0)
    assert belief_df.knowledge_horizons[0] == timedelta(days=22)
    assert belief_df.knowledge_horizons[1] == timedelta(days=23)


@pytest.fixture(scope="function")
def day_ahead_belief_about_ex_ante_economical_event(
    ex_ante_economics_sensor: DBSensor, test_source_a: DBBeliefSource
):
    """Define day-ahead belief about an ex-ante economical event."""
    belief = DBTimedBelief(
        source=test_source_a,
        sensor=ex_ante_economics_sensor,
        event_value=10,
        belief_time=datetime(2018, 1, 1, 10, tzinfo=utc),
        event_start=datetime(2018, 1, 2, 22, 45, tzinfo=utc),
    )
    session.add(belief)
    return belief


@pytest.fixture(scope="function")
def multiple_day_ahead_beliefs_about_ex_ante_economical_event(
    ex_ante_economics_sensor: DBSensor, test_source_a: DBBeliefSource
):
    """Define multiple day-ahead beliefs about an ex-ante economical event."""
    n = 10
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    beliefs = []
    for i in range(n):
        belief = DBTimedBelief(
            source=test_source_a,
            sensor=ex_ante_economics_sensor,
            event_value=10 + i,
            belief_time=ex_ante_economics_sensor.knowledge_time(event_start)
            - timedelta(hours=i + 1),
            event_start=event_start,
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


@pytest.fixture(scope="function")
def multiple_probabilistic_day_ahead_beliefs_about_ex_ante_economical_event(
    ex_ante_economics_sensor: DBSensor,
    ex_ante_economics_sensor_b: DBSensor,
    test_source_a: DBBeliefSource,
):
    """Define multiple probabilistic day-ahead beliefs about an ex-ante economical event on two sensors."""
    n = 10  # number of belief times
    np = 2  # number of probabilities per belief
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    beliefs = []
    for sensor in [ex_ante_economics_sensor, ex_ante_economics_sensor_b]:
        for i in range(n):
            for j in range(np):
                if sensor == ex_ante_economics_sensor and i == 0:
                    # Skip to ensure that sensor B has more recent beliefs
                    # This is to test the most_recent_beliefs_only parameter for a query on the other sensor
                    continue
                belief = DBTimedBelief(
                    source=test_source_a,
                    sensor=sensor,
                    event_value=10 + i - j / 100,
                    belief_time=ex_ante_economics_sensor.knowledge_time(event_start)
                    - timedelta(hours=i + 1),
                    event_start=event_start,
                    cumulative_probability=0.5 * (1 - j / np),
                )
                session.add(belief)
                beliefs.append(belief)
    return beliefs


@pytest.fixture(scope="function")
def multiple_day_after_beliefs_about_ex_ante_economical_event(
    ex_ante_economics_sensor: DBSensor, test_source_a: DBBeliefSource
):
    """Define multiple day-after beliefs about an ex-ante economical event."""
    n = 10
    event_start = datetime(2025, 1, 2, 23, 00, tzinfo=utc)
    beliefs = []
    for i in range(n):
        belief = DBTimedBelief(
            source=test_source_a,
            sensor=ex_ante_economics_sensor,
            event_value=10 + i,
            belief_time=ex_ante_economics_sensor.knowledge_time(event_start)
            + timedelta(hours=i + 1),
            event_start=event_start,
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


def test_query_belief_with_empty_source_list(
    ex_ante_economics_sensor: DBSensor,
    day_ahead_belief_about_ex_ante_economical_event: DBTimedBelief,
):
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        source=[],
    )
    assert belief_df.empty


@pytest.mark.parametrize(
    "beliefs_after, beliefs_before, expected_length",
    [
        (None, None, 1),  # Just one belief set up for this sensor
        (None, datetime(2017, 1, 1, 10, tzinfo=utc), 0),  # No beliefs a year earlier
        (datetime(2018, 1, 3, 10, tzinfo=utc), None, 0),  # No beliefs 2 months later
        (datetime(2018, 1, 1, 10, tzinfo=utc), None, 1),  # One belief after 10am UTC
        (None, datetime(2018, 1, 1, 9, tzinfo=utc), 0),  # No beliefs an hour earlier
        (datetime(2018, 1, 1, 13, tzinfo=utc), None, 0),  # No beliefs after 1pm UTC
    ],
)
def test_query_belief_by_belief_time(
    ex_ante_economics_sensor: DBSensor,
    day_ahead_belief_about_ex_ante_economical_event: DBTimedBelief,
    beliefs_after,
    beliefs_before,
    expected_length,
):
    bdf = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        beliefs_after=beliefs_after,
        beliefs_before=beliefs_before,
    )
    assert len(bdf) == expected_length

    if expected_length == 1:
        # By calling a pandas Series for its values we lose the timezone (a pandas bug still present in version 0.23.4)
        # This next test warns us when it has been fixed (if it fails, just replace != with ==).
        assert bdf.knowledge_times.values[0] != datetime(2018, 1, 1, 11, 0, tzinfo=utc)
        # And this test is just a workaround to test what we wanted to test.
        assert pd.Timestamp(bdf.knowledge_times.values[0]) == pd.Timestamp(
            datetime(2018, 1, 1, 11, 0)
        )


def test_query_belief_history(
    ex_ante_economics_sensor: DBSensor,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
):
    df = DBTimedBelief.search_session(session=session, sensor=ex_ante_economics_sensor)
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    df2 = df.belief_history(event_start).sort_index(
        level="belief_time", ascending=False
    )
    assert len(df2) == 10
    assert (df2["event_value"].values == np.arange(10, 20)).all()
    df3 = df.belief_history(
        event_start,
        belief_time_window=(
            datetime(2025, 1, 1, 7, tzinfo=utc),
            datetime(2025, 1, 1, 9, tzinfo=utc),
        ),
    )
    assert len(df3) == 3
    df4 = df.belief_history(
        event_start, belief_horizon_window=(timedelta(weeks=-10), timedelta(hours=2.5))
    )  # Only 2 beliefs were formed up to 2.5 hours before knowledge_time, and none after
    assert len(df4) == 2


def test_query_rolling_horizon(
    time_slot_sensor: DBSensor, rolling_day_ahead_beliefs_about_time_slot_events
):
    """Make sure that a rolling viewpoint includes the most recent beliefs."""
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(2050, 1, 1, 14, tzinfo=utc),
    )  # select beliefs up until 14 o'clock (4 events have 2 beliefs, and 1 event has 1 belief)
    rolling_df = belief_df.rolling_viewpoint(
        belief_horizon=timedelta(hours=49)
    )  # select only the five older beliefs
    assert len(rolling_df) == 5  # 5 older (made at 10,11,12,13,14 o'clock)
    assert (rolling_df["event_value"].values == np.arange(101, 106)).all()
    rolling_df = belief_df.rolling_viewpoint(
        belief_horizon=timedelta(hours=48)
    )  # select the most recent beliefs
    assert (
        len(rolling_df) == 5
    )  # 4 more recent (made at 11,12,13,14 o'clock), 1 older (at 14 o'clock, because the 15 o'clock is missing)
    assert (rolling_df["event_value"].values == [11, 12, 13, 14, 105]).all()


def test_query_fixed_horizon(
    time_slot_sensor: DBSensor,
    rolling_day_ahead_beliefs_about_time_slot_events,
    test_source_a,
    test_source_b,
):
    belief_time = datetime(2050, 1, 1, 11, tzinfo=utc)
    df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(2050, 1, 1, 15, tzinfo=utc),
        source=[test_source_a, test_source_b],
    )
    df2 = df.fixed_viewpoint(belief_time=belief_time)
    assert len(df2) == 2
    assert df2[df2.index.get_level_values("belief_time") > belief_time].empty
    assert (df2["event_value"].values == np.array([11, 102])).all()
    df3 = df.fixed_viewpoint(
        belief_time_window=(belief_time - timedelta(minutes=1), belief_time)
    )
    assert len(df3) == 2  # The belief formed at 10 AM is now considered too old
    assert (df3["event_value"].values == np.array([11, 102])).all()


def test_downsample(time_slot_sensor, rolling_day_ahead_beliefs_about_time_slot_events):
    """Downsample from 15 minutes to 2 hours."""
    new_resolution = timedelta(hours=2)
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(2100, 1, 1, 13, tzinfo=utc),
    )
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution


def test_upsample(time_slot_sensor, rolling_day_ahead_beliefs_about_time_slot_events):
    """Upsample from 15 minutes to 5 minutes."""
    new_resolution = timedelta(minutes=5)
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(2100, 1, 1, 13, tzinfo=utc),
    )
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution


def _test_empty_frame(time_slot_sensor):
    """pandas GH30517"""
    bdf = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(1900, 1, 1, 13, tzinfo=utc),
    )
    assert bdf.empty  # no data expected
    assert pd.api.types.is_datetime64_dtype(bdf.index.get_level_values("belief_time"))
    bdf = bdf.convert_index_from_belief_time_to_horizon()
    assert pd.api.types.is_timedelta64_dtype(
        bdf.index.get_level_values("belief_horizon")
    )  # dtype of belief_horizon is timedelta64[ns], so the minimum horizon on an empty BeliefsDataFrame is NaT instead of NaN


def test_search_by_sensor_id(
    ex_ante_economics_sensor: DBSensor,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
):
    """Check db query by sensor id, against query by sensor instance, for a non-empty dataset."""

    # Query all beliefs for this sensor, using sensor instance (our reference)
    df_by_instance = DBTimedBelief.search_session(
        session=session, sensor=ex_ante_economics_sensor, most_recent_beliefs_only=False
    )

    # Query all beliefs for this sensor, using sensor id (our test)
    df_by_id = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor.id,
        most_recent_beliefs_only=False,
    )
    assert not df_by_id.empty
    pd.testing.assert_frame_equal(df_by_id, df_by_instance)


def test_select_most_recent_deterministic_beliefs(
    ex_ante_economics_sensor: DBSensor,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
    multiple_day_after_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
):
    """Check db query filters for most recent beliefs, most recent events, and both at once."""

    # Query all beliefs for this sensor
    df = DBTimedBelief.search_session(
        session=session, sensor=ex_ante_economics_sensor, most_recent_beliefs_only=False
    )

    # Most recent beliefs selected after query (our reference)
    df_recent_beliefs_after_query = belief_utils.select_most_recent_belief(df)

    # Most recent beliefs selected within query (our test)
    df_recent_beliefs_within_query = DBTimedBelief.search_session(
        session=session, sensor=ex_ante_economics_sensor, most_recent_beliefs_only=True
    )
    pd.testing.assert_frame_equal(
        df_recent_beliefs_within_query, df_recent_beliefs_after_query
    )

    # Most recent events selected after query (our reference)
    df_recent_events_after_query = df[
        df.index.get_level_values("event_start") == df.event_starts.max()
    ]

    # Most recent events selected within query (our test)
    df_recent_events_within_query = DBTimedBelief.search_session(
        session=session, sensor=ex_ante_economics_sensor, most_recent_events_only=True
    )
    pd.testing.assert_frame_equal(
        df_recent_events_within_query, df_recent_events_after_query
    )

    # Most recent beliefs and most recent events selected after query (our reference)
    df_recent_both_after_query = df_recent_beliefs_after_query[
        df_recent_beliefs_after_query.index.get_level_values("event_start")
        == df_recent_beliefs_after_query.event_starts.max()
    ]

    # Most recent beliefs and most recent events selected within query (our test)
    df_recent_both_within_query = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        most_recent_events_only=True,
    )
    pd.testing.assert_frame_equal(
        df_recent_both_within_query, df_recent_both_after_query
    )


def test_select_most_recent_probabilistic_beliefs(
    ex_ante_economics_sensor: DBSensor,
    multiple_probabilistic_day_ahead_beliefs_about_ex_ante_economical_event: list[
        DBTimedBelief
    ],
):
    df = DBTimedBelief.search_session(
        session=session, sensor=ex_ante_economics_sensor, most_recent_beliefs_only=False
    )
    most_recent_df = belief_utils.select_most_recent_belief(df)
    df = DBTimedBelief.search_session(
        session=session, sensor=ex_ante_economics_sensor, most_recent_beliefs_only=True
    )
    pd.testing.assert_frame_equal(df, most_recent_df)


@pytest.mark.parametrize(
    "event_values, expected_unchanged_event_values",
    [
        ([10, 10, 10, 9, 9, 9], [None, 10, 10, None, 9, 9]),
        ([10, 11, 10, 9, 9.5, 9], [None, None, None, None, None, None]),
        ([10, 9, 10, 10, 9.5], [None, None, None, 10, None]),
        ([10, 10, 9, 10, 10], [None, 10, None, None, 10]),
    ],
)
def test_query_unchanged_beliefs(event_values, expected_unchanged_event_values):
    sensor = session.execute(select(DBSensor).limit(1)).scalar()
    source = session.execute(select(DBBeliefSource).limit(1)).scalar()
    beliefs = [
        DBTimedBelief(
            sensor=sensor,
            source=source,
            event_value=v,
            event_start=pd.Timestamp("2022-01-26 13:50+01:00").to_pydatetime(),
            belief_time=pd.Timestamp("2022-01-05 13:50+01:00").to_pydatetime()
            + i * timedelta(hours=1),
        )
        for i, v in enumerate(event_values)
    ]
    expected_unchanged_beliefs = BeliefsDataFrame(
        [
            TimedBelief(
                sensor=sensor,
                source=source,
                event_value=v,
                event_start=pd.Timestamp("2022-01-26 13:50+01:00").to_pydatetime(),
                belief_time=pd.Timestamp("2022-01-05 13:50+01:00").to_pydatetime()
                + i * timedelta(hours=1),
            )
            for i, v in enumerate(expected_unchanged_event_values)
            if v is not None
        ]
    )
    session.add_all(beliefs)
    all_beliefs_query = select(DBTimedBelief).filter(
        DBTimedBelief.sensor == sensor, DBTimedBelief.source == source
    )
    q = query_utils.query_unchanged_beliefs(
        session=session,
        query=all_beliefs_query,
    )
    unchanged_beliefs = BeliefsDataFrame(session.scalars(q).all())
    pd.testing.assert_frame_equal(unchanged_beliefs, expected_unchanged_beliefs)
