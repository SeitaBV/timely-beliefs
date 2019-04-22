import pytest
from typing import List
from datetime import datetime, timedelta

from pytz import utc
import numpy as np
import pandas as pd

from timely_beliefs import DBBeliefSource, DBSensor, DBTimedBelief
from timely_beliefs.base import session


@pytest.fixture(scope="function")
def day_ahead_belief_about_ex_post_time_slot_event(
    ex_post_time_slot_sensor: DBSensor, test_source_a: DBBeliefSource
):
    """Define day-ahead belief about an ex post time slot event."""
    belief = DBTimedBelief(
        source=test_source_a,
        sensor=ex_post_time_slot_sensor,
        value=10,
        belief_time=datetime(2018, 1, 1, 10, tzinfo=utc),
        event_start=datetime(2018, 1, 2, 22, 45, tzinfo=utc),
    )
    session.add(belief)
    return belief


@pytest.fixture(scope="function")
def multiple_day_ahead_beliefs_about_ex_post_time_slot_event(
    ex_post_time_slot_sensor: DBSensor, test_source_a: DBBeliefSource
):
    """Define multiple day-ahead beliefs about an ex post time slot event."""
    n = 10
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    beliefs = []
    for i in range(n):
        belief = DBTimedBelief(
            source=test_source_a,
            sensor=ex_post_time_slot_sensor,
            value=10 + i,
            belief_time=ex_post_time_slot_sensor.knowledge_time(event_start)
            - timedelta(hours=i + 1),
            event_start=event_start,
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


@pytest.fixture(scope="function", params=[1, 2])
def rolling_day_ahead_beliefs_about_time_slot_events(
    request, time_slot_sensor: DBSensor
):
    """Define multiple day-ahead beliefs about an ex post time slot event."""
    source = (
        session.query(DBBeliefSource)
        .filter(DBBeliefSource.id == request.param)
        .one_or_none()
    )

    beliefs = []
    for i in range(1, 11):  # ten events

        # Recent belief (horizon 48 hours)
        belief = DBTimedBelief(
            sensor=time_slot_sensor,
            source=source,
            value=10 + i,
            belief_time=datetime(2050, 1, 1, 10, tzinfo=utc) + timedelta(hours=i),
            event_start=datetime(2050, 1, 3, 10, tzinfo=utc) + timedelta(hours=i),
        )
        session.add(belief)
        beliefs.append(belief)

        # Slightly older beliefs (belief_time an hour earlier)
        belief = DBTimedBelief(
            sensor=time_slot_sensor,
            source=source,
            value=100 + i,
            belief_time=datetime(2050, 1, 1, 10, tzinfo=utc) + timedelta(hours=i - 1),
            event_start=datetime(2050, 1, 3, 10, tzinfo=utc) + timedelta(hours=i),
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


def test_query_belief_by_belief_time(
    ex_post_time_slot_sensor: DBSensor,
    day_ahead_belief_about_ex_post_time_slot_event: DBTimedBelief,
):
    belief_df = DBTimedBelief.query(
        sensor=ex_post_time_slot_sensor,
        belief_before=datetime(2018, 1, 1, 13, tzinfo=utc),
    )

    # By calling a pandas Series for its values we lose the timezone (a pandas bug still present in version 0.23.4)
    # This next test warns us when it has been fixed (if it fails, just replace != with ==).
    assert belief_df.knowledge_times.values[0] != datetime(
        2018, 1, 1, 11, 0, tzinfo=utc
    )
    # And this test is just a workaround to test what we wanted to test.
    assert pd.Timestamp(belief_df.knowledge_times.values[0]) == pd.Timestamp(
        datetime(2018, 1, 1, 11, 0)
    )

    # Just one belief found
    assert len(belief_df.index) == 1

    # No beliefs a year earlier
    assert (
        len(
            DBTimedBelief.query(
                sensor=ex_post_time_slot_sensor,
                belief_before=datetime(2017, 1, 1, 10, tzinfo=utc),
            ).index
        )
        == 0
    )

    # No beliefs 2 months later
    assert (
        len(
            DBTimedBelief.query(
                sensor=ex_post_time_slot_sensor,
                belief_not_before=datetime(2018, 1, 3, 10, tzinfo=utc),
            ).index
        )
        == 0
    )

    # One belief after 10am UTC
    assert (
        len(
            DBTimedBelief.query(
                sensor=ex_post_time_slot_sensor,
                belief_not_before=datetime(2018, 1, 1, 10, tzinfo=utc),
            ).index
        )
        == 1
    )

    # No beliefs an hour earlier
    assert (
        len(
            DBTimedBelief.query(
                sensor=ex_post_time_slot_sensor,
                belief_before=datetime(2018, 1, 1, 9, tzinfo=utc),
            ).index
        )
        == 0
    )

    # No beliefs after 1pm UTC
    assert (
        len(
            DBTimedBelief.query(
                sensor=ex_post_time_slot_sensor,
                belief_not_before=datetime(2018, 1, 1, 13, tzinfo=utc),
            ).index
        )
        == 0
    )


def test_query_belief_history(
    ex_post_time_slot_sensor: DBSensor,
    multiple_day_ahead_beliefs_about_ex_post_time_slot_event: List[DBTimedBelief],
):
    df = DBTimedBelief.query(sensor=ex_post_time_slot_sensor)
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
    # belief db selects all beliefs but one recent one (made exactly at 15h)
    belief_df = DBTimedBelief.query(
        sensor=time_slot_sensor, belief_before=datetime(2050, 1, 1, 15, tzinfo=utc)
    )
    rolling_df = belief_df.rolling_viewpoint(
        belief_horizon=timedelta(hours=49)
    )  # select only the five older beliefs
    assert len(rolling_df) == 5  # 5 older (made at 10,11,12,13,14 o'clock)
    assert (rolling_df["event_value"].values == np.arange(101, 106)).all()
    rolling_df = belief_df.rolling_viewpoint(
        belief_horizon=timedelta(hours=48)
    )  # select mostly recent beliefs
    assert (
        len(rolling_df) == 5
    )  # 4 early (made at 11,12,13,14 o'clock), 1 late ( at 14 o'clock)
    assert (rolling_df["event_value"].values == [11, 12, 13, 14, 105]).all()


def test_query_fixed_horizon(
    time_slot_sensor: DBSensor, rolling_day_ahead_beliefs_about_time_slot_events
):
    belief_time = datetime(2050, 1, 1, 11, tzinfo=utc)
    df = DBTimedBelief.query(
        sensor=time_slot_sensor,
        belief_before=datetime(2050, 1, 1, 15, tzinfo=utc),
        source=[1, 2],
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
    belief_df = DBTimedBelief.query(
        sensor=time_slot_sensor, belief_before=datetime(2100, 1, 1, 13, tzinfo=utc)
    )
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution


def test_upsample(time_slot_sensor, rolling_day_ahead_beliefs_about_time_slot_events):
    """Upsample from 15 minutes to 5 minutes."""
    new_resolution = timedelta(minutes=5)
    belief_df = DBTimedBelief.query(
        sensor=time_slot_sensor, belief_before=datetime(2100, 1, 1, 13, tzinfo=utc)
    )
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution
