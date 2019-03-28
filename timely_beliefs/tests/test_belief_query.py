import pytest
from typing import List
from datetime import datetime, timedelta

from pytz import utc
from numpy import arange #, append
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
    beliefs = []
    for i in range(n):
        belief = DBTimedBelief(
            source=test_source_a,
            sensor=ex_post_time_slot_sensor,
            value=10+i,
            belief_time=datetime(2025, 1, 1, 10, tzinfo=utc) - timedelta(hours=i),
            event_start=datetime(2025, 1, 2, 22, 45, tzinfo=utc),
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


@pytest.fixture(scope="module", params=[1, 2])
def rolling_day_ahead_beliefs_about_time_slot_events(
    request, time_slot_sensor: DBSensor
):
    """Define multiple day-ahead beliefs about an ex post time slot event."""
    n = 10  # number of events

    source = session.query(DBBeliefSource).filter(DBBeliefSource.id == request.param).one_or_none()

    beliefs = []
    for i in range(1, 11):  # ten events

        # Recent belief
        belief = DBTimedBelief(
            source=source,
            sensor=time_slot_sensor,
            value=10+i,
            belief_time=datetime(2050, 1, 1, 10, tzinfo=utc) + timedelta(hours=i),
            event_start=datetime(2050, 1, 3, 10, tzinfo=utc) + timedelta(hours=i),
        )
        session.add(belief)
        beliefs.append(belief)

        # Slightly older beliefs (belief_time an hour earlier)
        belief = DBTimedBelief(
            source=source,
            sensor=time_slot_sensor,
            value=100 + i,
            belief_time=datetime(2050, 1, 1, 10, tzinfo=utc) + timedelta(hours=i-1),
            event_start=datetime(2050, 1, 3, 10, tzinfo=utc) + timedelta(hours=i),
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


def test_query_belief_by_belief_time(ex_post_time_slot_sensor: DBSensor, day_ahead_belief_about_ex_post_time_slot_event: DBTimedBelief):
    belief_df = DBTimedBelief.query(sensor=ex_post_time_slot_sensor, belief_before=datetime(2018, 1, 1, 13, tzinfo=utc))

    # By calling a pandas Series for its values we lose the timezone (a pandas bug still present in version 0.23.4)
    # This next test warns us when it has been fixed (if it fails, just replace != with ==).
    assert belief_df.knowledge_times.values[0] != datetime(2018, 1, 1, 11, 0, tzinfo=utc)
    # And this test is just a workaround to test what we wanted to test.
    assert pd.Timestamp(belief_df.knowledge_times.values[0]) == pd.Timestamp(datetime(2018, 1, 1, 11, 0))

    # Just one belief found
    assert len(belief_df.index) == 1

    # No beliefs a year earlier
    assert len(DBTimedBelief.query(sensor=ex_post_time_slot_sensor, belief_before=datetime(2017, 1, 1, 10, tzinfo=utc)).index) == 0

    # No beliefs 2 months later
    assert len(DBTimedBelief.query(sensor=ex_post_time_slot_sensor, belief_not_before=datetime(2018, 1, 3, 10, tzinfo=utc)).index) == 0

    # One belief after 10am UTC
    assert len(DBTimedBelief.query(sensor=ex_post_time_slot_sensor, belief_not_before=datetime(2018, 1, 1, 10, tzinfo=utc)).index) == 1

    # No beliefs an hour earlier
    assert len(DBTimedBelief.query(sensor=ex_post_time_slot_sensor, belief_before=datetime(2018, 1, 1, 9, tzinfo=utc)).index) == 0

    # No beliefs after 1pm UTC
    assert len(DBTimedBelief.query(sensor=ex_post_time_slot_sensor, belief_not_before=datetime(2018, 1, 1, 13, tzinfo=utc)).index) == 0


def test_query_belief_history(ex_post_time_slot_sensor: DBSensor, multiple_day_ahead_beliefs_about_ex_post_time_slot_event: List[DBTimedBelief]):
    belief_df = DBTimedBelief.query(sensor=ex_post_time_slot_sensor).belief_history(event_start=datetime(2025, 1, 2, 22, 45, tzinfo=utc)).sort_index(level="belief_time", ascending=False)
    assert len(belief_df) == 10
    assert (belief_df["event_value"].values == arange(10, 20)).all()


def test_query_rolling_horizon(time_slot_sensor: DBSensor, rolling_day_ahead_beliefs_about_time_slot_events):
    belief_df = DBTimedBelief.query(sensor=time_slot_sensor, belief_before=datetime(2050, 1, 1, 15, tzinfo=utc))
    rolling_df = belief_df.rolling_horizon(belief_horizon=timedelta(hours=49))
    assert len(rolling_df) == 5  # 5 older (10,11,12,13,14 o'clock)
    assert (rolling_df["event_value"].values == arange(101, 106)).all()
    rolling_df = belief_df.rolling_horizon(belief_horizon=timedelta(hours=48))
    assert len(rolling_df) == 9  # 4 early (11,12,13,14 o'clock), 5 late (10,11,12,13,14 o'clock)
    assert (rolling_df["event_value"].values == [11, 101, 12, 102, 13, 103, 14, 104, 105]).all()


""" old version of above test, delete 
def test_query_rolling_horizon(time_slot_sensor: Sensor, rolling_day_ahead_beliefs_about_time_slot_events):
    belief_df = TimedBelief.query(sensor=time_slot_sensor, belief_before=datetime(2050, 1, 1, 15, tzinfo=utc)).rolling_horizon(belief_horizon=timedelta(days=2))
    assert len(belief_df) == 7
    assert (belief_df["event_value"].values == append(arange(10, 16), 106)).all()
"""


def test_downsample(time_slot_sensor):
    """Downsample from 15 minutes to 2 hours."""
    new_resolution = timedelta(hours=2)
    belief_df = DBTimedBelief.query(sensor=time_slot_sensor, belief_before=datetime(2100, 1, 1, 13, tzinfo=utc))
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution


def test_upsample(time_slot_sensor):
    """Upsample from 15 minutes to 5 minutes."""
    new_resolution = timedelta(minutes=5)
    belief_df = DBTimedBelief.query(sensor=time_slot_sensor, belief_before=datetime(2100, 1, 1, 13, tzinfo=utc))
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution
