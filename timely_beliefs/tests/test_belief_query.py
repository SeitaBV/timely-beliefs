import pytest
from typing import List
from datetime import datetime, timedelta

from pytz import utc
from numpy import arange
from pandas import Timestamp

from timely_beliefs import BeliefSource, Sensor, TimedBelief
from base import session


@pytest.fixture(scope="function")
def day_ahead_belief_about_ex_post_time_slot_event(
    ex_post_time_slot_sensor: Sensor, test_source: BeliefSource
):
    """Define day-ahead belief about an ex post time slot event."""
    belief = TimedBelief(
        source=test_source,
        sensor=ex_post_time_slot_sensor,
        value=10,
        belief_time=datetime(2018, 1, 1, 10, tzinfo=utc),
        event_start=datetime(2018, 1, 2, 22, 45, tzinfo=utc),
    )
    session.add(belief)
    return belief


@pytest.fixture(scope="function")
def multiple_day_ahead_beliefs_about_ex_post_time_slot_event(
    ex_post_time_slot_sensor: Sensor, test_source: BeliefSource
):
    """Define multiple day-ahead beliefs about an ex post time slot event."""
    n = 10
    beliefs = []
    for i in range(n):
        belief = TimedBelief(
            source=test_source,
            sensor=ex_post_time_slot_sensor,
            value=10+i,
            belief_time=datetime(2025, 1, 1, 10, tzinfo=utc) - timedelta(hours=i),
            event_start=datetime(2025, 1, 2, 22, 45, tzinfo=utc),
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


@pytest.fixture(scope="function")
def rolling_day_ahead_beliefs_about_time_slot_events(
    time_slot_sensor: Sensor, test_source: BeliefSource
):
    """Define multiple day-ahead beliefs about an ex post time slot event."""
    n = 10
    beliefs = []
    for i in range(n):
        belief = TimedBelief(
            source=test_source,
            sensor=time_slot_sensor,
            value=10+i,
            belief_time=datetime(2050, 1, 1, 10, tzinfo=utc) + timedelta(hours=i),
            event_start=datetime(2050, 1, 3, 10, tzinfo=utc) + timedelta(hours=i),
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


def test_persist_belief():
    assert session.query(TimedBelief).first()


def test_query_belief_by_belief_time(ex_post_time_slot_sensor: Sensor, day_ahead_belief_about_ex_post_time_slot_event: TimedBelief):
    belief_df = TimedBelief.query(sensor=ex_post_time_slot_sensor, belief_before=datetime(2018, 1, 1, 13, tzinfo=utc))

    # By calling a pandas Series for its values we lose the timezone (a pandas bug still present in version 0.23.4)
    # This next test warns us when it has been fixed (if it fails, just replace != with ==).
    assert belief_df.knowledge_times.values[0] != datetime(2018, 1, 1, 11, 0, tzinfo=utc)
    # And this test is just a workaround to test what we wanted to test.
    assert Timestamp(belief_df.knowledge_times.values[0]) == Timestamp(datetime(2018, 1, 1, 11, 0))

    # Just one belief found
    assert len(belief_df.index) == 1

    # No beliefs a year earlier
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, belief_before=datetime(2017, 1, 1, 10, tzinfo=utc)).index) == 0

    # No beliefs 2 months later
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, belief_not_before=datetime(2018, 1, 3, 10, tzinfo=utc)).index) == 0

    # One belief after 10am UTC
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, belief_not_before=datetime(2018, 1, 1, 10, tzinfo=utc)).index) == 1

    # No beliefs an hour earlier
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, belief_before=datetime(2018, 1, 1, 9, tzinfo=utc)).index) == 0

    # No beliefs after 1pm UTC
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, belief_not_before=datetime(2018, 1, 1, 13, tzinfo=utc)).index) == 0


def test_query_belief_history(ex_post_time_slot_sensor: Sensor, multiple_day_ahead_beliefs_about_ex_post_time_slot_event: List[TimedBelief]):
    belief_df = TimedBelief.query(sensor=ex_post_time_slot_sensor).belief_history(event_start=datetime(2025, 1, 2, 22, 45, tzinfo=utc))
    assert len(belief_df) == 10
    assert (belief_df["event_value"].values == arange(10, 20)).all()


def test_query_rolling_horizon(time_slot_sensor: Sensor, rolling_day_ahead_beliefs_about_time_slot_events):
    belief_df = TimedBelief.query(sensor=time_slot_sensor, belief_before=datetime(2050, 1, 1, 15, tzinfo=utc)).rolling_horizon(belief_horizon=timedelta(days=2))
    assert len(belief_df) == 6
    assert (belief_df["event_value"].values == arange(10, 16)).all()
