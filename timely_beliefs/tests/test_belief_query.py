import pytest
from typing import List
from datetime import datetime, timedelta

from pytz import utc
from numpy import arange, append
import pandas as pd

from timely_beliefs import BeliefSource, Sensor, TimedBelief
from base import session


@pytest.fixture(scope="function")
def day_ahead_belief_about_ex_post_time_slot_event(
    ex_post_time_slot_sensor: Sensor, test_source_a
):
    """Define day-ahead belief about an ex post time slot event."""
    belief = TimedBelief(
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
    ex_post_time_slot_sensor: Sensor, test_source_a
):
    """Define multiple day-ahead beliefs about an ex post time slot event."""
    n = 10
    beliefs = []
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    for i in range(n):
        belief = TimedBelief(
            source=test_source_a,
            sensor=ex_post_time_slot_sensor,
            value=10+i,
            belief_time=ex_post_time_slot_sensor.knowledge_time(event_start) - timedelta(hours=i+1),
            event_start=event_start,
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


@pytest.fixture(scope="module", params=[1, 2])
def rolling_day_ahead_beliefs_about_time_slot_events(
    request, time_slot_sensor: Sensor
):
    """Define multiple day-ahead beliefs about an ex post time slot event."""
    n = 10  # number of events

    source = session.query(BeliefSource).filter(BeliefSource.id == request.param).one_or_none()

    beliefs = []
    for i in range(n):

        # Older belief
        belief = TimedBelief(
            source=source,
            sensor=time_slot_sensor,
            value=10+i,
            belief_time=datetime(2050, 1, 1, 10, tzinfo=utc) + timedelta(hours=i),
            event_start=datetime(2050, 1, 3, 10, tzinfo=utc) + timedelta(hours=i),
        )
        session.add(belief)
        beliefs.append(belief)

        # More recent belief
        belief = TimedBelief(
            source=source,
            sensor=time_slot_sensor,
            value=100 + i,
            belief_time=datetime(2050, 1, 1, 10, tzinfo=utc) + timedelta(hours=i-1),
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
    assert pd.Timestamp(belief_df.knowledge_times.values[0]) == pd.Timestamp(datetime(2018, 1, 1, 11, 0))

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
    df = TimedBelief.query(sensor=ex_post_time_slot_sensor)
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    df2 = df.belief_history(event_start).sort_index(level="belief_time", ascending=False)
    assert len(df2) == 10
    assert (df2["event_value"].values == arange(10, 20)).all()
    df3 = df.belief_history(event_start, belief_time_window=(datetime(2025, 1, 1, 7, tzinfo=utc), datetime(2025, 1, 1, 9, tzinfo=utc)))
    assert len(df3) == 3
    df4 = df.belief_history(event_start, belief_horizon_window=(timedelta(weeks=-10), timedelta(hours=2.5)))  # Only 2 beliefs were formed up to 2.5 hours before knowledge_time, and none after
    assert len(df4) == 2


def test_query_rolling_horizon(time_slot_sensor: Sensor, rolling_day_ahead_beliefs_about_time_slot_events):
    belief_df = TimedBelief.query(sensor=time_slot_sensor, belief_before=datetime(2050, 1, 1, 15, tzinfo=utc)).rolling_horizon(belief_horizon=timedelta(days=2))
    assert len(belief_df) == 7
    assert (belief_df["event_value"].values == append(arange(10, 16), 106)).all()


def test_query_fixed_horizon(time_slot_sensor: Sensor, rolling_day_ahead_beliefs_about_time_slot_events):
    belief_time = datetime(2050, 1, 1, 11, tzinfo=utc)
    df = TimedBelief.query(sensor=time_slot_sensor, belief_before=datetime(2050, 1, 1, 15, tzinfo=utc)).fixed_horizon(belief_time=belief_time)
    assert len(df) == 3
    assert df[df.index.get_level_values("belief_time") > belief_time].empty


def test_downsample(time_slot_sensor):
    """Downsample from 15 minutes to 2 hours."""
    new_resolution = timedelta(hours=2)
    belief_df = TimedBelief.query(sensor=time_slot_sensor, belief_before=datetime(2100, 1, 1, 13, tzinfo=utc))
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution


def test_upsample(time_slot_sensor):
    """Upsample from 15 minutes to 5 minutes."""
    new_resolution = timedelta(minutes=5)
    belief_df = TimedBelief.query(sensor=time_slot_sensor, belief_before=datetime(2100, 1, 1, 13, tzinfo=utc))
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution
