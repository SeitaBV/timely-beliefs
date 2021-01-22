from datetime import datetime, timedelta

import pandas as pd
import pytest
from pytz import utc

from timely_beliefs import BeliefSource, Sensor, TimedBelief, utils


@pytest.fixture(scope="function")
def day_ahead_belief_about_instantaneous_event(
    instantaneous_sensor: Sensor, test_source_a: BeliefSource
):
    """Define day-ahead belief about an instantaneous event."""
    return TimedBelief(
        source=test_source_a,
        sensor=instantaneous_sensor,
        value=1,
        belief_time=datetime(2018, 1, 1, 15, tzinfo=utc),
        event_time=datetime(2018, 1, 2, 0, tzinfo=utc),
    )


@pytest.fixture(scope="function")
def day_ahead_belief_about_time_slot_event(
    time_slot_sensor: Sensor, test_source_a: BeliefSource
):
    """Define day-ahead belief about a time slot event."""
    return TimedBelief(
        source=test_source_a,
        sensor=time_slot_sensor,
        value=1,
        belief_time=datetime(2018, 1, 1, 15, tzinfo=utc),
        event_start=datetime(2018, 1, 2, 0, tzinfo=utc),
    )


@pytest.fixture(scope="function")
def day_ahead_belief_about_ex_post_time_slot_event(
    ex_post_time_slot_sensor: Sensor, test_source_a: BeliefSource
):
    """Define day-ahead belief about an ex post time slot event."""
    return TimedBelief(
        source=test_source_a,
        sensor=ex_post_time_slot_sensor,
        value=1,
        belief_time=datetime(2018, 1, 1, 15, tzinfo=utc),
        event_start=datetime(2018, 1, 2, 0, tzinfo=utc),
    )


def test_day_ahead_instantaneous_event_belief(
    day_ahead_belief_about_instantaneous_event: TimedBelief,
):
    assert (
        day_ahead_belief_about_instantaneous_event.event_start
        == day_ahead_belief_about_instantaneous_event.event_end
    )
    assert day_ahead_belief_about_instantaneous_event.belief_horizon == timedelta(
        hours=9
    )


def test_day_ahead_belief_about_time_slot_event(
    day_ahead_belief_about_time_slot_event: TimedBelief,
):
    assert (
        day_ahead_belief_about_time_slot_event.event_start
        < day_ahead_belief_about_time_slot_event.event_end
    )
    assert (
        day_ahead_belief_about_time_slot_event.belief_horizon
        == timedelta(hours=9) + day_ahead_belief_about_time_slot_event.event_resolution
    )


def test_day_ahead_belief_about_ex_post_time_slot_event(
    day_ahead_belief_about_ex_post_time_slot_event: TimedBelief,
):
    assert day_ahead_belief_about_ex_post_time_slot_event.knowledge_time == datetime(
        2018, 1, 1, 11, tzinfo=utc
    )
    assert day_ahead_belief_about_ex_post_time_slot_event.belief_horizon == -timedelta(
        hours=4
    )
    assert day_ahead_belief_about_ex_post_time_slot_event.belief_horizon == timedelta(
        hours=9
    ) - day_ahead_belief_about_ex_post_time_slot_event.sensor.knowledge_horizon(
        day_ahead_belief_about_ex_post_time_slot_event.event_start
    )


@pytest.mark.parametrize(
    "dt, ErrorType, match",
    [
        ("someday", ValueError, "not parse"),
        ("2003-01-05", TypeError, "timezone-naive"),
        (pd.Timestamp("2003-01-05").to_datetime64(), TypeError, "timezone-naive"),
    ],
)
def test_datetime_parsing(dt, ErrorType, match):
    with pytest.raises(ErrorType, match=match):
        utils.parse_datetime_like(dt)


@pytest.mark.parametrize(
    "td, ErrorType, match",
    [
        ("a while", ValueError, "not parse"),
    ],
)
def test_timedelta_parsing(td, ErrorType, match):
    with pytest.raises(ErrorType, match=match):
        utils.parse_timedelta_like(td)
