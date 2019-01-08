import pytest
from datetime import datetime

from pytz import utc
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
    session.flush()
    return belief


def test_persist_belief():
    assert session.query(TimedBelief).first()


def test_query_belief_by_belief_time(ex_post_time_slot_sensor: Sensor, day_ahead_belief_about_ex_post_time_slot_event):
    belief_df = TimedBelief.query(sensor=ex_post_time_slot_sensor, before=datetime(2018, 1, 1, 13, tzinfo=utc))

    # By calling a pandas Series for its values we lose the timezone (a pandas bug still present in version 0.23.4)
    # This next test warns us when it has been fixed (if it fails, just replace != with ==).
    assert belief_df["knowledge_time"].values[0] != datetime(2018, 1, 1, 11, 0, tzinfo=utc)
    # And this test is just a workaround to test what we wanted to test.
    assert Timestamp(belief_df["knowledge_time"].values[0]) == Timestamp(datetime(2018, 1, 1, 11, 0))

    # Just one belief found
    assert len(belief_df.index) == 1

    # No beliefs a year earlier
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, before=datetime(2017, 1, 1, 10, tzinfo=utc)).index) == 0

    # No beliefs 2 months later
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, not_before=datetime(2018, 1, 3, 10, tzinfo=utc)).index) == 0

    # One belief after 10am UTC
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, not_before=datetime(2018, 1, 1, 10, tzinfo=utc)).index) == 1

    # No beliefs an hour earlier
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, before=datetime(2018, 1, 1, 9, tzinfo=utc)).index) == 0

    # No beliefs after 1pm UTC
    assert len(TimedBelief.query(sensor=ex_post_time_slot_sensor, not_before=datetime(2018, 1, 1, 13, tzinfo=utc)).index) == 0
