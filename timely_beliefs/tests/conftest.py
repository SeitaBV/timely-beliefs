import sys
from datetime import datetime, timedelta

import pytest
from pytz import utc

from timely_beliefs import DBBeliefSource, DBSensor, DBTimedBelief
from timely_beliefs.db_base import Base
from timely_beliefs.sensors.func_store.knowledge_horizons import (
    determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock,
)
from timely_beliefs.tests import engine, session

if sys.version_info[0] == 3 and sys.version_info[1] == 6:
    # Ignore these tests for python==3.6
    collect_ignore = ["test_ignore_36.py"]


@pytest.fixture(scope="function")
def db():
    """
    For each test, provide a db object with the structure freshly created. This assumes a clean database.
    It does clean up after itself when it's done (drops everything).
    """

    Base.metadata.create_all(engine)

    yield Base.metadata

    # Explicitly close DB connection
    session.close()

    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function", autouse=True)
def instantaneous_sensor(db):
    """Define sensor for instantaneous events."""
    sensor = DBSensor(name="InstantaneousSensor")
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="function", autouse=True)
def time_slot_sensor(db):
    """Define sensor for time slot events."""
    sensor = DBSensor(
        name="TimeSlot15MinSensor", event_resolution=timedelta(minutes=15)
    )
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="function", autouse=True)
def ex_post_time_slot_sensor(db):
    """Define sensor for time slot events known in advance (ex post)."""
    return create_ex_post_time_slot_sensor("ExPostSensor")


@pytest.fixture(scope="function", autouse=False)
def ex_post_time_slot_sensor_b(db):
    """Define an almost identical ex-post time slot sensor, just with a different name."""
    return create_ex_post_time_slot_sensor("ExPostSensor B")


def create_ex_post_time_slot_sensor(name: str) -> DBSensor:
    """Define sensor for time slot events known in advance (ex post)."""
    sensor = DBSensor(
        name=name,
        event_resolution=timedelta(minutes=15),
        knowledge_horizon=(
            determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock,
            dict(x=1, y=12, z="Europe/Amsterdam"),
        ),
    )
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="function", autouse=True)
def test_source_a(db):
    """Define source for test beliefs."""
    source = DBBeliefSource("Source A")
    session.add(source)
    return source


@pytest.fixture(scope="function", autouse=True)
def test_source_b(db):
    """Define source for test beliefs."""
    source = DBBeliefSource("Source B")
    session.add(source)
    return source


@pytest.fixture(scope="function", autouse=True)
def test_source_without_initial_data(db):
    """Define source without initial test beliefs."""
    source = DBBeliefSource("Source without initial data")
    session.add(source)
    return source


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
