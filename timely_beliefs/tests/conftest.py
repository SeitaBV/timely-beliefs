import pytest
from datetime import timedelta

from timely_beliefs.base import Base, engine, session
from timely_beliefs import DBBeliefSource, DBSensor
from timely_beliefs.sensors.func_store.knowledge_horizons import (
    timedelta_x_days_ago_at_y_oclock
)


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
    sensor = DBSensor(
        name="ExPostSensor",
        event_resolution=timedelta(minutes=15),
        knowledge_horizon=(
            timedelta_x_days_ago_at_y_oclock,
            dict(x=1, y=12, z="Europe/Amsterdam"),
        ),
    )
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="function", autouse=True)
def test_source_a():
    """Define source for test beliefs."""
    source = DBBeliefSource()
    session.add(source)
    return source


@pytest.fixture(scope="function", autouse=True)
def test_source_b():
    """Define source for test beliefs."""
    source = DBBeliefSource()
    session.add(source)
    return source
