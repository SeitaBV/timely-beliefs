import pytest
from datetime import timedelta

from base import session
from timely_beliefs import BeliefSource, Sensor
from timely_beliefs.func_store.knowledge_horizons import timedelta_x_days_ago_at_y_oclock


@pytest.fixture(scope="module")
def instantaneous_sensor():
    """Define sensor for instantaneous events."""
    sensor = Sensor()
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="module")
def time_slot_sensor():
    """Define sensor for time slot events."""
    sensor = Sensor(event_resolution=timedelta(minutes=15))
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="module")
def ex_post_time_slot_sensor():
    """Define sensor for time slot events known in advance (ex post)."""
    sensor = Sensor(
        event_resolution=timedelta(minutes=15),
        knowledge_horizon=(timedelta_x_days_ago_at_y_oclock, dict(x=1, y=12, z="Europe/Amsterdam")),
    )
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="module")
def test_source():
    """Define source for test beliefs."""
    source = BeliefSource()
    session.add(source)
    session.flush()
    return source
