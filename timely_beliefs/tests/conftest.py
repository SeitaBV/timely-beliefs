import pytest
from datetime import timedelta

from base import session
from timely_beliefs import Sensor
from timely_beliefs import BeliefSource
from timely_beliefs.utils import timedelta_x_days_ago_at_y_oclock


@pytest.fixture(scope="module")
def instantaneous_sensor():
    """Define sensor for instantaneous events."""
    sensor = Sensor()
    session.add(sensor)
    return sensor


@pytest.fixture(scope="module")
def time_slot_sensor():
    """Define sensor for time slot events."""
    sensor = Sensor(event_resolution=timedelta(minutes=15))
    session.add(sensor)
    return sensor


@pytest.fixture(scope="module")
def ex_post_time_slot_sensor():
    """Define sensor for time slot events known in advance."""
    sensor = Sensor(
        event_resolution=timedelta(minutes=15),
        knowledge_horizon=(timedelta_x_days_ago_at_y_oclock, dict(x=1, y=12)),
    )
    session.add(sensor)
    return sensor


@pytest.fixture(scope="module")
def test_source():
    """Define source for test beliefs."""
    source = BeliefSource()
    session.add(source)
    return source
