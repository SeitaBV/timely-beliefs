import pytest
from datetime import timedelta

from timely_beliefs import Sensor
from timely_beliefs.utils import timedelta_x_days_ago_at_y_oclock


@pytest.fixture(scope="module")
def instantaneous_sensor():
    """Define sensor for instantaneous events."""
    return Sensor()


@pytest.fixture(scope="module")
def time_slot_sensor():
    """Define sensor for time slot events."""
    return Sensor(event_resolution=timedelta(minutes=15))


@pytest.fixture(scope="module")
def ex_post_time_slot_sensor():
    """Define sensor for time slot events known in advance."""
    return Sensor(
        event_resolution=timedelta(minutes=15),
        knowledge_horizon=(timedelta_x_days_ago_at_y_oclock, dict(x=1, y=12)),
    )
