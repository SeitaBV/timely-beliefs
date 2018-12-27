from timely_beliefs.sensors import Sensor
from base import session


def test_create_sensor():
    assert session.query(Sensor).first()
