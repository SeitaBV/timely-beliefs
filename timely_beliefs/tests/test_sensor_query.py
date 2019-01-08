from timely_beliefs import Sensor
from base import session


def test_persist_sensor():
    assert session.query(Sensor).first()
