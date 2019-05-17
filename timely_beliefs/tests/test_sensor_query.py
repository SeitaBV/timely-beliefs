from timely_beliefs import DBSensor
from timely_beliefs.tests import session


def test_persist_sensor(db):
    assert session.query(DBSensor).first()
