from timely_beliefs import DBSensor
from timely_beliefs.tests import session


def test_persist_sensor(db):
    assert session.query(DBSensor).first()


def test_ranged_sensor(db):
    sensor = (
        session.query(DBSensor)
        .filter(DBSensor.name == "InstantaneousRangedSensor")
        .one_or_none()
    )
    assert sensor.value_range == (0, 20)
