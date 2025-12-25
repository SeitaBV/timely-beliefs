from datetime import datetime, timedelta

from pytz import utc

from timely_beliefs import DBSensor


def test_instantaneous_sensor(instantaneous_sensor: DBSensor):
    assert instantaneous_sensor.event_resolution == timedelta()


def test_time_slot_sensor(time_slot_sensor: DBSensor):
    assert time_slot_sensor.event_resolution == timedelta(minutes=15)
    event_start = datetime(2018, 1, 1, 15, tzinfo=utc)
    event_end = event_start + time_slot_sensor.event_resolution
    assert time_slot_sensor.knowledge_time(event_start) == event_end
    assert (
        time_slot_sensor.knowledge_horizon(event_start)
        == -time_slot_sensor.event_resolution
    )


def test_ex_ante_economics_sensor(ex_ante_economics_sensor: DBSensor):
    event_start = datetime(2018, 1, 1, 15, tzinfo=utc)
    assert ex_ante_economics_sensor.knowledge_time(event_start) < event_start


def test_ex_ante_physics_sensor(ex_ante_physics_sensor: DBSensor):
    event_start = datetime(2026, 1, 1, 15, tzinfo=utc)
    assert ex_ante_physics_sensor.knowledge_time(event_start) == event_start
    assert ex_ante_physics_sensor.knowledge_horizon(event_start) == timedelta(0)
