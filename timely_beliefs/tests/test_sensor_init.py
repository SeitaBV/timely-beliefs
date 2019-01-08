from datetime import datetime, timedelta

from pytz import utc

from timely_beliefs import Sensor


def test_instantaneous_sensor(instantaneous_sensor: Sensor):
    assert instantaneous_sensor.event_resolution == timedelta()


def test_time_slot_sensor(time_slot_sensor: Sensor):
    assert time_slot_sensor.event_resolution == timedelta(minutes=15)
    event_end = datetime(2018, 1, 1, 15, tzinfo=utc)
    assert time_slot_sensor.knowledge_time(event_end) == event_end


def test_ex_post_time_slot_sensor(ex_post_time_slot_sensor: Sensor):
    event_end = datetime(2018, 1, 1, 15, tzinfo=utc)
    assert ex_post_time_slot_sensor.knowledge_time(event_end) < event_end
