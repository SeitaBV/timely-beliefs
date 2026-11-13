from __future__ import annotations

import inspect
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pytz import utc
from sqlalchemy import event

from timely_beliefs import DBSensor, DBTimedBelief
from timely_beliefs.sensors.func_store.knowledge_horizons import ex_post
from timely_beliefs.tests import session
from timely_beliefs.tests.conftest import (
    create_ex_ante_economics_sensor,
    create_ex_ante_physics_sensor,
)


def make_ex_post():
    """Define an ex-post sensor."""
    sensor = DBSensor(
        name="ex-post",
        event_resolution=timedelta(minutes=15),
        knowledge_horizon=(ex_post, dict(ex_post_horizon=timedelta(0))),
    )
    session.add(sensor)
    session.flush()
    return sensor


def make_sensors(instantaneous_sensor, time_slot_sensor, ex_ante_economics_sensor):
    """Define sensors that differ in event resolution and in knowledge horizon."""
    plain = DBSensor(name="plain 15", event_resolution=timedelta(minutes=15))
    hourly = DBSensor(name="hourly", event_resolution=timedelta(hours=1))
    session.add_all([plain, hourly])
    session.flush()
    return [
        instantaneous_sensor,
        time_slot_sensor,
        plain,
        hourly,
        ex_ante_economics_sensor,
        create_ex_ante_economics_sensor("ex-ante B"),
        make_ex_post(),
        create_ex_ante_physics_sensor("ex-ante physics"),
    ]


def add_beliefs(sensors, sources):
    """Add beliefs for each sensor and source; odd sensors form their beliefs 2 hours earlier."""
    start = datetime(2024, 1, 1, tzinfo=utc)
    beliefs = []
    for s, sensor in enumerate(sensors):
        for source in sources:
            for e in range(4):
                for b in range(3):
                    beliefs.append(
                        DBTimedBelief(
                            sensor=sensor,
                            source=source,
                            event_start=start
                            + sensor.event_resolution * e
                            + timedelta(hours=e),
                            belief_horizon=timedelta(hours=2 * (b + 1))
                            - timedelta(hours=3)
                            + timedelta(hours=2 * (s % 2)),
                            event_value=10.0 * s + e + b / 10,
                        )
                    )
    session.add_all(beliefs)
    session.flush()


FILTERS = {
    "none": {},
    "mrb": dict(most_recent_beliefs_only=True),
    "mre": dict(most_recent_events_only=True),
    "mrb+mre": dict(most_recent_beliefs_only=True, most_recent_events_only=True),
    "mro": dict(most_recent_only=True),
    "event_window": dict(
        event_starts_after=pd.Timestamp("2024-01-01 01:00Z"),
        event_ends_before=pd.Timestamp("2024-01-01 03:00Z"),
    ),
    "event_ends_after": dict(
        event_ends_after=pd.Timestamp("2024-01-01 01:30Z"),
        most_recent_beliefs_only=True,
    ),
    "event_starts_before": dict(
        event_starts_before=pd.Timestamp("2024-01-01 02:00Z"),
        most_recent_events_only=True,
    ),
    "beliefs_before": dict(beliefs_before=pd.Timestamp("2023-12-31 22:00Z")),
    "beliefs_before+mrb": dict(
        beliefs_before=pd.Timestamp("2024-01-01 00:30Z"), most_recent_beliefs_only=True
    ),
    "beliefs_between+mrb": dict(
        beliefs_after=pd.Timestamp("2023-12-31 20:00Z"),
        beliefs_before=pd.Timestamp("2024-01-01 01:30Z"),
        most_recent_beliefs_only=True,
    ),
    "h_at_least": dict(
        horizons_at_least=timedelta(hours=0), most_recent_beliefs_only=True
    ),
    "h_at_most": dict(horizons_at_most=timedelta(hours=0)),
    "source_a": dict(source="a"),
    "source_a+mrb+mre": dict(
        source="a", most_recent_beliefs_only=True, most_recent_events_only=True
    ),
    "mview_cutoff": dict(
        most_recent_beliefs_only=True, mview_cutoff=pd.Timestamp("2024-01-01 02:00Z")
    ),
    "no_tz": dict(
        place_beliefs_in_sensor_timezone=False,
        place_events_in_sensor_timezone=False,
        most_recent_beliefs_only=True,
    ),
    "empty_sources": dict(source=[]),
}


@pytest.mark.parametrize("use_mview", [False, True])
@pytest.mark.parametrize("name", list(FILTERS))
def test_multi_equals_single(
    name,
    use_mview,
    refresh_mview,
    instantaneous_sensor,
    time_slot_sensor,
    ex_ante_economics_sensor,
    test_source_a,
    test_source_b,
):
    """For each sensor, the result equals what a search for that sensor alone returns."""
    sensors = make_sensors(
        instantaneous_sensor, time_slot_sensor, ex_ante_economics_sensor
    )
    add_beliefs(sensors, [test_source_a, test_source_b])
    if use_mview:
        refresh_mview()
    kwargs = dict(FILTERS[name])
    if kwargs.get("source") == "a":
        kwargs["source"] = test_source_a
    order = sensors[::-1] + [sensors[2]]  # reversed, with a duplicate
    multi = DBTimedBelief.search_session_multi(
        session, order, use_materialized_view=use_mview, **kwargs
    )
    assert len(multi) == len(order)
    for sensor, df in zip(order, multi):
        single = DBTimedBelief.search_session(
            session, sensor, use_materialized_view=use_mview, **kwargs
        )
        pd.testing.assert_frame_equal(df, single)
        assert df.sensor == sensor
    if name != "empty_sources":
        assert any(len(df) for df in multi)


def test_multi_accepts_ids_and_raises_for_unknown(
    instantaneous_sensor, time_slot_sensor, test_source_a
):
    sensors = [instantaneous_sensor, time_slot_sensor]
    add_beliefs(sensors, [test_source_a])
    ids = [time_slot_sensor.id, instantaneous_sensor.id]
    multi = DBTimedBelief.search_session_multi(session, ids)
    assert [df.sensor.id for df in multi] == ids
    with pytest.raises(ValueError, match="No such sensor"):
        DBTimedBelief.search_session_multi(session, ids + [99999])
    with pytest.raises(ValueError, match="No such sensor"):
        DBTimedBelief.search_session(session, 99999)


def test_multi_accepts_numpy_ids(instantaneous_sensor, time_slot_sensor, test_source_a):
    """Ids taken from an array are numpy integers, and they should work like normal ids."""
    sensors = [instantaneous_sensor, time_slot_sensor]
    add_beliefs(sensors, [test_source_a])
    ids = np.array([time_slot_sensor.id, instantaneous_sensor.id])
    multi = DBTimedBelief.search_session_multi(session, list(ids))
    expected = DBTimedBelief.search_session_multi(session, [int(i) for i in ids])
    assert [df.sensor.id for df in multi] == [int(i) for i in ids]
    for df, other in zip(multi, expected):
        pd.testing.assert_frame_equal(df, other)
    single = DBTimedBelief.search_session(session, ids[0])
    pd.testing.assert_frame_equal(single, expected[0])


def test_unknown_sensors_are_named(instantaneous_sensor):
    with pytest.raises(ValueError, match=r"No such sensor\(s\): \[99998, 99999\]"):
        DBTimedBelief.search_session_multi(
            session, [instantaneous_sensor.id, 99999, 99998]
        )


def test_wrong_sensor_class_is_named(instantaneous_sensor):
    with pytest.raises(ValueError, match="sensor_class .*int.* is not a subclass"):
        DBTimedBelief.search_session_multi(
            session, [instantaneous_sensor.id], sensor_class=int
        )


def test_multi_copies_repeated_sensors(
    instantaneous_sensor, time_slot_sensor, test_source_a
):
    """A sensor that is given twice gets two frames, and a change in one does not change the other."""
    sensors = [instantaneous_sensor, time_slot_sensor]
    add_beliefs(sensors, [test_source_a])
    multi = DBTimedBelief.search_session_multi(
        session, [instantaneous_sensor, time_slot_sensor, instantaneous_sensor]
    )
    assert len(multi[0]) > 0
    assert multi[0] is not multi[2]
    pd.testing.assert_frame_equal(multi[0], multi[2])
    assert multi[2].sensor == instantaneous_sensor
    multi[2]["event_value"] = -1.0
    assert (multi[0]["event_value"] != -1.0).all()


def test_multi_has_the_arguments_of_search_session():
    """search_session_multi takes the arguments of search_session, with a list of sensors."""
    single = dict(inspect.signature(DBTimedBelief.search_session).parameters)
    multi = dict(inspect.signature(DBTimedBelief.search_session_multi).parameters)
    single.pop("sensor")
    multi.pop("sensors")
    assert list(multi) == list(single)
    for name, parameter in single.items():
        assert multi[name].default == parameter.default, name


def test_multi_rejects_unknown_arguments():
    """A misspelled argument is an error, also when there are no sensors to search for."""
    with pytest.raises(TypeError, match="beliefs_befor"):
        DBTimedBelief.search_session_multi(
            session, [], beliefs_befor=datetime(2024, 1, 1, tzinfo=utc)
        )


def test_multi_with_sensors_in_different_timezones(test_source_a):
    """Each result is placed in the timezone of its sensor, also around a change of the clock."""
    sensors = [
        DBSensor(
            name=f"hourly {timezone}",
            event_resolution=timedelta(hours=1),
            timezone=timezone,
        )
        for timezone in ("UTC", "Europe/Amsterdam", "Asia/Tokyo")
    ]
    session.add_all(sensors)
    session.flush()
    # Amsterdam changes its clock on 2024-03-31 at 01:00 UTC
    start = datetime(2024, 3, 30, 22, tzinfo=utc)
    session.add_all(
        DBTimedBelief(
            sensor=sensor,
            source=test_source_a,
            event_start=start + timedelta(hours=e),
            belief_horizon=timedelta(hours=h),
            event_value=float(10 * s + e + h),
        )
        for s, sensor in enumerate(sensors)
        for e in range(8)
        for h in (1, 3)
    )
    session.flush()
    for kwargs in (
        dict(),
        dict(
            event_ends_before=pd.Timestamp("2024-03-31 04:00Z"),
            most_recent_beliefs_only=True,
        ),
    ):
        multi = DBTimedBelief.search_session_multi(
            session, sensors, use_materialized_view=False, **kwargs
        )
        for sensor, df in zip(sensors, multi):
            single = DBTimedBelief.search_session(
                session, sensor, use_materialized_view=False, **kwargs
            )
            pd.testing.assert_frame_equal(df, single)
            assert len(df) > 0
            event_starts = df.index.get_level_values("event_start")
            assert str(event_starts.tz) == sensor.timezone


def test_multi_with_a_custom_join_on_the_sensor_table(test_source_a):
    """A filter on a joined table works, and a sensor that does not match gets an empty frame."""
    sensors = [
        DBSensor(name=name, event_resolution=timedelta(minutes=15))
        for name in ("keep a", "keep b", "drop c")
    ]
    session.add_all(sensors)
    session.flush()
    add_beliefs(sensors, [test_source_a])
    kwargs = dict(
        custom_join_targets=[DBSensor],
        custom_filter_criteria=[DBSensor.name.like("keep%")],
        use_materialized_view=False,
    )
    multi = DBTimedBelief.search_session_multi(session, sensors, **kwargs)
    for sensor, df in zip(sensors, multi):
        single = DBTimedBelief.search_session(session, sensor, **kwargs)
        pd.testing.assert_frame_equal(df, single)
    assert len(multi[0]) > 0 and len(multi[1]) > 0
    assert len(multi[2]) == 0


def statements(sensors, **kwargs):
    """Return the SQL statements that read beliefs."""
    seen = []

    def record(conn, cursor, statement, *a):
        seen.append(statement)

    event.listen(session.get_bind(), "before_cursor_execute", record)
    try:
        DBTimedBelief.search_session_multi(session, sensors, **kwargs)
    finally:
        event.remove(session.get_bind(), "before_cursor_execute", record)
    return [s for s in seen if "event_value" in s]


def test_one_query_per_group(
    instantaneous_sensor,
    time_slot_sensor,
    ex_ante_economics_sensor,
    test_source_a,
    test_source_b,
):
    """Sensors share a query, unless the filters depend on how they differ."""
    sensors = make_sensors(
        instantaneous_sensor, time_slot_sensor, ex_ante_economics_sensor
    )
    add_beliefs(sensors, [test_source_a])
    base = dict(use_materialized_view=False)
    assert len(statements(sensors, most_recent_beliefs_only=True, **base)) == 1
    assert len(statements(sensors, most_recent_events_only=True, **base)) == 1
    # resolution matters once event ends are filtered: 0 min, 15 min, 1 hour
    assert (
        len(
            statements(
                sensors, event_ends_before=pd.Timestamp("2024-01-02 00:00Z"), **base
            )
        )
        == 3
    )
    # knowledge horizon matters once belief times are filtered
    n = len(
        statements(sensors, beliefs_before=pd.Timestamp("2024-01-02 00:00Z"), **base)
    )
    assert 1 < n < len(sensors)
    assert len(statements(sensors, most_recent_only=True, **base)) == len(sensors)
