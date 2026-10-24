from __future__ import annotations

from datetime import datetime, timedelta

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
