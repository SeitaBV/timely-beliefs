from datetime import datetime, timedelta

import pandas as pd
import pytest
from pytz import utc

from timely_beliefs import BeliefSource, Sensor, TimedBelief, utils


@pytest.fixture(scope="function")
def day_ahead_belief_about_instantaneous_event(
    instantaneous_sensor: Sensor, test_source_a: BeliefSource
):
    """Define day-ahead belief about an instantaneous event."""
    return TimedBelief(
        source=test_source_a,
        sensor=instantaneous_sensor,
        event_value=1,
        belief_time=datetime(2018, 1, 1, 15, tzinfo=utc),
        event_time=datetime(2018, 1, 2, 0, tzinfo=utc),
    )


@pytest.fixture(scope="function")
def day_ahead_belief_about_time_slot_event(
    time_slot_sensor: Sensor, test_source_a: BeliefSource
):
    """Define day-ahead belief about a time slot event."""
    return TimedBelief(
        source=test_source_a,
        sensor=time_slot_sensor,
        event_value=1,
        belief_time=datetime(2018, 1, 1, 15, tzinfo=utc),
        event_start=datetime(2018, 1, 2, 0, tzinfo=utc),
    )


@pytest.fixture(scope="function")
def day_ahead_belief_about_ex_ante_economical_event(
    ex_ante_economics_sensor: Sensor, test_source_a: BeliefSource
):
    """Define day-ahead belief about an ex-ante economical event."""
    return TimedBelief(
        source=test_source_a,
        sensor=ex_ante_economics_sensor,
        event_value=1,
        belief_time=datetime(2018, 1, 1, 15, tzinfo=utc),
        event_start=datetime(2018, 1, 2, 0, tzinfo=utc),
    )


def test_day_ahead_instantaneous_event_belief(
    day_ahead_belief_about_instantaneous_event: TimedBelief,
):
    assert (
        day_ahead_belief_about_instantaneous_event.event_start
        == day_ahead_belief_about_instantaneous_event.event_end
    )
    assert day_ahead_belief_about_instantaneous_event.belief_horizon == timedelta(
        hours=9
    )


def test_day_ahead_belief_about_time_slot_event(
    day_ahead_belief_about_time_slot_event: TimedBelief,
):
    assert (
        day_ahead_belief_about_time_slot_event.event_start
        < day_ahead_belief_about_time_slot_event.event_end
    )
    assert (
        day_ahead_belief_about_time_slot_event.belief_horizon
        == timedelta(hours=9) + day_ahead_belief_about_time_slot_event.event_resolution
    )


def test_day_ahead_belief_about_ex_ante_economical_event(
    day_ahead_belief_about_ex_ante_economical_event: TimedBelief,
):
    assert day_ahead_belief_about_ex_ante_economical_event.knowledge_time == datetime(
        2018, 1, 1, 11, tzinfo=utc
    )
    assert day_ahead_belief_about_ex_ante_economical_event.belief_horizon == -timedelta(
        hours=4
    )
    assert day_ahead_belief_about_ex_ante_economical_event.belief_horizon == timedelta(
        hours=9
    ) - day_ahead_belief_about_ex_ante_economical_event.sensor.knowledge_horizon(
        day_ahead_belief_about_ex_ante_economical_event.event_start
    )


@pytest.mark.parametrize(
    "dt, ErrorType, match",
    [
        ("someday", ValueError, "not parse"),
        ("2003-01-05", TypeError, "timezone-naive"),
        (pd.Timestamp("2003-01-05").to_datetime64(), TypeError, "timezone-naive"),
    ],
)
def test_datetime_parsing(dt, ErrorType, match):
    with pytest.raises(ErrorType, match=match):
        utils.parse_datetime_like(dt)


@pytest.mark.parametrize(
    "td, ErrorType, match",
    [
        ("a while", ValueError, "not parse"),
    ],
)
def test_timedelta_parsing(td, ErrorType, match):
    with pytest.raises(ErrorType, match=match):
        utils.parse_timedelta_like(td)


def test_source_ordering_is_a_total_order_for_same_name_sources():
    """Distinct sources sharing a name must still have a strict total order.

    Regression test for https://github.com/SeitaBV/timely-beliefs/issues/238.
    """
    a = BeliefSource("same name")
    b = BeliefSource("same name")
    assert (a < b) != (b < a)
    assert (a > b) != (b > a)
    assert not (a < b and a > b)


def test_concat_frames_with_same_name_sources():
    """Concatenating frames whose source levels hold distinct sources sharing a
    name must not map any source to NaN.

    Regression test for https://github.com/SeitaBV/timely-beliefs/issues/238.
    """
    from timely_beliefs import BeliefsDataFrame

    sensor = Sensor("total order sensor", event_resolution=timedelta(hours=1))
    sources = [BeliefSource("s" + str(i % 2 + 1)) for i in range(6)]
    event_starts = pd.date_range("2025-01-01", periods=5, freq="1h", tz="UTC")
    belief_times = pd.date_range("2024-12-31", periods=3, freq="1h", tz="UTC")
    spec = [
        (4, 1, 0, [0.3, 0.7]),
        (2, 3, 2, [0.5]),
        (1, 3, 2, [0.5]),
        (3, 0, 1, [0.3, 0.7]),
        (1, 3, 1, [0.5]),
        (5, 2, 0, [0.5]),
        (4, 2, 2, [0.3, 0.7]),
    ]
    frames = [
        BeliefsDataFrame(
            [
                TimedBelief(
                    sensor=sensor,
                    source=sources[s],
                    event_start=event_starts[e],
                    belief_time=belief_times[b],
                    cumulative_probability=cp,
                    event_value=1.0,
                )
                for cp in cps
            ]
        )
        for s, e, b, cps in spec
    ]
    bdf = pd.concat(frames)
    returned_sources = bdf.index.get_level_values("source")
    assert all(isinstance(source, BeliefSource) for source in returned_sources)
