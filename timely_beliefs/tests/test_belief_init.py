import warnings
from datetime import datetime, timedelta

import pandas as pd
import pytest
from pytz import utc

import timely_beliefs as tb
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


def test_timedelta_parsing_passes_on_unrelated_warnings(monkeypatch):
    """A warning other than pandas' FutureWarning about ambiguous units does not stop the parsing.

    For example, pandas 2 emits a DeprecationWarning on numpy 2.5 when parsing a timedelta,
    which used to be raised as an error, so beliefs with a horizon given as a string could not be created.
    The warning should reach the caller as a warning instead.
    """

    class WarningTimedelta(pd.Timedelta):
        def __new__(cls, *args, **kwargs):
            warnings.warn("unrelated deprecation", DeprecationWarning)
            # An instance of this class, so the parser still recognises it as a pd.Timedelta.
            return super().__new__(cls, *args, **kwargs)

    monkeypatch.setattr(utils.pd, "Timedelta", WarningTimedelta)
    with pytest.warns(DeprecationWarning, match="unrelated deprecation"):
        td = utils.parse_timedelta_like("PT1H")
    assert type(td) is timedelta
    assert td == timedelta(hours=1)


class HashCountingSource(BeliefSource):
    """A source that counts how often a source is hashed."""

    hashes = 0

    def __hash__(self):
        HashCountingSource.hashes += 1
        return id(self)


def test_categorical_source_column():
    """A frame built from a categorical source column equals one built from a column of source objects.

    Its source level holds only the sources that have beliefs, as an object Index,
    so search_session can pass its sources as a categorical column without changing the frames it returns.
    Building it hashes each source a few times, rather than a few times for each row.
    """
    sensor = Sensor("sensor", event_resolution=timedelta(minutes=15))
    sources = [HashCountingSource(name) for name in ("A", "B", "without beliefs")]
    n = 1000
    codes = [i % 2 for i in range(n)]
    event_starts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    data = dict(
        event_start=event_starts,
        belief_time=event_starts - timedelta(hours=1),
        cumulative_probability=0.5,
        event_value=range(n),
    )
    from_objects = tb.BeliefsDataFrame(
        pd.DataFrame({**data, "source": [sources[c] for c in codes]}), sensor=sensor
    )
    HashCountingSource.hashes = 0
    from_categories = tb.BeliefsDataFrame(
        pd.DataFrame(
            {**data, "source": pd.Categorical.from_codes(codes, categories=sources)}
        ),
        sensor=sensor,
    )
    assert HashCountingSource.hashes < 50
    pd.testing.assert_frame_equal(from_categories, from_objects)
    source_level = from_categories.index.levels[
        from_categories.index.names.index("source")
    ]
    assert type(source_level) is pd.Index
    assert source_level.dtype == object
    assert set(source_level) == set(sources[:2])
