import importlib.util
import sys
from datetime import datetime, timedelta

import pytest
from pytz import utc

from timely_beliefs import DBBeliefSource, DBSensor, DBTimedBelief
from timely_beliefs.db_base import Base
from timely_beliefs.sensors.func_store.knowledge_horizons import (
    at_date,
    ex_post,
    x_days_ago_at_y_oclock,
)
from timely_beliefs.tests import engine, session

collect_ignore_glob = []
if sys.version_info[0] == 3 and sys.version_info[1] == 6:
    # Ignore these tests for python==3.6
    print("Ignoring optional test modules prepended with 'test_ignore_36__'.")
    collect_ignore_glob += ["test_ignore_36__*.py"]
if importlib.util.find_spec("altair") is None:
    # Ignore these tests if altair is not installed
    print("Ignoring optional test modules prepended with 'test_viz__'.")
    collect_ignore_glob += ["test_viz__*.py"]
if importlib.util.find_spec("sktime") is None:
    # Ignore these tests if sktime is not installed
    print("Ignoring optional test modules prepended with 'test_forecast__'")
    collect_ignore_glob += ["test_forecast__*.py"]


@pytest.fixture(scope="function")
def db():
    """
    For each test, provide a db object with the structure freshly created. This assumes a clean database.
    It does clean up after itself when it's done (drops everything).
    """

    Base.metadata.create_all(engine)

    yield Base.metadata

    # Explicitly close DB connection
    session.close()

    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function", autouse=True)
def instantaneous_sensor(db):
    """Define sensor for instantaneous events."""
    sensor = DBSensor(name="InstantaneousSensor")
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="function", autouse=True)
def time_slot_sensor(db):
    """Define sensor for time slot events."""
    sensor = DBSensor(
        name="TimeSlot15MinSensor", event_resolution=timedelta(minutes=15)
    )
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="function", autouse=True)
def ex_ante_economics_sensor(db):
    """Define sensor for time slot events known in advance (ex ante)."""
    return create_ex_ante_economics_sensor("ex-ante sensor A")


@pytest.fixture(scope="function", autouse=False)
def ex_ante_economics_sensor_b(db):
    """Define an almost identical ex-ante time slot sensor, just with a different name."""
    return create_ex_ante_economics_sensor("ex-ante sensor B")


def create_ex_ante_economics_sensor(name: str) -> DBSensor:
    """Define sensor for economical events known in advance (ex ante)."""
    sensor = DBSensor(
        name=name,
        event_resolution=timedelta(minutes=15),
        knowledge_horizon=(
            x_days_ago_at_y_oclock,
            dict(x=1, y=12, z="Europe/Amsterdam"),
        ),
    )
    session.add(sensor)
    session.flush()
    return sensor


def create_ex_post_physics_sensor(name: str) -> DBSensor:
    """Define sensor for physical events known after the fact (ex post)."""
    sensor = DBSensor(
        name=name,
        event_resolution=timedelta(minutes=15),
        knowledge_horizon=(
            ex_post,
            dict(event_resolution=timedelta(minutes=15), ex_post_horizon=timedelta(0)),
        ),
    )
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="function", autouse=False)
def unique_knowledge_time_sensor() -> DBSensor:
    """Define sensor recording events with a unique knowledge time."""
    sensor = DBSensor(
        name="SinglePublicationSensor",
        event_resolution=timedelta(hours=1),
        knowledge_horizon=(
            at_date,
            dict(knowledge_time=datetime(1990, 5, 10, 0, tzinfo=utc)),
        ),
    )
    session.add(sensor)
    session.flush()
    return sensor


@pytest.fixture(scope="function", autouse=True)
def test_source_a(db):
    """Define source for test beliefs."""
    source = DBBeliefSource("Source A")
    session.add(source)
    return source


@pytest.fixture(scope="function", autouse=True)
def test_source_b(db):
    """Define source for test beliefs."""
    source = DBBeliefSource("Source B")
    session.add(source)
    return source


@pytest.fixture(scope="function", autouse=True)
def test_source_without_initial_data(db):
    """Define source without initial test beliefs."""
    source = DBBeliefSource("Source without initial data")
    session.add(source)
    return source


@pytest.fixture(scope="function", params=[1, 2])
def rolling_day_ahead_beliefs_about_time_slot_events(
    request, time_slot_sensor: DBSensor
):
    """Define multiple day-ahead beliefs about an ex post time slot event."""
    source = (
        session.query(DBBeliefSource)
        .filter(DBBeliefSource.id == request.param)
        .one_or_none()
    )

    beliefs = []
    for i in range(1, 11):  # ten events

        # Recent belief (horizon 48 hours)
        belief = DBTimedBelief(
            sensor=time_slot_sensor,
            source=source,
            event_value=10 + i,
            belief_time=datetime(2050, 1, 1, 10, tzinfo=utc) + timedelta(hours=i),
            event_start=datetime(2050, 1, 3, 10, tzinfo=utc) + timedelta(hours=i),
        )
        session.add(belief)
        beliefs.append(belief)

        # Slightly older beliefs (belief_time an hour earlier)
        belief = DBTimedBelief(
            sensor=time_slot_sensor,
            source=source,
            event_value=100 + i,
            belief_time=datetime(2050, 1, 1, 10, tzinfo=utc) + timedelta(hours=i - 1),
            event_start=datetime(2050, 1, 3, 10, tzinfo=utc) + timedelta(hours=i),
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs
