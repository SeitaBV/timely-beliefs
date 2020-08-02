from datetime import datetime, timedelta

from sqlalchemy import Column, Float, Integer, ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from pytz import timezone

from timely_beliefs import BeliefSource, DBBeliefSource, DBSensor, DBTimedBelief
from timely_beliefs.sources.classes import BeliefSourceDBMixin
from timely_beliefs.beliefs.classes import DBTimedBeliefMixin
from timely_beliefs.tests import session
from timely_beliefs.db_base import Base


class RatedBeliefSource(DBBeliefSource):
    """ Subclassing a belief source, adding a field."""

    rating = Column(Float(), default=0)

    def __init__(self, rating: float = None, **kwargs):
        self.rating = rating
        DBBeliefSource.__init__(self, **kwargs)


def test_subclassing_source(db):
    session.add(RatedBeliefSource(name="test_source", rating=5))
    session.add(RatedBeliefSource(name="dummy"))

    assert (
        session.query(RatedBeliefSource).filter(RatedBeliefSource.rating == 5).count()
        == 1
    )


# TODO: test if we can add a sensor and then add beliefs
# Maybe that is a worthwhile test on its own before subclassing ...


class RatedBeliefSourceCustomTable(Base, BeliefSourceDBMixin):
    """ A custom db class for representing belief sources.
    BeliefSource properties are added via a Mixin.
    Here we can change the table name.
    We'd also need to take care of relationships ourselves, though!"""

    __tablename__ = "my_belief_source"

    rating = Column(Float(), default=0)


def test_custom_source_with_mixin(db):
    session.add(RatedBeliefSourceCustomTable(name="test_source", rating=7))
    session.add(RatedBeliefSourceCustomTable(name="dummy"))

    assert (
        session.query(RatedBeliefSourceCustomTable)
        .filter(RatedBeliefSourceCustomTable.rating == 7)
        .count()
        == 1
    )

    assert "my_belief_source" in db.tables.keys()


class HappyTimedBeliefCustomTable(Base, DBTimedBeliefMixin):
    """ A custom db class for representing beliefs.
    We overwrite the source_id reference, to our custom source table (see above)
    """

    __tablename__ = "my_timed_belief"

    happiness = Column(Float(), default=0)

    @declared_attr
    def source_id(cls):
        return Column(Integer, ForeignKey("my_belief_source.id"), primary_key=True)

    def __init__(self, sensor, source, happiness: float = None, **kwargs):
        self.happiness = happiness
        DBTimedBeliefMixin.__init__(self, sensor, source, **kwargs)


def test_custom_source_and_beliefs_with_mixin(db):
    source = RatedBeliefSourceCustomTable(name="test_source", rating=7)
    session.add(source)

    sensor = DBSensor(name="AnySensor")
    session.add(sensor)

    session.flush()

    now = datetime.now(tz=timezone("Europe/Amsterdam"))
    belief = HappyTimedBeliefCustomTable(
        sensor=sensor,
        source=source,
        belief_time=now,
        event_start=now + timedelta(minutes=3),
        value=100,
        happiness=3,
    )
    session.add(belief)

    q = session.query(RatedBeliefSourceCustomTable).filter(
        RatedBeliefSourceCustomTable.rating == 7
    )
    assert q.count() == 1
    assert q.first().rating == 7

    q = session.query(HappyTimedBeliefCustomTable).filter(
        HappyTimedBeliefCustomTable.happiness == 3
    )
    assert q.count() == 1
    the_belief = q.first()
    assert the_belief.event_value == belief.event_value
    assert the_belief.sensor.__class__ == DBSensor
    assert the_belief.source.__class__ == RatedBeliefSourceCustomTable

    assert "my_belief_source" in db.tables.keys()
    assert "my_timed_belief" in db.tables.keys()
