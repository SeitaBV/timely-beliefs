from datetime import datetime, timedelta

from pytz import timezone
from sqlalchemy import Column, Float, ForeignKey, Integer, func, select
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import backref, relationship

import timely_beliefs.utils as tb_utils
from timely_beliefs import BeliefsDataFrame, DBBeliefSource, DBSensor
from timely_beliefs.beliefs.classes import TimedBeliefDBMixin
from timely_beliefs.db_base import Base
from timely_beliefs.sources.classes import BeliefSourceDBMixin
from timely_beliefs.tests import session


class RatedSource(DBBeliefSource):
    """Subclassing a belief source, adding a field."""

    rating = Column(Float(), default=0)

    def __init__(self, rating: float = None, **kwargs):
        self.rating = rating
        DBBeliefSource.__init__(self, **kwargs)

    def __repr__(self):
        return "<RatedSource name=%s rating=%d>" % (self.name, self.rating)


def test_subclassing_source(db):
    session.add(RatedSource(name="test_source", rating=5))
    session.add(RatedSource(name="dummy"))

    q = select(RatedSource)
    print(session.scalars(q).all())
    print(db.tables.keys())

    assert (
        session.execute(
            select(func.count())
            .select_from(RatedSource)
            .filter(RatedSource.rating == 5)
        ).scalar()
        == 1
    )

    # We made one with default rating (0) and in conftest three are made in advance
    assert (
        session.execute(
            select(func.count())
            .select_from(RatedSource)
            .filter(RatedSource.rating == 0)
        ).scalar()
        == 4
    )


class RatedSourceInCustomTable(Base, BeliefSourceDBMixin):
    """A custom db class for representing belief sources.
    BeliefSource properties are added via the Mixin, so the Base is added by us.
    Here, we can change the table name.
    """

    __tablename__ = "my_belief_source"

    rating = Column(Float(), default=0)


def test_custom_source_with_mixin(db):
    session.add(RatedSourceInCustomTable(name="test_source", rating=7))
    session.add(RatedSourceInCustomTable(name="dummy"))

    assert (
        session.execute(
            select(func.count())
            .select_from(RatedSourceInCustomTable)
            .filter(RatedSourceInCustomTable.rating == 7)
        ).scalar()
        == 1
    )

    assert "my_belief_source" in db.tables.keys()


class JoyfulBeliefInCustomTable(Base, TimedBeliefDBMixin):
    """A custom db class for representing beliefs.
    We overwrite the source_id reference, to our custom source table (see above).
    We also specify the source relationship here, so code can use it.
    """

    __tablename__ = "my_timed_belief"

    happiness = Column(Float(), default=0)

    @declared_attr
    def source_id(cls):
        return Column(Integer, ForeignKey("my_belief_source.id"), primary_key=True)

    source = relationship(
        "RatedSourceInCustomTable", backref=backref("beliefs", lazy=True)
    )

    def __init__(
        self,
        sensor: DBSensor,
        source: DBBeliefSource,
        happiness: float = None,
        **kwargs
    ):
        self.happiness = happiness
        TimedBeliefDBMixin.__init__(self, sensor, source, **kwargs)
        base_kwargs = tb_utils.remove_class_init_kwargs(TimedBeliefDBMixin, kwargs)
        Base.__init__(self, **base_kwargs)


def test_custom_source_and_beliefs_with_mixin(db):
    source = RatedSourceInCustomTable(name="test_source", rating=7)
    session.add(source)

    sensor = DBSensor(name="AnySensor")
    session.add(sensor)

    session.flush()

    now = datetime.now(tz=timezone("Europe/Amsterdam"))
    belief = JoyfulBeliefInCustomTable(
        sensor=sensor,
        source=source,
        belief_time=now,
        event_start=now + timedelta(minutes=3),
        event_value=100,
        happiness=3,
    )
    session.add(belief)

    q = select(RatedSourceInCustomTable).filter(RatedSourceInCustomTable.rating == 7)
    assert (
        session.execute(
            select(func.count())
            .select_from(RatedSourceInCustomTable)
            .filter_by(rating=7)
        ).scalar()
        == 1
    )
    assert session.scalar(q.limit(1)).rating == 7

    q = select(JoyfulBeliefInCustomTable).filter(
        JoyfulBeliefInCustomTable.happiness == 3
    )
    assert (
        session.execute(
            select(func.count())
            .select_from(JoyfulBeliefInCustomTable)
            .filter_by(happiness=3)
        ).scalar()
        == 1
    )
    the_belief = session.scalar(q.limit(1))
    assert the_belief.event_value == belief.event_value
    assert the_belief.sensor.__class__ == DBSensor
    assert the_belief.source.__class__ == RatedSourceInCustomTable
    assert the_belief.source == source

    assert "my_belief_source" in db.tables.keys()
    assert "my_timed_belief" in db.tables.keys()

    bdf = JoyfulBeliefInCustomTable.search_session(session, sensor)
    assert isinstance(bdf, BeliefsDataFrame)
    assert len(bdf) == 1
