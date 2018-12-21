from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, Interval, Float, ForeignKey
from sqlalchemy.orm import relationship, backref, Query

from base import Base


class TimedBelief(Base):
    """"""

    event_start = Column(DateTime(timezone=True), primary_key=True)
    belief_horizon = Column(Interval(), nullable=False, primary_key=True)
    event_value = Column(Float, nullable=False)
    sensor_id = Column(
        Integer(), ForeignKey("sensor.id", ondelete="CASCADE"), primary_key=True
    )
    source_id = Column(Integer, ForeignKey("sources.id"), primary_key=True)
    sensor = relationship(
        "Sensor",
        backref=backref(
            "beliefs",
            lazy=True,
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )
    source = relationship(
        "Source",
        backref=backref(
            "beliefs",
            lazy=True,
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )

    @property
    def event_end(self) -> datetime:
        return self.event_start + self.sensor.event_resolution

    @property
    def knowledge_time(self) -> datetime:
        return self.sensor.knowledge_time(self.event_end)

    @property
    def belief_time(self) -> datetime:
        return self.knowledge_time - self.belief_horizon

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_query(self) -> Query:
        pass
