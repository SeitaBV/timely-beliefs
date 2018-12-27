from datetime import datetime, timedelta

from sqlalchemy import Column, DateTime, Integer, Interval, Float, ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref, Query

from base import Base
from timely_beliefs.sensors import Sensor
from timely_beliefs.sources import BeliefSource


class TimedBelief(Base):
    """"""

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    event_start = Column(DateTime(timezone=True), primary_key=True)
    belief_horizon = Column(Interval(), nullable=False, primary_key=True)
    event_value = Column(Float, nullable=False)
    sensor_id = Column(
        Integer(), ForeignKey("sensor.id", ondelete="CASCADE"), primary_key=True
    )
    source_id = Column(Integer, ForeignKey("belief_source.id"), primary_key=True)
    sensor = relationship(
        "Sensor",
        backref=backref(
            "beliefs", lazy=True, cascade="all, delete-orphan", passive_deletes=True
        ),
    )
    source = relationship(
        "BeliefSource",
        backref=backref(
            "beliefs", lazy=True, cascade="all, delete-orphan", passive_deletes=True
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

    def __init__(self, sensor: Sensor, source: BeliefSource, value: float, **kwargs):
        self.sensor = sensor
        self.source = source
        self.event_value = value
        if "event_start" in kwargs:
            self.event_start = kwargs["event_start"]
        elif "event_time" in kwargs:
            if self.sensor.event_resolution != timedelta():
                raise KeyError(
                    "Sensor has a non-zero resolution, so it doesn't measure instantaneous events. "
                    "Use event_start instead of event_time."
                )
            self.event_start = kwargs["event_time"]
        if "belief_horizon" in kwargs:
            self.belief_horizon = kwargs["belief_horizon"]
        elif "belief_time" in kwargs:
            self.belief_horizon = (
                self.sensor.knowledge_time(self.event_end) - kwargs["belief_time"]
            )

    def make_query(self) -> Query:
        pass
