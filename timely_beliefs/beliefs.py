from datetime import datetime, timedelta

from pandas import DataFrame
from sqlalchemy import Column, DateTime, Integer, Interval, Float, ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm import relationship, backref

from base import Base, session
from timely_beliefs import Sensor
from timely_beliefs import BeliefSource
from timely_beliefs.utils import eval_verified_knowledge_horizon_fnc, enforce_utc, create_beliefs_df


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

    @hybrid_property
    def event_end(self) -> datetime:
        return self.event_start + self.sensor.event_resolution

    @hybrid_property
    def knowledge_time(self) -> datetime:
        return self.sensor.knowledge_time(self.event_start)

    @hybrid_property
    def knowledge_horizon(self) -> timedelta:
        return self.sensor.knowledge_horizon(self.event_start)

    @hybrid_property
    def event_resolution(self) -> datetime:
        return self.sensor.event_resolution

    @hybrid_property
    def belief_time(self) -> datetime:
        return self.knowledge_time - self.belief_horizon

    def __init__(self, sensor: Sensor, source: BeliefSource, value: float, **kwargs):
        self.sensor = sensor
        self.source = source
        self.event_value = value
        if "event_start" in kwargs:
            self.event_start = enforce_utc(kwargs["event_start"])
        elif "event_time" in kwargs:
            if self.sensor.event_resolution != timedelta():
                raise KeyError(
                    "Sensor has a non-zero resolution, so it doesn't measure instantaneous events. "
                    "Use event_start instead of event_time."
                )
            self.event_start = enforce_utc(kwargs["event_time"])
        if "belief_horizon" in kwargs:
            self.belief_horizon = kwargs["belief_horizon"]
        elif "belief_time" in kwargs:
            belief_time = enforce_utc(kwargs["belief_time"])
            self.belief_horizon = (
                self.sensor.knowledge_time(self.event_start) - belief_time
            )

    @hybrid_method
    def query(self, sensor: Sensor, before: datetime = None, not_before: datetime = None) -> DataFrame:
        """Query beliefs about sensor events.
        :param sensor: sensor to which the beliefs pertain
        :param before: only return beliefs formed before this datetime (inclusive)
        :param not_before: only return beliefs formed after this datetime (inclusive)
        :returns: a multi-index DataFrame with all relevant beliefs
        """

        # Check for timezone-aware datetime input
        if before is not None:
            before = enforce_utc(before)
        if not_before is not None:
            not_before = enforce_utc(not_before)

        # Query sensor for relevant timing properties
        event_resolution, knowledge_horizon_fnc, knowledge_horizon_par = session.query(
            Sensor.event_resolution, Sensor.knowledge_horizon_fnc, Sensor.knowledge_horizon_par
        ).filter(Sensor.id == sensor.id).one_or_none()

        # Get bounds on the knowledge horizon (so we can already roughly filter by belief time)
        knowledge_horizon_min, knowledge_horizon_max = eval_verified_knowledge_horizon_fnc(
            knowledge_horizon_fnc,
            knowledge_horizon_par,
            None
        )

        # Query based on start_time_window
        q = session.query(self).filter(self.sensor_id == sensor.id)

        # Apply rough belief time filter
        if before is not None:
            q = q.filter(self.event_start + event_resolution <= before + self.belief_horizon + knowledge_horizon_max)
        if not_before is not None:
            q = q.filter(self.event_start + event_resolution >= not_before + self.belief_horizon + knowledge_horizon_min)

        # Build our DataFrame of beliefs
        df = create_beliefs_df(q.all())

        # Actually filter by belief time
        if before is not None:
            df = df.loc[df["belief_time"] <= before]
        if not_before is not None:
            df = df.loc[df["belief_time"] >= not_before]

        return df
