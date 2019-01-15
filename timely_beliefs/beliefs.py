from typing import List
from datetime import datetime, timedelta

from pandas import DataFrame, MultiIndex
from sqlalchemy import Column, DateTime, Integer, Interval, Float, ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm import relationship, backref

from base import Base, session
from timely_beliefs import Sensor
from timely_beliefs import BeliefSource
from timely_beliefs.utils import eval_verified_knowledge_horizon_fnc, enforce_utc


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
    def belief_percentile(self) -> float:
        """Overwrite in ProbabilisticTimedBelief subclass."""
        return 0.5

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
    def query(
            self,
            sensor: Sensor,
            event_before: datetime = None,
            event_not_before: datetime = None,
            belief_before: datetime = None,
            belief_not_before: datetime = None
    ) -> DataFrame:
        """Query beliefs about sensor events.
        :param sensor: sensor to which the beliefs pertain
        :param event_before: only return beliefs about events that end before this datetime (inclusive)
        :param event_not_before: only return beliefs about events that start after this datetime (inclusive)
        :param belief_before: only return beliefs formed before this datetime (inclusive)
        :param belief_not_before: only return beliefs formed after this datetime (inclusive)
        :returns: a multi-index DataFrame with all relevant beliefs
        """

        # Check for timezone-aware datetime input
        if event_before is not None:
            event_before = enforce_utc(event_before)
        if event_not_before is not None:
            event_not_before = enforce_utc(event_not_before)
        if belief_before is not None:
            belief_before = enforce_utc(belief_before)
        if belief_not_before is not None:
            belief_not_before = enforce_utc(belief_not_before)

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

        # Apply event time filter
        if event_before is not None:
            q = q.filter(self.event_start + event_resolution <= event_before)
        if event_not_before is not None:
            q = q.filter(self.event_start >= event_not_before)

        # Apply rough belief time filter
        if belief_before is not None:
            q = q.filter(self.event_start <= belief_before + self.belief_horizon + knowledge_horizon_max)
        if belief_not_before is not None:
            q = q.filter(self.event_start >= belief_not_before + self.belief_horizon + knowledge_horizon_min)

        # Build our DataFrame of beliefs
        df = BeliefsDataFrame(sensor=sensor, beliefs=q.all())

        # Actually filter by belief time
        if belief_before is not None:
            df = df[df.index.get_level_values("belief_time") <= belief_before]
        if belief_not_before is not None:
            df = df[df.index.get_level_values("belief_time") >= belief_not_before]

        return df


class BeliefsDataFrame(DataFrame):
    """Beliefs about a sensor.
    A BeliefsDataFrame object is a pandas.DataFrame with the following specific data columns and MultiIndex levels:

    columns: ["event_value"]
    index levels: ["event_start", "belief_time", "source_id", "belief_percentile"]

    In addition to the standard DataFrame constructor arguments,
    BeliefsDataFrame also accepts the following keyword arguments:

    :param: sensor: the Sensor object that the beliefs pertain to
    :param: beliefs: a list of TimedBelief objects used to initialise the BeliefsDataFrame
    """

    _metadata = ["sensor"]

    @property
    def _constructor(self):
        return BeliefsDataFrame

    @property
    def convert_index_from_belief_time_to_horizon(self):
        belief_times = self.index.get_level_values("belief_time")
        from_belief_time_to_horizon = {
            i: j for i in belief_times for j in self.knowledge_times - belief_times
        }
        df = self.rename(from_belief_time_to_horizon, axis="index")
        df.index.rename("belief_horizon", level="belief_time", inplace=True)
        return df

    @property
    def knowledge_times(self):
        event_starts = self.index.get_level_values("event_start").to_series(keep_tz=True)
        return event_starts.apply(lambda event_start: self.sensor.knowledge_time(event_start))

    @hybrid_method
    def belief_history(self, event_start):
        return self.xs(event_start, level="event_start")

    @hybrid_method
    def rolling_horizon(self, belief_horizon):
        df = self.convert_index_from_belief_time_to_horizon
        return df[df.index.get_level_values("belief_horizon") >= belief_horizon]

    def __init__(self, *args, **kwargs):
        """Initialise a multi-index DataFrame with beliefs about a unique sensor."""

        # Obtain parameters that are specific to our DataFrame subclass
        sensor: Sensor = kwargs.pop("sensor", None)
        beliefs: List[TimedBelief] = kwargs.pop("beliefs", None)

        # Use our constructor if initialising from a previous DataFrame (e.g. when slicing), copying the Sensor metadata
        if beliefs is None:
            super().__init__(*args, **kwargs)
            return

        # Define our columns and indices
        columns = ["event_value"]
        indices = ["event_start", "belief_time", "source_id", "belief_percentile"]

        # Call the pandas DataFrame constructor with the right input
        kwargs["columns"] = columns
        if beliefs:
            kwargs["data"] = [[getattr(i, j) for j in columns] for i in beliefs]
            kwargs["index"] = MultiIndex.from_tuples([[getattr(i, j) for j in indices] for i in beliefs], names=indices)
        else:
            kwargs["index"] = MultiIndex(levels=[[] for i in indices], labels=[[] for i in indices], names=indices)
        super().__init__(*args, **kwargs)

        # Set the Sensor metadata (including timing properties of the sensor)
        self.sensor = sensor
