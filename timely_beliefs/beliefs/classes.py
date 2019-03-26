from typing import List, Optional
from datetime import datetime, timedelta
import math

import pandas as pd
from pandas.tseries.frequencies import to_offset
from sqlalchemy import Column, DateTime, Integer, Interval, Float, ForeignKey
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm import relationship, backref

from timely_beliefs.base import Base, session
from timely_beliefs.sources.classes import BeliefSource, DBBeliefSource
from timely_beliefs.sensors.classes import Sensor, DBSensor
from timely_beliefs.utils import enforce_utc
from timely_beliefs.sensors import utils as sensor_utils
from timely_beliefs.beliefs import utils as belief_utils


class TimedBelief(object):
    """
    The basic description of a data point as a belief, which includes the following:
    - a sensor (what the belief is about)
    - an event (an instant or period of time that the belief is about)
    - a horizon (indicating when the belief was formed with respect to the event)
    - a source (who or what formed the belief)
    - a value (what was believed)
    - a cumulative probability (the likelihood of the value being equal or lower than stated)*

    * The default assumption is that the mean value is given (cp=0.5), but if no beliefs about possible other outcomes
    are given, then this will be treated as a deterministic belief (cp=1). As an alternative to specifying an cumulative
    probability explicitly, you can specify an integer number of standard deviations which is translated
    into a cumulative probability assuming a normal distribution (e.g. sigma=-1 becomes cp=0.1587).
    """

    event_start: datetime
    belief_horizon: timedelta
    event_value: float
    sensor: Sensor
    source: BeliefSource
    cumulative_probability: float

    def __init__(self, sensor: Sensor, source: BeliefSource, value: float, **kwargs):
        self.sensor = sensor
        self.source = source
        self.event_value = value

        if "cumulative_probability" in kwargs:
            self.cumulative_probability = kwargs["cumulative_probability"]
        elif "cp" in kwargs:
            self.cumulative_probability = kwargs["cp"]
        elif "sigma" in kwargs:
            self.cumulative_probability = (
                1 / 2 + (math.erf(kwargs["sigma"] / 2 ** 0.5)) / 2
            )
        else:
            self.cumulative_probability = 0.5
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
    def event_resolution(self) -> timedelta:
        return self.sensor.event_resolution

    @hybrid_property
    def belief_time(self) -> datetime:
        return self.knowledge_time - self.belief_horizon

    @property
    def source_id(self):
        """Convenience method so these and DBTimedBelief can be treated equally"""
        if self.source is not None:
            return self.source.name
        return None


class DBTimedBelief(Base, TimedBelief):
    """Database representation of TimedBelief"""

    __tablename__ = "timed_beliefs"

    event_start = Column(DateTime(timezone=True), primary_key=True)
    belief_horizon = Column(Interval(), nullable=False, primary_key=True)
    cumulative_probability = Column(Float, nullable=False, primary_key=True)
    event_value = Column(Float, nullable=False)
    sensor_id = Column(
        Integer(), ForeignKey("sensor.id", ondelete="CASCADE"), primary_key=True
    )
    source_id = Column(Integer, ForeignKey("belief_source.id"), primary_key=True)
    sensor = relationship(
        "DBSensor",
        backref=backref(
            "beliefs", lazy=True, cascade="all, delete-orphan", passive_deletes=True
        ),
    )
    source = relationship(
        "DBBeliefSource",
        backref=backref(
            "beliefs", lazy=True, cascade="all, delete-orphan", passive_deletes=True
        ),
    )

    def __init__(
        self, sensor: DBSensor, source: DBBeliefSource, value: float, **kwargs
    ):
        TimedBelief.__init__(self, sensor, source, value, **kwargs)
        Base.__init__(self)

    @hybrid_method
    def query(
        self,
        sensor: DBSensor,
        event_before: datetime = None,
        event_not_before: datetime = None,
        belief_before: datetime = None,
        belief_not_before: datetime = None,
    ) -> "BeliefsDataFrame":
        """Query beliefs about sensor events.
        :param sensor: sensor to which the beliefs pertain
        :param event_before: only return beliefs about events that end before this datetime (inclusive)
        :param event_not_before: only return beliefs about events that start after this datetime (inclusive)
        :param belief_before: only return beliefs formed before this datetime (inclusive)
        :param belief_not_before: only return beliefs formed after this datetime (inclusive)
        :returns: a multi-index DataFrame with all relevant beliefs

        TODO: rename params for clarity: event_finished_before, even_starts_not_before (or similar), same for beliefs
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
        event_resolution, knowledge_horizon_fnc, knowledge_horizon_par = (
            session.query(
                DBSensor.event_resolution,
                DBSensor.knowledge_horizon_fnc,
                DBSensor.knowledge_horizon_par,
            )
            .filter(DBSensor.id == sensor.id)
            .one_or_none()
        )

        # Get bounds on the knowledge horizon (so we can already roughly filter by belief time)
        knowledge_horizon_min, knowledge_horizon_max = sensor_utils.eval_verified_knowledge_horizon_fnc(
            knowledge_horizon_fnc, knowledge_horizon_par, None
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
            q = q.filter(
                self.event_start
                <= belief_before + self.belief_horizon + knowledge_horizon_max
            )
        if belief_not_before is not None:
            q = q.filter(
                self.event_start
                >= belief_not_before + self.belief_horizon + knowledge_horizon_min
            )

        # Build our DataFrame of beliefs
        df = BeliefsDataFrame(sensor=sensor, beliefs=q.all())

        # Actually filter by belief time
        if belief_before is not None:
            df = df[df.index.get_level_values("belief_time") < belief_before]
        if belief_not_before is not None:
            df = df[df.index.get_level_values("belief_time") >= belief_not_before]

        return df


class BeliefsSeries(pd.Series):
    """Just for slicing, to keep around the metadata."""

    _metadata = ["sensor", "event_resolution"]

    @property
    def _constructor(self):
        return BeliefsSeries

    @property
    def _constructor_expanddim(self):
        return BeliefsDataFrame

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self."""
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return


class BeliefsDataFrame(pd.DataFrame):
    """Beliefs about a sensor.
    A BeliefsDataFrame object is a pandas.DataFrame with the following specific data columns and MultiIndex levels:

    columns: ["event_value"]
    index levels: ["event_start", "belief_time", "source", "cumulative_probability"]

    Note that source can either be an ID or a name, depending on the beliefs being used (TimedBelief or DBTimedBelief).

    In addition to the standard DataFrame constructor arguments,
    BeliefsDataFrame also accepts the following keyword arguments:

    :param: sensor: the Sensor object that the beliefs pertain to
    :param: beliefs: a list of TimedBelief objects used to initialise the BeliefsDataFrame
    """

    _metadata = ["sensor", "event_resolution"]

    @property
    def _constructor(self):
        return BeliefsDataFrame

    def __init__(self, *args, **kwargs):
        """Initialise a multi-index DataFrame with beliefs about a unique sensor."""

        # Obtain parameters that are specific to our DataFrame subclass
        sensor: Sensor = kwargs.pop("sensor", None)
        beliefs: List[TimedBelief] = kwargs.pop("beliefs", None)

        # Use our constructor if initialising from a previous DataFrame (e.g. when slicing), copying the Sensor metadata
        # TODO: how is the metadata copied here?
        if beliefs is None:
            super().__init__(*args, **kwargs)
            return

        # Define our columns and indices
        columns = ["event_value"]
        indices = ["event_start", "belief_time", "source", "cumulative_probability"]

        # Call the pandas DataFrame constructor with the right input
        kwargs["columns"] = columns
        if beliefs:
            beliefs = sorted(
                beliefs,
                key=lambda b: (
                    b.event_start,
                    b.belief_time,
                    b.source_id,
                    b.cumulative_probability,
                ),
            )
            kwargs["data"] = [[getattr(i, j) for j in columns] for i in beliefs]
            kwargs["index"] = pd.MultiIndex.from_tuples(
                [[getattr(i, j) for j in indices] for i in beliefs], names=indices
            )
        else:
            kwargs["index"] = pd.MultiIndex(
                levels=[[] for _ in indices], codes=[[] for _ in indices], names=indices
            )
        super().__init__(*args, **kwargs)

        # Set the Sensor metadata (including timing properties of the sensor)
        self.sensor = sensor
        self.event_resolution = self.sensor.event_resolution

    def append_from_time_series(
        self,
        event_value_series: pd.Series,
        source: BeliefSource,
        belief_horizon: timedelta,
    ) -> "BeliefsDataFrame":
        """Append beliefs from time series entries into this BeliefsDataFrame. Sensor is assumed to be the same.
        Returns a new BeliefsDataFrame object.
        TODO: enable to add probability data."""
        beliefs = belief_utils.load_time_series(
            event_value_series, self.sensor, source, belief_horizon
        )
        return self.append(BeliefsDataFrame(sensor=self.sensor, beliefs=beliefs))

    @property
    def _constructor_sliced(self):
        return BeliefsSeries

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self."""
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    def convert_index_from_belief_time_to_horizon(self) -> "BeliefsDataFrame":
        return belief_utils.replace_multi_index_level(
            self, "belief_time", self.belief_horizons
        )

    def convert_index_from_event_start_to_end(self) -> "BeliefsDataFrame":
        return belief_utils.replace_multi_index_level(
            self, "event_start", self.event_ends
        )

    @property
    def knowledge_times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            self.event_starts.to_series(keep_tz=True, name="knowledge_time").apply(
                lambda event_start: self.sensor.knowledge_time(event_start)
            )
        )

    @property
    def knowledge_horizons(self) -> pd.TimedeltaIndex:
        return pd.TimedeltaIndex(
            self.event_starts.to_series(keep_tz=True, name="knowledge_horizon").apply(
                lambda event_start: self.sensor.knowledge_horizon(event_start)
            )
        )

    @property
    def belief_times(self) -> pd.DatetimeIndex:
        return self.index.get_level_values("belief_time")

    @property
    def belief_horizons(self) -> pd.TimedeltaIndex:
        return (self.knowledge_times - self.belief_times).rename("belief_horizon")

    @property
    def event_starts(self) -> pd.DatetimeIndex:
        return self.index.get_level_values("event_start")

    @property
    def event_ends(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            self.event_starts.to_series(keep_tz=True, name="event_end").apply(
                lambda event_start: event_start + self.sensor.event_resolution
            )
        )

    @hybrid_method
    def belief_history(self, event_start: datetime) -> "BeliefsDataFrame":
        """Select all beliefs about a single event, identified by the event's start time."""
        return self.xs(enforce_utc(event_start), level="event_start").sort_index()

    @hybrid_method
    def fixed_horizon(self, belief_time: datetime) -> "BeliefsDataFrame" :
        """Select the most recent belief about each event at a given belief time."""
        df = self[self.index.get_level_values("belief_time") <= enforce_utc(belief_time)]
        return belief_utils.select_most_recent_belief(df)

    @hybrid_method
    def rolling_horizon(self, belief_horizon: timedelta) -> "BeliefsDataFrame":
        """Select the most recent belief about each event,
        at least some duration in advance of knowledge time (pass a positive belief_horizon),
        or at most some duration after knowledge time (pass a negative belief_horizon)."""
        df = self.convert_index_from_belief_time_to_horizon()
        df = df[df.index.get_level_values("belief_horizon") >= belief_horizon]
        return belief_utils.select_most_recent_belief(df)

    @hybrid_method
    def resample_events(
        self, event_resolution: timedelta, distribution: Optional[str] = None
    ) -> "BeliefsDataFrame":
        """Aggregate over multiple events (downsample) or split events into multiple sub-events (upsample)."""

        if self.empty:
            return self

        df = (
            self.groupby(
                [
                    pd.Grouper(
                        freq=to_offset(event_resolution).freqstr, level="event_start"
                    ),
                    "source",
                ],
                group_keys=False,
            )
            .apply(
                lambda x: belief_utils.resample_event_start(
                    x,
                    event_resolution,
                    input_resolution=self.event_resolution,
                    distribution=distribution,
                )
            )
            .sort_index()
        )

        # Update metadata with new resolution
        df.event_resolution = event_resolution

        # Put back lost metadata (because groupby statements still tend to lose it)
        df.sensor = self.sensor

        return df
