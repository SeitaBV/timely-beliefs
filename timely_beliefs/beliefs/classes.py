from typing import Any, Callable, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.core.groupby import DataFrameGroupBy
from sqlalchemy import Column, DateTime, Integer, Interval, Float, ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm import relationship, backref

from base import Base, session
from timely_beliefs import Sensor
from timely_beliefs import BeliefSource
from timely_beliefs.beliefs.utils import get_mean_belief, get_belief_at_cumulative_probability
from timely_beliefs.utils import enforce_utc
from timely_beliefs.sensors import utils as sensor_utils
from timely_beliefs.beliefs import utils as belief_utils


class TimedBelief(Base):
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

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    event_start = Column(DateTime(timezone=True), primary_key=True)
    belief_horizon = Column(Interval(), nullable=False, primary_key=True)
    cumulative_probability = Column(Float, nullable=False, primary_key=True)
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
        self.source_id = source.id
        self.event_value = value
        if "cumulative_probability" in kwargs:
            self.cumulative_probability = kwargs["cumulative_probability"]
        elif "cp" in kwargs:
            self.cumulative_probability = kwargs["cp"]
        elif "sigma" in kwargs:
            self.cumulative_probability = 1/2 + (math.erf(kwargs["sigma"] / 2**0.5))/2
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

    @hybrid_method
    def query(
            self,
            sensor: Sensor,
            event_before: datetime = None,
            event_not_before: datetime = None,
            belief_before: datetime = None,
            belief_not_before: datetime = None,
            source_id: int = None,
    ) -> "BeliefsDataFrame":
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
        knowledge_horizon_min, knowledge_horizon_max = sensor_utils.eval_verified_knowledge_horizon_fnc(
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

        # Apply source filter
        if source_id is not None:
            q = q.filter(self.source_id == source_id)

        # Build our DataFrame of beliefs
        df = BeliefsDataFrame(sensor=sensor, beliefs=q.all())

        # Actually filter by belief time
        if belief_before is not None:
            df = df[df.index.get_level_values("belief_time") <= belief_before]
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
    index levels: ["event_start", "belief_time", "source_id", "cumulative_probability"]

    In addition to the standard DataFrame constructor arguments,
    BeliefsDataFrame also accepts the following keyword arguments:

    :param: sensor: the Sensor object that the beliefs pertain to
    :param: beliefs: a list of TimedBelief objects used to initialise the BeliefsDataFrame
    """

    _metadata = ["sensor", "event_resolution"]

    @property
    def _constructor(self):
        return BeliefsDataFrame

    @property
    def _constructor_sliced(self):
        return BeliefsSeries

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self."""
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    def convert_index_from_belief_time_to_horizon(self) -> "BeliefsDataFrame":
        return belief_utils.replace_multi_index_level(self, "belief_time", self.belief_horizons)

    def convert_index_from_belief_horizon_to_time(self) -> "BeliefsDataFrame":
        return belief_utils.replace_multi_index_level(self, "belief_horizon", self.belief_times)

    def convert_index_from_event_start_to_end(self) -> "BeliefsDataFrame":
        return belief_utils.replace_multi_index_level(self, "event_start", self.event_ends)

    @property
    def knowledge_times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.event_starts.to_series(keep_tz=True, name="knowledge_time").apply(lambda event_start: self.sensor.knowledge_time(event_start)))

    @property
    def knowledge_horizons(self) -> pd.TimedeltaIndex:
        return pd.TimedeltaIndex(self.event_starts.to_series(keep_tz=True, name="knowledge_horizon").apply(lambda event_start: self.sensor.knowledge_horizon(event_start)))

    @property
    def belief_times(self) -> pd.DatetimeIndex:
        if "belief_time" in self.index.names:
            return self.index.get_level_values("belief_time")
        else:
            return (self.knowledge_times - self.belief_horizons).rename("belief_time")

    @property
    def belief_horizons(self) -> pd.TimedeltaIndex:
        if "belief_horizon" in self.index.names:
            return self.index.get_level_values("belief_horizon")
        else:
            return (self.knowledge_times - self.belief_times).rename("belief_horizon")

    @property
    def event_starts(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.index.get_level_values("event_start"))

    @property
    def event_ends(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.event_starts.to_series(keep_tz=True, name="event_end").apply(lambda event_start: event_start + self.event_resolution))

    @hybrid_method
    def for_each_belief(self, fnc: Callable = None, *args: Any, **kwargs: Any) -> Union["BeliefsDataFrame", DataFrameGroupBy]:
        """Convenient function to apply a function to each belief in the BeliefsDataFrame.
        A belief is a group with unique event start, belief time and source id. A deterministic belief is defined by a
        single row, whereas a probabilistic belief is defined by multiple rows.
        If no function is given, return the GroupBy object.

        :Example:

        >>> # Apply some function that accepts a DataFrame, a positional argument and a keyword argument
        >>> df.for_each_belief(some_function, True, a=1)
        >>> # Pipe some other function that accepts a GroupBy object, a positional argument and a keyword argument
        >>> df.for_each_belief().pipe(some_other_function, True, a=1)
        """
        index_names = []
        index_names.append("event_start") if "event_start" in self.index.names else index_names.append("event_end")
        index_names.append("belief_time") if "belief_time" in self.index.names else index_names.append("belief_horizon")
        index_names.append("source_id")
        gr = self.groupby(level=index_names, group_keys=False)
        if fnc is not None:
            return gr.apply(lambda x: fnc(x, *args, **kwargs))
        return gr

    @hybrid_method
    def belief_history(
        self,
        event_start: datetime,
        belief_time_window: Tuple[Optional[datetime], Optional[datetime]] = (None, None),
        belief_horizon_window: Tuple[Optional[timedelta], Optional[timedelta]] = (None, None),
    ) -> "BeliefsDataFrame":
        """Select all beliefs about a single event, identified by the event's start time.
        Optionally select a history of beliefs formed within a certain time window.
        Alternatively, select a history of beliefs formed a certain horizon window before knowledge time (with negative
        horizons indicating post knowledge time).

        :Example:

        >>> # Window selecting beliefs formed before June 20th 2018
        >>> df.belief_history(event_start, belief_time_window=(None, datetime(2018, 6, 20)))
        >>> # Window selecting beliefs formed from 5 to 10 hours before knowledge time
        >>> df.belief_history(event_start, belief_horizon_window=(timedelta(hours=5), timedelta(hours=10)))
        >>> # Window selecting beliefs formed from 2 hours after to 10 hours before knowledge time
        >>> df.belief_history(event_start, belief_horizon_window=(timedelta(hours=-2), timedelta(hours=10)))

        :param: event_start: start time of the event
        :param: belief_time_window: optional tuple specifying a time window within which beliefs should have been formed
        :param: belief_horizon_window: optional tuple specifying a horizon window (e.g. between 1 and 2 hours before the event value could have been known)
        """
        df = self.xs(enforce_utc(event_start), level="event_start", drop_level=False).sort_index()
        if belief_time_window[0] is not None:
            df = df[df.index.get_level_values("belief_time") >= belief_time_window[0]]
        if belief_time_window[1] is not None:
            df = df[df.index.get_level_values("belief_time") <= belief_time_window[1]]
        if belief_horizon_window != (None, None):
            if belief_time_window != (None, None):
                raise ValueError("Cannot pass both a belief time window and belief horizon window.")
            df = df.convert_index_from_belief_time_to_horizon()
            if belief_horizon_window[0] is not None:
                df = df[df.index.get_level_values("belief_horizon") >= belief_horizon_window[0]]
            if belief_horizon_window[1] is not None:
                df = df[df.index.get_level_values("belief_horizon") <= belief_horizon_window[1]]
            df = df.convert_index_from_belief_horizon_to_time()
        df.index = df.index.droplevel("event_start")
        return df

    @hybrid_method
    def fixed_horizon(
        self,
        belief_time: datetime = None,
        belief_time_window: Tuple[Optional[datetime], Optional[datetime]] = (None, None),
    ) -> "BeliefsDataFrame":
        """Select the most recent belief about each event at a given belief time.
        Alternatively, select the most recent belief formed within a certain time window. This allows setting a maximum
        freshness of the data.

        :Example:

        >>> # Time selecting the latest beliefs formed before June 6th 2018 about each event
        >>> df.fixed_horizon(belief_time=datetime(2018, 6, 6))
        >>> # Or equivalently:
        >>> df.fixed_horizon(belief_time_window=(None, datetime(2018, 6, 6)))
        >>> # Time window selecting the latest beliefs formed from June 1st to June 6th (up to June 6th 0:00 AM)
        >>> df.fixed_horizon(belief_time_window=(datetime(2018, 6, 1), datetime(2018, 6, 6)))

        :param: belief_time: datetime indicating the belief should be formed at least before this time
        :param: belief_time_window: optional tuple specifying a time window within which beliefs should have been formed
        """
        if belief_time is not None:
            if belief_time_window != (None, None):
                raise ValueError("Cannot pass both a belief time and belief time window.")
            belief_time_window = (None, belief_time)
        df = self
        if belief_time_window[0] is not None:
            df = df[df.index.get_level_values("belief_time") >= enforce_utc(belief_time_window[0])]
        if belief_time_window[1] is not None:
            df = df[df.index.get_level_values("belief_time") <= enforce_utc(belief_time_window[1])]
        return belief_utils.select_most_recent_belief(df)

    @hybrid_method
    def rolling_horizon(
        self,
        belief_horizon: timedelta = None,
        belief_horizon_window: Tuple[Optional[timedelta], Optional[timedelta]] = (None, None),
    ) -> "BeliefsDataFrame":
        """Select the most recent belief about each event,
        at least some duration in advance of knowledge time (pass a positive belief_horizon),
        or at most some duration after knowledge time (pass a negative belief_horizon).
        Alternatively, select the most recent belief formed within a certain horizon window before knowledge time (with
        negative horizons indicating post knowledge time). This allows setting a maximum acceptable freshness of the
        data.

        :Example:

        >>> # Horizon selecting the latest belief formed about each event at least 1 day beforehand
        >>> df.rolling_horizon(belief_horizon=timedelta(days=1))
        >>> # Or equivalently:
        >>> df.rolling_horizon(belief_horizon_window=(timedelta(days=1), None))
        >>> # Horizon window selecting the latest belief formed about each event at least 1 day, but at most 2 days, beforehand
        >>> df.rolling_horizon(belief_horizon_window=(timedelta(days=1), timedelta(days=2)))

        :param: belief_horizon: timedelta indicating the belief should be formed at least this duration before knowledge time
        :param: belief_horizon_window: optional tuple specifying a horizon window (e.g. between 1 and 2 days before the event value could have been known)
        """
        if belief_horizon is not None:
            if belief_horizon_window != (None, None):
                raise ValueError("Cannot pass both a belief horizon and belief horizon window.")
            belief_horizon_window = (belief_horizon, None)
        df = self.convert_index_from_belief_time_to_horizon()
        if belief_horizon_window[0] is not None:
            df = df[df.index.get_level_values("belief_horizon") >= belief_horizon_window[0]]
        if belief_horizon_window[1] is not None:
            df = df[df.index.get_level_values("belief_horizon") <= belief_horizon_window[1]]
        return belief_utils.select_most_recent_belief(df)

    @hybrid_method
    def resample_events(self, event_resolution: timedelta, distribution: Optional[str] = None) -> "BeliefsDataFrame":
        """Aggregate over multiple events (downsample) or split events into multiple sub-events (upsample)."""

        if self.empty:
            return self

        df = self.groupby(
            [pd.Grouper(freq=to_offset(event_resolution).freqstr, level="event_start"), "source_id"], group_keys=False
        ).apply(lambda x: belief_utils.resample_event_start(x, event_resolution, input_resolution=self.event_resolution, distribution=distribution)).sort_index()

        # Update metadata with new resolution
        df.event_resolution = event_resolution

        # Put back lost metadata (because groupby statements still tend to lose it)
        df.sensor = self.sensor

        return df

    def rolling_horizon_accuracy(
        self,
        belief_horizon: timedelta = None,
        belief_horizon_window: Tuple[Optional[timedelta], Optional[timedelta]] = (None, None),
        belief_horizon_anchor: timedelta = None,
        source_id_anchor: int = None,
    ) -> "BeliefsDataFrame":
        """Get the accuracy of beliefs at a given horizon, with respect to the most recent beliefs about each event.
        By default the mean absolute error (MAE) is returned.
        Alternatively, select the accuracy of beliefs formed within a certain horizon window before knowledge time (with
        negative horizons indicating post knowledge time). This allows setting a maximum acceptable freshness of the
        data.
        Optionally, set an anchor to get the accuracy of beliefs at a given horizon with respect to some other horizon,
        and/or another source.
        This allows to define what is considered to be true at a certain time after an event.

        :Example:

        >>> # Horizon selecting the accuracy of beliefs formed about each event at least 1 day beforehand
        >>> df.rolling_horizon_accuracy(belief_horizon=timedelta(days=1))
        >>> # Or equivalently:
        >>> df.rolling_horizon_accuracy(belief_horizon_window=(timedelta(days=1), None))
        >>> # Horizon window selecting the accuracy of beliefs formed about each event at least 1 day, but at most 2 days, beforehand
        >>> df.rolling_horizon_accuracy(belief_horizon_window=(timedelta(days=1), timedelta(days=2)))
        >>> # Horizon and horizon anchor selecting the accuracy of beliefs formed 10 days beforehand with respect to 1 day past knowledge time
        >>> df.rolling_horizon_accuracy(belief_horizon=timedelta(days=10), belief_horizon_anchor=timedelta(days=-1))
        >>> # Horizon and source anchor selecting the accuracy of beliefs formed 10 days beforehand with respect to the latest belief formed by source 3
        >>> df.rolling_horizon_accuracy(belief_horizon=timedelta(days=10), source_id_anchor=3)

        :param: belief_horizon: timedelta indicating the belief should be formed at least this duration before knowledge time
        :param: belief_horizon_window: optional tuple specifying a horizon window (e.g. between 1 and 2 days before the event value could have been known)
        :param: belief_horizon_anchor: optional timedelta to indicate that the accuracy should be determined with respect to the latest belief at this duration past knowledge time
        :param: source_id_anchor: optional integer to indicate that the accuracy should be determined with respect to the beliefs held by the source with the given id
        """
        df = self
        df_forecast = df.rolling_horizon(belief_horizon, belief_horizon_window)
        if belief_horizon_anchor is None:
            df_true = belief_utils.select_most_recent_belief(df)
        else:
            df_true = belief_utils.select_most_recent_belief(df[df.index.get_level_values("belief_horizon") >= belief_horizon_anchor])

        # Todo: allow to estimate the mean from the given cdf points for non-discrete distributions, e.g. uniform
        # Todo: compute the ranked probability score for probabilistic forecasts
        df_forecast = df_forecast.for_each_belief(get_mean_belief)
        df_true = df_true.for_each_belief(get_belief_at_cumulative_probability, cumulative_probability=0.5)
        df_forecast.index = df_forecast.index.droplevel("belief_horizon")
        df_true.index = df_true.index.droplevel("belief_horizon")
        df_forecast.columns = ["forecast"]
        df_true.columns = ["true"]
        df = pd.concat([df_forecast, df_true], axis=1)

        def decide_who_is_right(df: "BeliefsDataFrame", right_source_id: int) -> "BeliefsDataFrame":
            if right_source_id in df.index.get_level_values(level="source_id").values:
                df["true"] = df["true"].xs(right_source_id, level="source_id").values[0]
            else:
                raise KeyError("Source id %s not found in BeliefsDataFrame." % right_source_id)
            return df

        if source_id_anchor is not None:
            df = df.groupby(level="event_start").apply(lambda x: decide_who_is_right(x, source_id_anchor))

        def calculate_mae(df: "BeliefsDataFrame") -> np.ndarray:
            return np.mean(abs(df["forecast"] - df["true"]))

        mae = df.groupby(level="source_id").apply(lambda x: calculate_mae(x))
        mae.name = "mean_average_error"
        return mae

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
        indices = ["event_start", "belief_time", "source_id", "cumulative_probability"]

        # Call the pandas DataFrame constructor with the right input
        kwargs["columns"] = columns
        if beliefs:
            beliefs = sorted(beliefs, key=lambda b: (b.event_start, b.belief_time, b.source_id, b.cumulative_probability))
            kwargs["data"] = [[getattr(i, j) for j in columns] for i in beliefs]
            kwargs["index"] = pd.MultiIndex.from_tuples([[getattr(i, j) for j in indices] for i in beliefs], names=indices)
        else:
            kwargs["index"] = pd.MultiIndex(levels=[[] for i in indices], labels=[[] for i in indices], names=indices)
        super().__init__(*args, **kwargs)

        # Set the Sensor metadata (including timing properties of the sensor)
        self.sensor = sensor
        self.event_resolution = self.sensor.event_resolution
