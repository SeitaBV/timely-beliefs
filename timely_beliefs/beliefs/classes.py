from typing import Any, Callable, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import math

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from pandas.tseries.frequencies import to_offset
from sqlalchemy import Column, DateTime, Integer, Interval, Float, ForeignKey
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm import relationship, backref
import altair as alt

from timely_beliefs.base import Base, session
from timely_beliefs.sources.classes import BeliefSource, DBBeliefSource
from timely_beliefs.sensors.classes import Sensor, DBSensor
import timely_beliefs.utils as tb_utils
from timely_beliefs.sensors import utils as sensor_utils
from timely_beliefs.beliefs import utils as belief_utils
from timely_beliefs.visualization import utils as visualization_utils


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
            self.event_start = tb_utils.enforce_utc(kwargs["event_start"])
        elif "event_time" in kwargs:
            if self.sensor.event_resolution != timedelta():
                raise KeyError(
                    "Sensor has a non-zero resolution, so it doesn't measure instantaneous events. "
                    "Use event_start instead of event_time."
                )
            self.event_start = tb_utils.enforce_utc(kwargs["event_time"])
        if "belief_horizon" in kwargs:
            self.belief_horizon = kwargs["belief_horizon"]
        elif "belief_time" in kwargs:
            belief_time = tb_utils.enforce_utc(kwargs["belief_time"])
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
        source: Union[int, List[int], str, List[str]] = None,
    ) -> "BeliefsDataFrame":
        """Query beliefs about sensor events.
        :param sensor: sensor to which the beliefs pertain
        :param event_before: only return beliefs about events that end before this datetime (inclusive)
        :param event_not_before: only return beliefs about events that start after this datetime (inclusive)
        :param belief_before: only return beliefs formed before this datetime (inclusive)
        :param belief_not_before: only return beliefs formed after this datetime (inclusive)
        :param source: only return beliefs formed by the given source or list of sources (pass their id or name)
        :returns: a multi-index DataFrame with all relevant beliefs

        TODO: rename params for clarity: event_finished_before, even_starts_not_before (or similar), same for beliefs
        """

        # Check for timezone-aware datetime input
        if event_before is not None:
            event_before = tb_utils.enforce_utc(event_before)
        if event_not_before is not None:
            event_not_before = tb_utils.enforce_utc(event_not_before)
        if belief_before is not None:
            belief_before = tb_utils.enforce_utc(belief_before)
        if belief_not_before is not None:
            belief_not_before = tb_utils.enforce_utc(belief_not_before)

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

        # Apply source filter
        if source is not None:
            source_list = [source] if not isinstance(source, list) else source
            id_list = [s for s in source_list if isinstance(s, int)]
            name_list = [s for s in source_list if isinstance(s, str)]
            if len(id_list) + len(name_list) < len(source_list):
                unidentifiable_list = [
                    s
                    for s in source_list
                    if not isinstance(s, int) and not isinstance(s, str)
                ]
                raise ValueError(
                    "Query by source failed: query only possible by integer id or string name. Failed sources: %s"
                    % unidentifiable_list
                )
            else:
                q = q.join(DBBeliefSource).filter(
                    (self.source_id.in_(id_list)) | (DBBeliefSource.name.in_(name_list))
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

    :param sensor: the Sensor object that the beliefs pertain to
    :param beliefs: a list of TimedBelief objects used to initialise the BeliefsDataFrame
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
                    b.source,
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
            )  # Todo support pandas 0.23
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
        return tb_utils.replace_multi_index_level(
            self, "belief_time", self.belief_horizons
        )

    def convert_index_from_belief_horizon_to_time(self) -> "BeliefsDataFrame":
        return tb_utils.replace_multi_index_level(
            self, "belief_horizon", self.belief_times
        )

    def convert_index_from_event_start_to_end(self) -> "BeliefsDataFrame":
        return tb_utils.replace_multi_index_level(self, "event_start", self.event_ends)

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
        return pd.DatetimeIndex(
            self.event_starts.to_series(keep_tz=True, name="event_end").apply(
                lambda event_start: event_start + self.event_resolution
            )
        )

    @hybrid_method
    def for_each_belief(
        self, fnc: Callable = None, *args: Any, **kwargs: Any
    ) -> Union["BeliefsDataFrame", DataFrameGroupBy]:
        """Convenient function to apply a function to each belief in the BeliefsDataFrame.
        A belief is a group with unique event start, belief time and source. A deterministic belief is defined by a
        single row, whereas a probabilistic belief is defined by multiple rows.
        If no function is given, return the GroupBy object.

        :Example:

        >>> # Apply some function that accepts a DataFrame, a positional argument and a keyword argument
        >>> df.for_each_belief(some_function, True, a=1)
        >>> # Pipe some other function that accepts a GroupBy object, a positional argument and a keyword argument
        >>> df.for_each_belief().pipe(some_other_function, True, a=1)
        >>> # If you want to call this method within another groupby function, pass the df group explicitly
        >>> df.for_each_belief(some_function, True, a=1, df=df)
        """
        df = kwargs.pop("df", self)
        index_names = []
        index_names.extend(
            ["event_start"]
            if "event_start" in df.index.names
            else ["event_end"]
            if "event_end" in df.index.names
            else []
        )
        index_names.extend(
            ["belief_time"]
            if "belief_time" in df.index.names
            else ["belief_horizon"]
            if "belief_horizon" in df.index.names
            else []
        )
        index_names.append("source")
        gr = df.groupby(level=index_names, group_keys=False)
        if fnc is not None:
            return gr.apply(lambda x: fnc(x, *args, **kwargs))
        return gr

    @hybrid_method
    def belief_history(
        self,
        event_start: datetime,
        belief_time_window: Tuple[Optional[datetime], Optional[datetime]] = (
            None,
            None,
        ),
        belief_horizon_window: Tuple[Optional[timedelta], Optional[timedelta]] = (
            None,
            None,
        ),
    ) -> "BeliefsDataFrame":
        """Select all beliefs about a single event, identified by the event's start time.
        Optionally select a history of beliefs formed within a certain time window.
        Alternatively, select a history of beliefs formed a certain horizon window before knowledge time (with negative
        horizons indicating post knowledge time).

        :Example:

        >>> # Select beliefs formed before June 20th 2018
        >>> df.belief_history(event_start, belief_time_window=(None, datetime(2018, 6, 20, tzinfo=utc)))
        >>> # Select beliefs formed from 5 to 10 hours before knowledge time
        >>> df.belief_history(event_start, belief_horizon_window=(timedelta(hours=5), timedelta(hours=10)))
        >>> # Select beliefs formed from 2 hours after to 10 hours before knowledge time
        >>> df.belief_history(event_start, belief_horizon_window=(timedelta(hours=-2), timedelta(hours=10)))

        :param event_start: start time of the event
        :param belief_time_window: optional tuple specifying a time window within which beliefs should have been formed
        :param belief_horizon_window: optional tuple specifying a horizon window (e.g. between 1 and 2 hours before the event value could have been known)
        """
        df = self.xs(
            tb_utils.enforce_utc(event_start), level="event_start", drop_level=False
        ).sort_index()
        if belief_time_window[0] is not None:
            df = df[df.index.get_level_values("belief_time") >= belief_time_window[0]]
        if belief_time_window[1] is not None:
            df = df[df.index.get_level_values("belief_time") <= belief_time_window[1]]
        if belief_horizon_window != (None, None):
            if belief_time_window != (None, None):
                raise ValueError(
                    "Cannot pass both a belief time window and belief horizon window."
                )
            df = df.convert_index_from_belief_time_to_horizon()
            if belief_horizon_window[0] is not None:
                df = df[
                    df.index.get_level_values("belief_horizon")
                    >= belief_horizon_window[0]
                ]
            if belief_horizon_window[1] is not None:
                df = df[
                    df.index.get_level_values("belief_horizon")
                    <= belief_horizon_window[1]
                ]
            df = df.convert_index_from_belief_horizon_to_time()
        df.index = df.index.droplevel("event_start")
        return df

    @hybrid_method
    def fixed_viewpoint(
        self,
        belief_time: datetime = None,
        belief_time_window: Tuple[Optional[datetime], Optional[datetime]] = (
            None,
            None,
        ),
    ) -> "BeliefsDataFrame":
        """Select the most recent belief about each event at a given belief time.
        NB: with a fixed viewpoint the horizon increases as you look further ahead.
        Alternatively, select the most recent belief formed within a certain time window. This allows setting a maximum
        freshness of the data.

        :Example:

        >>> # Select the latest beliefs formed before June 6th 2018 about each event
        >>> df.fixed_viewpoint(belief_time=datetime(2018, 6, 6))
        >>> # Or equivalently:
        >>> df.fixed_viewpoint(belief_time_window=(None, datetime(2018, 6, 6, tzinfo=utc)))
        >>> # Select the latest beliefs formed from June 1st to June 6th (up to June 6th 0:00 AM)
        >>> df.fixed_viewpoint(belief_time_window=(datetime(2018, 6, 1, tzinfo=utc), datetime(2018, 6, 6, tzinfo=utc)))

        :param belief_time: datetime indicating the belief should be formed at least before this time
        :param belief_time_window: optional tuple specifying a time window within which beliefs should have been formed
        """
        if belief_time is not None:
            if belief_time_window != (None, None):
                raise ValueError(
                    "Cannot pass both a belief time and belief time window."
                )
            belief_time_window = (None, belief_time)
        df = self
        if "belief_time" not in df.index.names:
            df = df.convert_index_from_belief_horizon_to_time()
        if belief_time_window[0] is not None:
            df = df[
                df.index.get_level_values("belief_time")
                >= tb_utils.enforce_utc(belief_time_window[0])
            ]
        if belief_time_window[1] is not None:
            df = df[
                df.index.get_level_values("belief_time")
                <= tb_utils.enforce_utc(belief_time_window[1])
            ]
        return belief_utils.select_most_recent_belief(df)

    @hybrid_method
    def rolling_viewpoint(
        self,
        belief_horizon: timedelta = None,
        belief_horizon_window: Tuple[Optional[timedelta], Optional[timedelta]] = (
            None,
            None,
        ),
    ) -> "BeliefsDataFrame":
        """Select the most recent belief about each event,
        at least some duration in advance of knowledge time (pass a positive belief_horizon),
        or at most some duration after knowledge time (pass a negative belief_horizon).
        NB: with a rolling viewpoint the horizon stays the same, because your viewpoint moves with you as you look further ahead.
        Alternatively, select the most recent belief formed within a certain horizon window before knowledge time (with
        negative horizons indicating post knowledge time). This allows setting a maximum acceptable freshness of the
        data.

        :Example:

        >>> # Select the latest belief formed about each event at least 1 day beforehand
        >>> df.rolling_viewpoint(belief_horizon=timedelta(days=1))
        >>> # Or equivalently:
        >>> df.rolling_viewpoint(belief_horizon_window=(timedelta(days=1), None))
        >>> # Select the latest belief formed about each event at least 1 day, but at most 2 days, beforehand
        >>> df.rolling_viewpoint(belief_horizon_window=(timedelta(days=1), timedelta(days=2)))

        :param belief_horizon: timedelta indicating the belief should be formed at least this duration before knowledge time
        :param belief_horizon_window: optional tuple specifying a horizon window (e.g. between 1 and 2 days before the event value could have been known)
        """
        if belief_horizon is not None:
            if belief_horizon_window != (None, None):
                raise ValueError(
                    "Cannot pass both a belief horizon and belief horizon window."
                )
            belief_horizon_window = (belief_horizon, None)
        df = self
        if "belief_horizon" not in df.index.names:
            df = df.convert_index_from_belief_time_to_horizon()
        if belief_horizon_window[0] is not None:
            df = df[
                df.index.get_level_values("belief_horizon") >= belief_horizon_window[0]
            ]
        if belief_horizon_window[1] is not None:
            df = df[
                df.index.get_level_values("belief_horizon") <= belief_horizon_window[1]
            ]
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

    def accuracy(
        self,
        t: Union[datetime, timedelta] = None,
        reference_source: "BeliefSource" = None,
        keep_reference_observation: bool = False,
    ) -> "BeliefsDataFrame":
        """Simply get the accuracy of beliefs about events, at a given time (pass a datetime), at a given horizon
                (pass a timedelta), or as a function of horizon (the default).

        By default the accuracy is determined with respect to the most recent beliefs held by the same source.
        Optionally, set a reference source to determine accuracy with respect to beliefs held by a specific source.

        By default the accuracy is determined with respect to the most recent beliefs.

        By default the mean absolute error (MAE), the mean absolute percentage error (MAPE) and
        the weighted absolute percentage error (WAPE) are returned.

        For more options, use df.fixed_viewpoint_accuracy() or df.rolling_viewpoint_accuracy() instead.

        :param reference_source: optional BeliefSource to indicate that the accuracy should be determined with respect to the beliefs held by the given source
        :param keep_reference_observation: Set to True to return the reference observation used to calculate mape and wape as a DataFrame column
        """

        if t is None:
            return pd.concat(
                [
                    pd.concat(
                        [
                            self.rolling_viewpoint_accuracy(
                                h,
                                reference_source=reference_source,
                                keep_reference_observation=keep_reference_observation,
                            )
                        ],
                        keys=[h],
                        names=["belief_horizon"],
                    )
                    for h in self.lineage.belief_horizons
                ]
            )
        elif isinstance(t, datetime):
            return self.fixed_viewpoint_accuracy(
                t,
                reference_source=reference_source,
                keep_reference_observation=keep_reference_observation,
            )
        elif isinstance(t, timedelta):
            return self.rolling_viewpoint_accuracy(
                t,
                reference_source=reference_source,
                keep_reference_observation=keep_reference_observation,
            )

    def fixed_viewpoint_accuracy(
        self,
        belief_time: datetime = None,
        belief_time_window: Tuple[Optional[datetime], Optional[datetime]] = (
            None,
            None,
        ),
        reference_belief_time: datetime = None,
        reference_belief_horizon: timedelta = None,
        reference_source: int = None,
        keep_reference_observation: bool = False,
    ) -> "BeliefsDataFrame":
        """Get the accuracy of beliefs about events at a given time.

        Alternatively, select the accuracy of beliefs formed within a certain time window. This allows setting a maximum
        acceptable freshness of the data.

        By default the accuracy is determined with respect to the most recent beliefs held by the same source.
        Optionally, set a reference belief time to determine accuracy with respect to beliefs at a specific time.
        Alternatively, set a reference belief horizon instead of a reference belief time.
        Optionally, set a reference source to determine accuracy with respect to beliefs held by a specific source.
        These allow to define what is considered to be true at a certain time.

        By default the mean absolute error (MAE), the mean absolute percentage error (MAPE) and
        the weighted absolute percentage error (WAPE) are returned.

        :Example:

        >>> from datetime import datetime
        >>> from pytz import utc
        >>> from timely_beliefs.tests import example_df as df
        >>> # Select the accuracy of beliefs held about each event on June 2nd (midnight)
        >>> df.fixed_viewpoint_accuracy(belief_time=datetime(2013, 6, 2, tzinfo=utc))
        >>> # Or equivalently:
        >>> df.fixed_viewpoint_accuracy(belief_time_window=(None, datetime(2013, 6, 2, tzinfo=utc)))
        >>> # Select the accuracy of beliefs formed about each event on June 1st
        >>> df.fixed_viewpoint_accuracy(belief_time_window=(datetime(2013, 6, 1, tzinfo=utc), datetime(2013, 6, 2, tzinfo=utc)))
        >>> # Select the accuracy of beliefs held on June 2nd with respect to beliefs held on June 10th
        >>> df.fixed_viewpoint_accuracy(belief_time=datetime(2013, 6, 2, tzinfo=utc), reference_belief_time=datetime(2013, 6, 10, tzinfo=utc))
        >>> # Select the accuracy of beliefs held on June 2nd with respect to 1 day past knowledge time
        >>> df.fixed_viewpoint_accuracy(belief_time=datetime(2013, 6, 2, tzinfo=utc), reference_belief_horizon=timedelta(days=-1))
        >>> # Select the accuracy of beliefs held on June 2nd with respect to the latest belief formed by the first source
        >>> df.fixed_viewpoint_accuracy(belief_time=datetime(2013, 6, 2, tzinfo=utc), reference_source=df.lineage.sources[0])
                                       mae      mape      wape
        source
        <BeliefSource Source A>    0.00000  0.000000  0.000000
        <BeliefSource Source B>  125.85325  0.503413  0.503413

        :param belief_time: datetime indicating the belief should be formed at this time at the latest
        :param belief_time_window: optional tuple specifying a time window in which the belief should have been formed (e.g. between June 1st and 2nd)
        :param reference_belief_time: optional datetime to indicate that the accuracy should be determined with respect to the latest belief held at this time
        :param reference_belief_horizon: optional timedelta to indicate that the accuracy should be determined with respect to the latest belief at this duration past knowledge time
        :param reference_source: optional BeliefSource to indicate that the accuracy should be determined with respect to the beliefs held by the given source
        :param keep_reference_observation: Set to True to return the reference observation used to calculate mape and wape as a DataFrame column
        :returns: BeliefsDataFrame with columns for mae, mape and wape (and optionally, the reference values), indexed by source only
        """
        df = self
        df_forecast = df.fixed_viewpoint(belief_time, belief_time_window)
        if reference_belief_time is None:
            if reference_belief_horizon is None:
                df_observation = belief_utils.select_most_recent_belief(df)
            else:
                df = df.convert_index_from_belief_time_to_horizon()
                df_observation = belief_utils.select_most_recent_belief(
                    df[
                        df.index.get_level_values("belief_horizon")
                        >= reference_belief_horizon
                    ]
                )
        else:
            if reference_belief_horizon is None:
                df = df.convert_index_from_belief_horizon_to_time()
                df_observation = belief_utils.select_most_recent_belief(
                    df[
                        df.index.get_level_values("belief_time")
                        <= reference_belief_time
                    ]
                )
            else:
                raise ValueError(
                    "Cannot pass both a reference belief time and a reference belief horizon."
                )
        return belief_utils.compute_accuracy_scores(
            df_forecast, df_observation, reference_source, keep_reference_observation
        )

    def rolling_viewpoint_accuracy(
        self,
        belief_horizon: timedelta = None,
        belief_horizon_window: Tuple[Optional[timedelta], Optional[timedelta]] = (
            None,
            None,
        ),
        reference_belief_horizon: timedelta = None,
        reference_source: int = None,
        keep_reference_observation: bool = False,
    ) -> "BeliefsDataFrame":
        """Get the accuracy of beliefs about events at a given horizon.

        Alternatively, set a horizon window to select the accuracy of beliefs formed within a certain time window before
        knowledge time (with negative horizons indicating post knowledge time).
        This allows setting a maximum acceptable freshness of the data.

        By default the accuracy is determined with respect to the most recent beliefs held by the same source.
        Optionally, set a reference belief horizon to determine accuracy with respect to beliefs at a specific horizon.
        Optionally, set a reference source to determine accuracy with respect to beliefs held by a specific source.
        These allow to define what is considered to be true at a certain time after an event.

        By default the mean absolute error (MAE), the mean absolute percentage error (MAPE) and
        the weighted absolute percentage error (WAPE) are returned.

        :Example:

        >>> # Select the accuracy of beliefs formed about each event at least 1 day beforehand
        >>> df.rolling_viewpoint_accuracy(belief_horizon=timedelta(days=1))
        >>> # Or equivalently:
        >>> df.rolling_viewpoint_accuracy(belief_horizon_window=(timedelta(days=1), None))
        >>> # Select the accuracy of beliefs formed about each event at least 1 day, but at most 2 days, beforehand
        >>> df.rolling_viewpoint_accuracy(belief_horizon_window=(timedelta(days=1), timedelta(days=2)))
        >>> # Select the accuracy of beliefs formed 10 days beforehand with respect to 1 day past knowledge time
        >>> df.rolling_viewpoint_accuracy(belief_horizon=timedelta(days=10), reference_belief_horizon=timedelta(days=-1))
        >>> # Select the accuracy of beliefs formed 10 days beforehand with respect to the latest belief formed by source 3
        >>> df.rolling_viewpoint_accuracy(belief_horizon=timedelta(days=10), reference_source=3)

        :param belief_horizon: timedelta indicating the belief should be formed at least this duration before knowledge time
        :param belief_horizon_window: optional tuple specifying a horizon window (e.g. between 1 and 2 days before the event value could have been known)
        :param reference_belief_horizon: optional timedelta to indicate that the accuracy should be determined with respect to the latest belief at this duration past knowledge time
        :param reference_source: optional BeliefSource to indicate that the accuracy should be determined with respect to the beliefs held by the given source
        :param keep_reference_observation: Set to True to return the reference observation used to calculate mape and wape as a DataFrame column
        """
        df = self
        df_forecast = df.rolling_viewpoint(belief_horizon, belief_horizon_window)
        if reference_belief_horizon is None:
            df_observation = belief_utils.select_most_recent_belief(df)
        else:
            df_observation = belief_utils.select_most_recent_belief(
                df[
                    df.index.get_level_values("belief_horizon")
                    >= reference_belief_horizon
                ]
            )
        return belief_utils.compute_accuracy_scores(
            df_forecast, df_observation, reference_source, keep_reference_observation
        )

    def plot(self, reference_source: BeliefSource) -> alt.LayerChart:
        """Visualize the BeliefsDataFrame in an interactive Altair chart.

        :param reference_source: BeliefSource to indicate that the accuracy should be determined with respect to the beliefs held by the given source
        :returns: Altair chart object with a vega-lite representation (for more information, see reference below).

        >>> chart = df.plot(df.lineage.sources[0])
        >>> chart.save("chart.json")
        >>> chart.serve()

        References
        ----------
        Altair: Declarative Visualization in Python.
            https://altair-viz.github.io
        """
        return visualization_utils.plot(self, reference_source=reference_source)
