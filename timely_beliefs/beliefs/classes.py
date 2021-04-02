import math
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import altair as alt
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, Interval
from sqlalchemy.ext.declarative import declared_attr, has_inherited_table
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm import Session, backref, relationship
from sqlalchemy.schema import UniqueConstraint

import timely_beliefs.utils as tb_utils
from timely_beliefs.beliefs import probabilistic_utils
from timely_beliefs.beliefs import utils as belief_utils
from timely_beliefs.beliefs.utils import is_pandas_structure, is_tb_structure
from timely_beliefs.db_base import Base
from timely_beliefs.sensors import utils as sensor_utils
from timely_beliefs.sensors.classes import DBSensor, Sensor, SensorDBMixin
from timely_beliefs.sources import utils as source_utils
from timely_beliefs.sources.classes import BeliefSource, DBBeliefSource
from timely_beliefs.visualization import utils as visualization_utils

METADATA = ["sensor", "event_resolution"]
DatetimeLike = Union[datetime, str, pd.Timestamp]
TimedeltaLike = Union[timedelta, str, pd.Timedelta]


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
    event_value: float  # todo: allow string to represent beliefs about labels? But what would nominal data mean for the interpretation of cp?
    sensor: Sensor
    source: BeliefSource
    cumulative_probability: float

    def __init__(  # noqa: C901 todo: the noqa can probably be removed when we deprecate the value argument
        self,
        sensor: Sensor,
        source: Union[BeliefSource, str, int],
        event_value: Optional[float] = None,
        value: Optional[float] = None,
        cumulative_probability: Optional[float] = None,
        cp: Optional[float] = None,
        sigma: Optional[float] = None,
        event_start: Optional[DatetimeLike] = None,
        event_time: Optional[DatetimeLike] = None,
        belief_horizon: Optional[TimedeltaLike] = None,
        belief_time: Optional[DatetimeLike] = None,
    ):
        self.sensor = sensor
        self.source = source_utils.ensure_source_exists(source)
        # todo: deprecate the 'value' argument in favor of 'event_value' (announced v1.1.0)
        self.event_value = tb_utils.replace_deprecated_argument(
            "value", value, "event_value", event_value
        )

        if [cumulative_probability, cp, sigma].count(None) not in (2, 3):
            raise ValueError(
                "Must specify either cumulative_probability, cp, sigma or none of them (0.5 is the default value)."
            )
        if cumulative_probability is not None:
            self.cumulative_probability = cumulative_probability
        elif cp is not None:
            self.cumulative_probability = cp
        elif sigma is not None:
            self.cumulative_probability = 1 / 2 + (math.erf(sigma / 2 ** 0.5)) / 2
        else:
            self.cumulative_probability = 0.5

        if [event_start, event_time].count(None) != 1:
            raise ValueError("Must specify either an event_start or an event_time.")
        elif event_start is not None:
            self.event_start = tb_utils.parse_datetime_like(event_start, "event_start")
        elif event_time is not None:
            if self.sensor.event_resolution != timedelta():
                raise KeyError(
                    "Sensor has a non-zero resolution, so it doesn't measure instantaneous events. "
                    "Use event_start instead of event_time."
                )
            self.event_start = tb_utils.parse_datetime_like(event_time, "event_time")

        if [belief_horizon, belief_time].count(None) != 1:
            raise ValueError("Must specify either a belief_horizon or a belief_time.")
        elif belief_horizon is not None:
            self.belief_horizon = tb_utils.parse_timedelta_like(belief_horizon)
        elif belief_time is not None:
            belief_time = tb_utils.parse_datetime_like(belief_time, "belief_time")
            self.belief_horizon = (
                self.sensor.knowledge_time(self.event_start, self.event_resolution)
                - belief_time
            )

    def __repr__(self):
        return (
            "<TimedBelief: at %s, the value of %s is %.2f (by %s with horizon %s)>"
            % (
                self.event_start,
                self.sensor,
                self.event_value,
                self.source,
                self.belief_horizon,
            )
        )

    @hybrid_property
    def event_end(self) -> datetime:
        return self.event_start + self.sensor.event_resolution

    @hybrid_property
    def knowledge_time(self) -> datetime:
        return self.sensor.knowledge_time(self.event_start, self.event_resolution)

    @hybrid_property
    def knowledge_horizon(self) -> timedelta:
        return self.sensor.knowledge_horizon(self.event_start, self.event_resolution)

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


class TimedBeliefDBMixin(TimedBelief):
    """
    Mixin class for a table with beliefs.
    The fields source and sensor do not point to another table - overwrite them to make that happen.
    """

    @declared_attr
    def __table_args__(cls):
        if has_inherited_table(cls):
            return (
                UniqueConstraint(
                    "event_start",
                    "belief_horizon",
                    "sensor_id",
                    "source_id",
                    name="_one_belief_by_one_source_uc",
                ),
            )
        return None

    event_start = Column(DateTime(timezone=True), primary_key=True, index=True)
    belief_horizon = Column(Interval(), nullable=False, primary_key=True)
    cumulative_probability = Column(
        Float, nullable=False, primary_key=True, default=0.5
    )
    event_value = Column(Float, nullable=False)

    @declared_attr
    def sensor_id(cls):
        return Column(
            Integer(),
            ForeignKey("sensor.id", ondelete="CASCADE"),
            primary_key=True,
            index=True,
        )

    @declared_attr
    def source_id(cls):
        return Column(Integer, ForeignKey("belief_source.id"), primary_key=True)

    def __init__(
        self,
        sensor: DBSensor,
        source: DBBeliefSource,
        event_value: Optional[float] = None,
        value: Optional[float] = None,
        cumulative_probability: Optional[float] = None,
        cp: Optional[float] = None,
        sigma: Optional[float] = None,
        event_start: Optional[DatetimeLike] = None,
        event_time: Optional[DatetimeLike] = None,
        belief_horizon: Optional[TimedeltaLike] = None,
        belief_time: Optional[DatetimeLike] = None,
    ):
        # todo: deprecate the 'value' argument in favor of 'event_value' (announced v1.3.0)
        event_value = tb_utils.replace_deprecated_argument(
            "value", value, "event_value", event_value
        )
        self.sensor_id = sensor.id
        self.source_id = source.id
        TimedBelief.__init__(
            self,
            sensor=sensor,
            source=source,
            event_value=event_value,
            cumulative_probability=cumulative_probability,
            cp=cp,
            sigma=sigma,
            event_start=event_start,
            event_time=event_time,
            belief_horizon=belief_horizon,
            belief_time=belief_time,
        )

    @classmethod
    def add_to_session(
        cls,
        session: Session,
        beliefs_data_frame: "BeliefsDataFrame",
        commit_transaction: bool = False,
    ):
        """Add a BeliefsDataFrame as timed beliefs to a database session.

        :param session: the database session to use
        :param beliefs_data_frame: the BeliefsDataFrame to be persisted
        :param commit_transaction: if True, the session is committed
                                   if False, you can still add other data to the session
                                   and commit it all within an atomic transaction
        """
        # Belief timing is stored as the belief horizon rather than as the belief time
        belief_records = (
            beliefs_data_frame.convert_index_from_belief_time_to_horizon()
            .reset_index()
            .to_dict("records")
        )
        beliefs = [cls(sensor=beliefs_data_frame.sensor, **d) for d in belief_records]
        session.add_all(beliefs)
        if commit_transaction:
            session.commit()

    @classmethod
    @tb_utils.append_doc_of("TimedBeliefDBMixin.search_session")
    def query(cls, *args, **kwargs):
        """Function will be deprecated. Please switch to using search_session."""
        # todo: deprecate this function (announced v1.3.0), which can clash with SQLAlchemy's Model.query()
        import warnings

        warnings.warn(
            "Function 'query' will be replaced by 'search_session'.",
            FutureWarning,
        )
        return cls.search_session(*args, **kwargs)

    @classmethod
    def search_session(
        cls,
        session: Session,
        sensor: SensorDBMixin,
        event_before: Optional[datetime] = None,
        event_not_before: Optional[datetime] = None,
        belief_before: Optional[datetime] = None,
        belief_not_before: Optional[datetime] = None,
        source: Optional[Union[BeliefSource, List[BeliefSource]]] = None,
    ) -> "BeliefsDataFrame":
        """Search a database session for beliefs about sensor events.
        :param session: the database session to use
        :param sensor: sensor to which the beliefs pertain
        :param event_before: only return beliefs about events that end before this datetime (inclusive)
        :param event_not_before: only return beliefs about events that start after this datetime (inclusive)
        :param belief_before: only return beliefs formed before this datetime (inclusive)
        :param belief_not_before: only return beliefs formed after this datetime (inclusive)
        :param source: only return beliefs formed by the given source or list of sources
        :returns: a multi-index DataFrame with all relevant beliefs

        TODO: rename params for clarity: event_finished_before, event_starts_not_before (or similar), same for beliefs
        """

        # Check for timezone-aware datetime input
        if event_before is not None:
            event_before = tb_utils.parse_datetime_like(event_before, "event_before")
        if event_not_before is not None:
            event_not_before = tb_utils.parse_datetime_like(
                event_not_before, "event_not_before"
            )
        if belief_before is not None:
            belief_before = tb_utils.parse_datetime_like(belief_before, "belief_before")
        if belief_not_before is not None:
            belief_not_before = tb_utils.parse_datetime_like(
                belief_not_before, "belief_not_before"
            )

        # Query sensor for relevant timing properties
        event_resolution, knowledge_horizon_fnc, knowledge_horizon_par = (
            session.query(
                sensor.__class__.event_resolution,
                sensor.__class__.knowledge_horizon_fnc,
                sensor.__class__.knowledge_horizon_par,
            )
            .filter(sensor.__class__.id == sensor.id)
            .one_or_none()
        )

        # Get bounds on the knowledge horizon (so we can already roughly filter by belief time)
        (
            knowledge_horizon_min,
            knowledge_horizon_max,
        ) = sensor_utils.eval_verified_knowledge_horizon_fnc(
            knowledge_horizon_fnc,
            knowledge_horizon_par,
            event_resolution=event_resolution,
            get_bounds=True,
        )

        # Query based on start_time_window
        q = session.query(cls).filter(cls.sensor_id == sensor.id)

        # Apply event time filter
        if event_before is not None:
            q = q.filter(cls.event_start + event_resolution <= event_before)
        if event_not_before is not None:
            q = q.filter(cls.event_start >= event_not_before)

        # Apply rough belief time filter
        if belief_before is not None:
            q = q.filter(
                cls.event_start
                <= belief_before + cls.belief_horizon + knowledge_horizon_max
            )
        if belief_not_before is not None:
            q = q.filter(
                cls.event_start
                >= belief_not_before + cls.belief_horizon + knowledge_horizon_min
            )

        # Apply source filter
        if source is not None:
            sources: list = [source] if not isinstance(source, list) else source
            source_cls = sources[0].__class__
            q = q.join(source_cls).filter(cls.source_id.in_([s.id for s in sources]))

        # Build our DataFrame of beliefs
        df = BeliefsDataFrame(sensor=sensor, beliefs=q.all())

        # Actually filter by belief time
        if belief_before is not None:
            df = df[df.index.get_level_values("belief_time") < belief_before]
        if belief_not_before is not None:
            df = df[df.index.get_level_values("belief_time") >= belief_not_before]

        return df


class DBTimedBelief(Base, TimedBeliefDBMixin):
    """Database representation of TimedBelief.
    We get fields from the Mixin and configure sensor and source relationships.
    We are not sure why the relationships cannot live in the Mixin as declared attributes,
    but they have to be here (thus other custom implementations need to include them, as well).
    """

    __tablename__ = "timed_beliefs"

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
        self,
        sensor: DBSensor,
        source: DBBeliefSource,
        event_value: Optional[float] = None,
        value: Optional[float] = None,
        cumulative_probability: Optional[float] = None,
        cp: Optional[float] = None,
        sigma: Optional[float] = None,
        event_start: Optional[DatetimeLike] = None,
        event_time: Optional[DatetimeLike] = None,
        belief_horizon: Optional[TimedeltaLike] = None,
        belief_time: Optional[DatetimeLike] = None,
    ):
        # todo: deprecate the 'value' argument in favor of 'event_value' (announced v1.3.0)
        event_value = tb_utils.replace_deprecated_argument(
            "value", value, "event_value", event_value
        )
        TimedBeliefDBMixin.__init__(
            self,
            sensor=sensor,
            source=source,
            event_value=event_value,
            cumulative_probability=cumulative_probability,
            cp=cp,
            sigma=sigma,
            event_start=event_start,
            event_time=event_time,
            belief_horizon=belief_horizon,
            belief_time=belief_time,
        )
        Base.__init__(self)


class BeliefsSeries(pd.Series):
    """Just for slicing, to keep around the metadata."""

    _metadata = METADATA

    @property
    def _constructor(self):
        def f(*args, **kwargs):
            """ Call __finalize__() after construction to inherit metadata. """
            return BeliefsSeries(*args, **kwargs).__finalize__(self, method="inherit")

        return f

    @property
    def _constructor_expanddim(self):
        def f(*args, **kwargs):
            """ Call __finalize__() after construction to inherit metadata. """
            # adapted from https://github.com/pandas-dev/pandas/issues/19850#issuecomment-367934440
            return BeliefsDataFrame(*args, **kwargs).__finalize__(
                self, method="inherit"
            )

        # workaround from https://github.com/pandas-dev/pandas/issues/32860#issuecomment-697993089
        f._get_axis_number = super(BeliefsSeries, self)._get_axis_number

        return f

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self."""
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        if hasattr(other, "name"):
            self.name = other.name
        return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return


class BeliefsDataFrame(pd.DataFrame):
    """Beliefs about a sensor.
    A BeliefsDataFrame object is a pandas.DataFrame with the following specific data columns and MultiIndex levels:

    columns: ["event_value"]
    index levels: ["event_start", "belief_time", "source", "cumulative_probability"]

    To initialize, pass sensor=Sensor("sensor_name"), together with data through one of these methods:
        Method 1:   pass a list of TimedBelief objects.
        Method 2:   pass a pandas DataFrame with columns ["event_start", "belief_time", "source", "event_value"]
                    - Optional column: "cumulative_probability" (the default is 0.5)
                    - Alternatively, use keyword arguments to replace columns containing unique values for each belief
        Method 3:   pass a pandas Series with DatetimeIndex and keyword arguments for "belief_time" or "belief_horizon", and "source"
                    - Alternatively, use the "event_start" keyword argument to ignore the index

    In addition to the standard DataFrame constructor arguments,
    BeliefsDataFrame also accepts the following keyword arguments:

    :param beliefs: a list of TimedBelief objects used to initialise the BeliefsDataFrame
    :param sensor: the Sensor object that each belief pertains to
    :param source: the source of each belief in the input DataFrame (a BeliefSource, str or int)
    :param event_start: the start of the event that each belief pertains to (a datetime)
    :param belief_time: the time at which each belief was formed (a datetime) - use this as alternative to belief_horizon
    :param belief_horizon: how long before (the event could be known) each belief was formed (a timedelta) - use this as alternative to belief_time
    :param cumulative_probability: a float in the range [0, 1] describing the cumulative probability of each belief - use this e.g. to initialize a BeliefsDataFrame containing only the values at 95% cumulative probability
    """

    _metadata = METADATA

    @property
    def _constructor(self):
        def f(*args, **kwargs):
            """ Call __finalize__() after construction to inherit metadata. """
            return BeliefsDataFrame(*args, **kwargs).__finalize__(
                self, method="inherit"
            )

        return f

    @property
    def _constructor_sliced(self):
        def f(*args, **kwargs):
            """ Call __finalize__() after construction to inherit metadata. """
            # adapted from https://github.com/pandas-dev/pandas/issues/19850#issuecomment-367934440
            return BeliefsSeries(*args, **kwargs).__finalize__(self, method="inherit")

        return f

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self."""
        # merge operation: using metadata of the left object
        if method == "merge":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == "concat":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self

    def __init__(  # noqa: C901 todo: refactor, e.g. by detecting initialization method
        self, *args, **kwargs
    ):
        """Initialise a multi-index DataFrame with beliefs about a unique sensor."""

        # Initialized with a BeliefsSeries or BeliefsDataFrame
        if len(args) > 0 and isinstance(args[0], (BeliefsSeries, BeliefsDataFrame)):
            super().__init__(*args, **kwargs)
            assign_sensor_and_event_resolution(
                self, args[0].sensor, args[0].event_resolution
            )
            return

        # Obtain parameters that are specific to our DataFrame subclass
        sensor: Sensor = kwargs.pop("sensor", None)
        event_resolution: TimedeltaLike = kwargs.pop("event_resolution", None)
        source: Union[BeliefSource, str, int] = kwargs.pop("source", None)
        source: BeliefSource = source_utils.ensure_source_exists(
            source, allow_none=True
        )
        event_start: DatetimeLike = kwargs.pop("event_start", None)
        belief_time: DatetimeLike = kwargs.pop("belief_time", None)
        belief_horizon: datetime = kwargs.pop("belief_horizon", None)
        cumulative_probability: float = kwargs.pop("cumulative_probability", None)
        beliefs: List[TimedBelief] = kwargs.pop("beliefs", None)
        if beliefs is None:  # check if args contains a list of beliefs
            for i, arg in enumerate(args):
                if isinstance(arg, list):
                    if all(isinstance(b, TimedBelief) for b in arg):
                        args = list(args)
                        beliefs = args.pop(
                            i
                        )  # arg contains beliefs, and we simultaneously remove it from args
                        args = tuple(args)
                        break

        # Define our columns and indices
        columns = ["event_value"]
        indices = ["event_start", "belief_time", "source", "cumulative_probability"]
        default_types = {
            "event_value": float,
            "event_start": datetime,
            "event_end": datetime,
            "belief_time": datetime,
            "belief_horizon": timedelta,
            "source": BeliefSource,
            "cumulative_probability": float,
        }

        # Pick an initialization method
        if beliefs:
            # Method 1

            # Call the pandas DataFrame constructor with the right input
            kwargs["columns"] = columns

            # Check for different sensors
            unique_sensors = set(belief.sensor for belief in beliefs)
            if len(unique_sensors) != 1:
                raise ValueError("BeliefsDataFrame cannot describe multiple sensors.")
            sensor = list(unique_sensors)[0]

            # Check for different sources with the same name
            unique_sources = set(belief.source for belief in beliefs)
            unique_source_names = set(source.name for source in unique_sources)
            if len(unique_source_names) != len(unique_sources):
                raise ValueError(
                    "Source names must be unique. Cannot initialise BeliefsDataFrame given the following unique sources:\n%s"
                    % unique_sources
                )

            # Construct data and index from beliefs before calling super class
            beliefs = sorted(
                set(beliefs),
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
            super().__init__(*args, **kwargs)

        else:
            # Method 2 and 3

            # Interpret initialisation with a pandas Series (preprocessing step of method 3)
            if len(args) > 0 and isinstance(args[0], pd.Series):
                args = list(args)
                args[0] = args[0].copy()  # avoid inplace operations
                args[0] = args[0].to_frame(
                    name="event_value" if not args[0].name else args[0].name
                )
                if isinstance(args[0].index, pd.DatetimeIndex) and event_start is None:
                    args[0].index.name = (
                        "event_start" if not args[0].index.name else args[0].index.name
                    )
                    args[0].reset_index(inplace=True)
                args = tuple(args)
            elif len(args) > 0 and isinstance(args[0], pd.DataFrame):
                # Avoid inplace operations on the input DataFrame
                args = list(args)
                args[0] = args[0].copy()  # avoid inplace operations
                args = tuple(args)

            super().__init__(*args, **kwargs)

            if len(args) == 0 or (self.empty and is_pandas_structure(args[0])):
                set_columns_and_indices_for_empty_frame(
                    self, columns, indices, default_types
                )
            elif is_pandas_structure(args[0]) and not is_tb_structure(args[0]):
                # Set (possibly overwrite) each index level to a unique value if set explicitly
                if source is not None:
                    self["source"] = source_utils.ensure_source_exists(source)
                elif "source" not in self:
                    raise KeyError("DataFrame should contain column named 'source'.")
                elif not isinstance(self["source"].dtype, BeliefSource):
                    self["source"] = self["source"].apply(
                        source_utils.ensure_source_exists
                    )
                if event_start is not None:
                    self["event_start"] = tb_utils.parse_datetime_like(
                        event_start, "event_start"
                    )
                elif "event_start" not in self and "event_end" not in self:
                    raise KeyError(
                        "DataFrame should contain column named 'event_start' or 'event_end'."
                    )
                else:
                    self["event_start"] = self["event_start"].apply(
                        lambda x: tb_utils.parse_datetime_like(x, "event_start")
                    )
                if belief_time is not None:
                    self["belief_time"] = tb_utils.parse_datetime_like(
                        belief_time, "belief_time"
                    )
                elif belief_horizon is not None:
                    self["belief_horizon"] = belief_horizon
                elif "belief_time" not in self and "belief_horizon" not in self:
                    raise KeyError(
                        "DataFrame should contain column named 'belief_time' or 'belief_horizon'."
                    )
                elif "belief_time" in self:
                    self["belief_time"] = self["belief_time"].apply(
                        lambda x: tb_utils.parse_datetime_like(x, "belief_time")
                    )
                elif not pd.api.types.is_timedelta64_dtype(
                    self["belief_horizon"]
                ) and self["belief_horizon"].dtype not in (timedelta, pd.Timedelta):
                    raise TypeError(
                        "belief_horizon should be of type datetime.timedelta or pandas.Timedelta."
                    )
                if cumulative_probability is not None:
                    self["cumulative_probability"] = cumulative_probability
                elif "cumulative_probability" not in self:
                    self["cumulative_probability"] = 0.5
                if "event_value" not in self:
                    raise KeyError(
                        "DataFrame should contain column named 'event_value'."
                    )

                # Check for correct types and convert if possible
                self["event_start"] = pd.to_datetime(self["event_start"])
                if "belief_time" in self:
                    self["belief_time"] = pd.to_datetime(self["belief_time"])
                self["source"] = self["source"].apply(source_utils.ensure_source_exists)

                # Set index levels and metadata
                if "belief_horizon" in self and "belief_time" not in self:
                    indices = [
                        "belief_horizon" if index == "belief_time" else index
                        for index in indices
                    ]
                if "event_end" in self and "event_start" not in self:
                    indices = [
                        "event_end" if index == "event_start" else index
                        for index in indices
                    ]
                self.set_index(indices, inplace=True)

        assign_sensor_and_event_resolution(
            self, sensor, tb_utils.parse_timedelta_like(event_resolution)
        )

    def append_from_time_series(
        self,
        event_value_series: pd.Series,
        source: BeliefSource,
        belief_horizon: Union[timedelta, pd.Series],
        cumulative_probability: float = 0.5,
    ) -> "BeliefsDataFrame":
        """Append beliefs from time series entries into this BeliefsDataFrame. Sensor is assumed to be the same.
        Returns a new BeliefsDataFrame object."""
        beliefs = belief_utils.load_time_series(
            event_value_series,
            self.sensor,
            source,
            belief_horizon,
            cumulative_probability,
        )
        return self.append(BeliefsDataFrame(sensor=self.sensor, beliefs=beliefs))

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

    def drop_belief_time_or_horizon_index_level(self) -> "BeliefsDataFrame":
        return self.droplevel(
            "belief_horizon" if "belief_horizon" in self.index.names else "belief_time"
        )

    def set_event_value_from_source(
        self, reference_source: BeliefSource
    ) -> "BeliefsDataFrame":
        return self.groupby(level=["event_start"], group_keys=False).apply(
            lambda x: x.groupby(level=["source"]).pipe(
                probabilistic_utils.set_truth, reference_source
            )
        )

    @property
    def knowledge_times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            self.event_starts.to_series(name="knowledge_time").apply(
                lambda event_start: self.sensor.knowledge_time(
                    event_start, self.event_resolution
                )
            )
        )

    @property
    def knowledge_horizons(self) -> pd.TimedeltaIndex:
        return pd.TimedeltaIndex(
            self.event_starts.to_series(name="knowledge_horizon").apply(
                lambda event_start: self.sensor.knowledge_horizon(
                    event_start, self.event_resolution
                )
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
            self.event_starts.to_series(name="event_end").apply(
                lambda event_start: event_start + self.event_resolution
            )
        )

    @property
    def sources(self) -> pd.Index:
        """Shorthand to list the source of each belief in the BeliefsDataFrame."""
        # Todo: subclass pd.Index to create a BeliefSourceIndex?
        return self.index.get_level_values("source")

    @property
    def event_time_window(self) -> Tuple[datetime, datetime]:
        start, end = self.index.get_level_values("event_start")[[0, -1]]
        end = end + self.event_resolution
        return start, end

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
        event_start: DatetimeLike,
        belief_time_window: Tuple[Optional[DatetimeLike], Optional[DatetimeLike]] = (
            None,
            None,
        ),
        belief_horizon_window: Tuple[
            Optional[TimedeltaLike], Optional[TimedeltaLike]
        ] = (
            None,
            None,
        ),
        keep_event_start: bool = False,
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
        :param belief_horizon_window: optional tuple specifying a horizon window
               (e.g. between 1 and 2 hours before the event value could have been known)
        """

        if self.empty:
            return self

        df = self.xs(
            tb_utils.parse_datetime_like(event_start, "event_start"),
            level="event_start",
            drop_level=False,
        ).sort_index()
        if belief_time_window[0] is not None:
            df = df[
                df.index.get_level_values("belief_time")
                >= tb_utils.parse_datetime_like(belief_time_window[0], "belief_time")
            ]
        if belief_time_window[1] is not None:
            df = df[
                df.index.get_level_values("belief_time")
                <= tb_utils.parse_datetime_like(belief_time_window[1], "belief_time")
            ]
        if belief_horizon_window != (None, None):
            if belief_time_window != (None, None):
                raise ValueError(
                    "Cannot pass both a belief time window and belief horizon window."
                )
            df = df.convert_index_from_belief_time_to_horizon()
            if belief_horizon_window[0] is not None:
                df = df[
                    df.index.get_level_values("belief_horizon")
                    >= tb_utils.parse_timedelta_like(belief_horizon_window[0])
                ]
            if belief_horizon_window[1] is not None:
                df = df[
                    df.index.get_level_values("belief_horizon")
                    <= tb_utils.parse_timedelta_like(belief_horizon_window[1])
                ]
            df = df.convert_index_from_belief_horizon_to_time()
        if not keep_event_start:
            df = df.droplevel("event_start")
        return df

    @hybrid_method
    def fixed_viewpoint(
        self,
        belief_time: DatetimeLike = None,
        belief_time_window: Tuple[Optional[DatetimeLike], Optional[DatetimeLike]] = (
            None,
            None,
        ),
        update_belief_times: bool = False,
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

        :param belief_time: datetime indicating the belief should be formed at least before or at this time
        :param belief_time_window: optional tuple specifying a time window within which beliefs should have been formed
        :param update_belief_times: if True, update the belief time of each belief with the given fixed viewpoint
        """

        if self.empty:
            return self

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
                >= tb_utils.parse_datetime_like(belief_time_window[0], "belief_time")
            ]
        if belief_time_window[1] is not None:
            df = df[
                df.index.get_level_values("belief_time")
                <= tb_utils.parse_datetime_like(belief_time_window[1], "belief_time")
            ]
        df = belief_utils.select_most_recent_belief(df)
        if update_belief_times is True:
            return tb_utils.replace_multi_index_level(
                df,
                "belief_time",
                pd.DatetimeIndex(data=[belief_time_window[1]] * len(df.index)),
            )
        else:
            return df

    @hybrid_method
    def rolling_viewpoint(
        self,
        belief_horizon: TimedeltaLike = None,
        belief_horizon_window: Tuple[
            Optional[TimedeltaLike], Optional[TimedeltaLike]
        ] = (
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
        :param belief_horizon_window: optional tuple specifying a horizon window
               (e.g. between 1 and 2 days before the event value could have been known)
        """

        if self.empty:
            return self

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
                df.index.get_level_values("belief_horizon")
                >= tb_utils.parse_timedelta_like(belief_horizon_window[0])
            ]
        if belief_horizon_window[1] is not None:
            df = df[
                df.index.get_level_values("belief_horizon")
                <= tb_utils.parse_timedelta_like(belief_horizon_window[1])
            ]
        return belief_utils.select_most_recent_belief(df)

    @hybrid_method
    def resample_events(
        self,
        event_resolution: TimedeltaLike,
        distribution: Optional[str] = None,
        keep_only_most_recent_belief: bool = False,
    ) -> "BeliefsDataFrame":
        """Aggregate over multiple events (downsample) or split events into multiple sub-events (upsample).

        NB If you need to only keep the most recent belief,
        set keep_only_most_recent_belief=True for a significant speed boost.
        """

        if self.empty:
            return self
        event_resolution = tb_utils.parse_timedelta_like(event_resolution)
        if event_resolution == self.event_resolution:
            return self
        df = self

        belief_timing_col = (
            "belief_time" if "belief_time" in df.index.names else "belief_horizon"
        )

        # fast track a common case where each event has only one deterministic belief and only the most recent belief is needed
        if (
            df.lineage.number_of_beliefs == df.lineage.number_of_events
            and keep_only_most_recent_belief
            and df.lineage.number_of_sources == 1
        ):
            if event_resolution > self.event_resolution:
                # downsample
                column_functions = {
                    "event_value": "mean",
                    "source": "first",  # keep the only source
                    belief_timing_col: "max"
                    if belief_timing_col == "belief_time"
                    else "min",  # keep only most recent belief
                    "cumulative_probability": "prod",  # assume independent variables
                }
                df = downsample_beliefs_data_frame(
                    df, event_resolution, column_functions
                )
                df.event_resolution = event_resolution
            else:
                # upsample
                df = df.reset_index(
                    level=[belief_timing_col, "source", "cumulative_probability"]
                )
                new_index = pd.date_range(
                    start=df.index[0],
                    periods=len(df) * (self.event_resolution // event_resolution),
                    freq=event_resolution,
                    name="event_start",
                )
                df = df.reindex(new_index).fillna(method="pad")
                df.event_resolution = event_resolution
                df = df.set_index(
                    [belief_timing_col, "source", "cumulative_probability"], append=True
                )

        # slow track in case each event has more than 1 belief or probabilistic beliefs
        else:
            if belief_timing_col == "belief_horizon":
                df = df.convert_index_from_belief_horizon_to_time()
            df = (
                df.groupby(
                    [pd.Grouper(freq=event_resolution, level="event_start"), "source"],
                    group_keys=False,
                )
                .apply(
                    lambda x: belief_utils.resample_event_start(
                        x,
                        event_resolution,
                        input_resolution=self.event_resolution,
                        distribution=distribution,
                        keep_only_most_recent_belief=keep_only_most_recent_belief,
                    )
                )
                .sort_index()
            )

            # Update metadata with new resolution
            df.event_resolution = event_resolution

            # Put back lost metadata (because groupby statements still tend to lose it)
            df.sensor = self.sensor

            if belief_timing_col == "belief_horizon":
                df = df.convert_index_from_belief_time_to_horizon()

        return df

    def accuracy(
        self,
        t: Union[datetime, timedelta] = None,
        reference_belief_horizon: timedelta = None,
        reference_source: BeliefSource = None,
        lite_metrics: bool = False,
    ) -> "BeliefsDataFrame":
        """Simply get the accuracy of beliefs about events, at a given time (pass a datetime), at a given horizon
        (pass a timedelta), or as a function of horizon (the default).

        By default the accuracy is determined with respect to the most recent beliefs held by the same source.
        Optionally, set a reference source to determine accuracy with respect to beliefs held by a specific source.

        By default the accuracy is determined with respect to the most recent beliefs.

        By default the mean absolute error (MAE), the mean absolute percentage error (MAPE) and
        the weighted absolute percentage error (WAPE) are returned.

        For more options, use df.fixed_viewpoint_accuracy() or df.rolling_viewpoint_accuracy() instead.

        :param t: optional datetime or timedelta for a fixed or rolling viewpoint, respectively
        :param reference_belief_horizon: optional timedelta to indicate that
               the accuracy should be determined with respect to the latest belief at this duration past knowledge time
        :param reference_source: optional BeliefSource to indicate that
               the accuracy should be determined with respect to the beliefs held by the given source
        :param lite_metrics: if True, skip calculation of MAPE and WAPE
        """

        df = self
        if t is None:

            # Set reference values if needed
            if (
                reference_belief_horizon is not None
                or reference_source is not None
                or "reference_value" not in df.columns
            ):
                df = df.set_reference_values(
                    reference_belief_horizon=reference_belief_horizon,
                    reference_source=reference_source,
                )

            return pd.concat(
                [
                    df.rolling_viewpoint_accuracy(h, lite_metrics=lite_metrics)
                    for h in df.lineage.belief_horizons
                ],
                keys=df.lineage.belief_horizons,
                names=["belief_horizon"],
            )

        elif isinstance(t, datetime):
            return df.fixed_viewpoint_accuracy(
                t, reference_source=reference_source, lite_metrics=lite_metrics
            )
        elif isinstance(t, timedelta):
            return df.rolling_viewpoint_accuracy(
                t, reference_source=reference_source, lite_metrics=lite_metrics
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
        reference_source: BeliefSource = None,
        lite_metrics: bool = False,
    ) -> "BeliefsDataFrame":
        """Get the accuracy of beliefs about events at a given time.

        Alternatively, select the accuracy of beliefs formed within a certain time window. This allows setting a maximum
        acceptable freshness of the data.

        By default the accuracy is determined with respect to the reference values in the `reference_value` column.
        This column is created if it does not exist, or if one of the following reference parameters is not None.
        By default, the reference values are the most recent beliefs held by the same source.
        - Optionally, set a reference belief time to determine accuracy with respect to beliefs at a specific time.
        - Alternatively, set a reference belief horizon instead of a reference belief time.
        - Optionally, set a reference source to determine accuracy with respect to beliefs held by a specific source.
        These options allow to define what is considered to be true at a certain time.

        By default the mean absolute error (MAE), the mean absolute percentage error (MAPE) and
        the weighted absolute percentage error (WAPE) are returned.

        :Example:

        >>> from datetime import datetime
        >>> from pytz import utc
        >>> from timely_beliefs.examples import example_df as df
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
        :param belief_time_window: optional tuple specifying a time window in which the belief should have been formed
               (e.g. between June 1st and 2nd)
        :param reference_belief_time: optional datetime to indicate that
               the accuracy should be determined with respect to the latest belief held at this time
        :param reference_belief_horizon: optional timedelta to indicate that
               the accuracy should be determined with respect to the latest belief at this duration past knowledge time
        :param reference_source: optional BeliefSource to indicate that
               the accuracy should be determined with respect to the beliefs held by the given source
        :param lite_metrics: if True, skip calculation of MAPE and WAPE
        :returns: BeliefsDataFrame with columns for mae, mape and wape (and optionally, the reference values), indexed by source only
        """
        df = self

        # Set reference values if needed
        if (
            reference_belief_time is not None
            or reference_belief_horizon is not None
            or reference_source is not None
            or "reference_value" not in df.columns
        ):
            df = df.set_reference_values(
                reference_belief_time=reference_belief_time,
                reference_belief_horizon=reference_belief_horizon,
                reference_source=reference_source,
            )

        # Take a fixed viewpoint
        df = df.fixed_viewpoint(belief_time, belief_time_window)

        return belief_utils.compute_accuracy_scores(df, lite_metrics=lite_metrics)

    def rolling_viewpoint_accuracy(
        self,
        belief_horizon: timedelta = None,
        belief_horizon_window: Tuple[Optional[timedelta], Optional[timedelta]] = (
            None,
            None,
        ),
        reference_belief_horizon: timedelta = None,
        reference_source: BeliefSource = None,
        lite_metrics: bool = False,
    ) -> "BeliefsDataFrame":
        """Get the accuracy of beliefs about events at a given horizon.

        Alternatively, set a horizon window to select the accuracy of beliefs formed within a certain time window before
        knowledge time (with negative horizons indicating post knowledge time).
        This allows setting a maximum acceptable freshness of the data.

        By default the accuracy is determined with respect to the reference values in the `reference_value` column.
        This column is created if it does not exist, or if one of the following reference parameters is not None.
        By default, the reference values are the most recent beliefs held by the same source.
        - Optionally, set a reference belief horizon to determine accuracy with respect to beliefs at a specific horizon.
        - Optionally, set a reference source to determine accuracy with respect to beliefs held by a specific source.
        These options allow to define what is considered to be true at a certain time after an event.

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
        >>> # Select the accuracy of beliefs formed 10 days beforehand with respect to the latest belief formed by the first source
        >>> df.rolling_viewpoint_accuracy(belief_horizon=timedelta(days=10), reference_source=df.lineage.sources[0])

        :param belief_horizon: timedelta indicating the belief should be formed at least this duration before knowledge time
        :param belief_horizon_window: optional tuple specifying a horizon window
               (e.g. between 1 and 2 days before the event value could have been known)
        :param reference_belief_horizon: optional timedelta to indicate that
               the accuracy should be determined with respect to the latest belief at this duration past knowledge time
        :param reference_source: optional BeliefSource to indicate that
               the accuracy should be determined with respect to the beliefs held by the given source
        :param lite_metrics: if True, skip calculation of MAPE and WAPE
        """
        df = self

        # Set reference values if needed
        if (
            reference_belief_horizon is not None
            or reference_source is not None
            or "reference_value" not in df.columns
        ):
            df = df.set_reference_values(
                reference_belief_horizon=reference_belief_horizon,
                reference_source=reference_source,
            )

        # Take a rolling viewpoint
        df = df.rolling_viewpoint(belief_horizon, belief_horizon_window)

        return belief_utils.compute_accuracy_scores(df, lite_metrics=lite_metrics)

    def plot(
        self,
        show_accuracy: bool = False,
        active_fixed_viewpoint_selector: bool = True,
        reference_source: BeliefSource = None,
        intuitive_forecast_horizon: bool = True,
        interpolate: bool = True,
    ) -> alt.LayerChart:
        """Visualize the BeliefsDataFrame in an interactive Altair chart.

        :param show_accuracy: Set to False to plot time series data only
        :param active_fixed_viewpoint_selector: If true, fixed viewpoint beliefs can be selected
        :param reference_source: BeliefSource to indicate that
               the accuracy should be determined with respect to the beliefs held by the given source
        :param intuitive_forecast_horizon: If true, horizons are shown with respect to event start rather than knowledge time
        :param interpolate: If True, the time series chart shows a user-friendly interpolated line
               rather than more accurate stripes indicating average values
        :returns: Altair chart object with a vega-lite representation (for more information, see reference below).

        >>> chart = df.plot(df.lineage.sources[0])
        >>> chart.save("chart.json")
        >>> chart.serve()

        References
        ----------
        Altair: Declarative Visualization in Python.
            https://altair-viz.github.io
        """
        return visualization_utils.plot(
            self,
            show_accuracy=show_accuracy,
            active_fixed_viewpoint_selector=active_fixed_viewpoint_selector,
            reference_source=reference_source,
            intuitive_forecast_horizon=intuitive_forecast_horizon,
            interpolate=interpolate,
        )

    @staticmethod
    def plot_ridgeline_fixed_viewpoint(
        reference_time: datetime,
        df: "BeliefsDataFrame",
        future_only: bool = False,
        distribution: str = "uniform",
        event_value_window: Tuple[float, float] = None,
    ) -> alt.FacetChart:
        """Create a ridgeline plot of the latest beliefs held at a certain reference time.

        :param reference_time: datetime, reference to determine belief horizons
        :param df: BeliefsDataFrame
        :param future_only: if True mask the past
        :param distribution: string, distribution name to use (discrete, normal or uniform)
        :param event_value_window: optional tuple specifying an event value window for the x-axis
               (e.g. plot temperatures between -1 and 21 degrees Celsius)
        """
        if df.lineage.number_of_sources > 1:
            raise ValueError(
                "Cannot create plot beliefs from multiple sources. BeliefsDataFrame must contain beliefs from a single source."
            )

        if future_only is True:
            df = df[df.index.get_level_values("event_start") >= reference_time]
        df = df.fixed_viewpoint(
            reference_time, update_belief_times=True
        ).convert_index_from_belief_time_to_horizon()

        return visualization_utils.ridgeline_plot(
            df, True, distribution, event_value_window
        )

    @staticmethod
    def plot_ridgeline_belief_history(
        event_start: datetime,
        df: "BeliefsDataFrame",
        past_only: bool = False,
        distribution: str = "uniform",
        event_value_window: Tuple[float, float] = None,
    ) -> alt.FacetChart:
        """Create a ridgeline plot of the belief history about a specific event.

        :param event_start: datetime, indicating the start time of the event for which to plot the belief history
        :param df: BeliefsDataFrame
        :param past_only: if True mask the future (i.e. mask any updates of beliefs after knowledge time)
        :param distribution: string, distribution name to use (discrete, normal or uniform)
        :param event_value_window: optional tuple specifying an event value window for the x-axis
               (e.g. plot temperatures between -1 and 21 degrees Celsius)
        """
        if df.lineage.number_of_sources > 1:
            raise ValueError(
                "Cannot create plot beliefs from multiple sources. BeliefsDataFrame must contain beliefs from a single source."
            )

        df = df.belief_history(
            event_start=event_start,
            belief_horizon_window=(timedelta(hours=0) if past_only else None, None),
            keep_event_start=True,
        ).convert_index_from_belief_time_to_horizon()

        return visualization_utils.ridgeline_plot(
            df, False, distribution, event_value_window
        )

    def set_reference_values(
        self,
        reference_belief_time: datetime = None,
        reference_belief_horizon: timedelta = None,
        reference_source: BeliefSource = None,
        return_expected_value: bool = False,
    ) -> "BeliefsDataFrame":
        """Add a column with reference values.
        By default the reference will be the expected value of the most recent belief held by the same source.
        Optionally, set a reference belief horizon.
        Optionally, set a reference source present in the BeliefsDataFrame.
        These options allow to define what is considered to be true at a certain time after an event.
        Todo: Option to set probabilistic reference values

        :param reference_belief_time: optional datetime to indicate that
               the accuracy should be determined with respect to the latest belief held at this time
        :param reference_belief_horizon: optional timedelta to indicate that
               the accuracy should be determined with respect to the latest belief at this duration past knowledge time
        :param reference_source: optional BeliefSource to indicate that
               the accuracy should be determined with respect to the beliefs held by the given source
        :param return_expected_value: if True, set a deterministic reference by picking the expected value
        """

        df = self

        reference_df = df.groupby(level="event_start").apply(
            lambda x: belief_utils.set_reference(
                x,
                reference_belief_time=reference_belief_time,
                reference_belief_horizon=reference_belief_horizon,
                reference_source=reference_source,
                return_expected_value=return_expected_value,
            )
        )

        # Concat to add a column while keeping original cp index level
        if return_expected_value is True:
            if "belief_time" in df.index.names:
                df = df.convert_index_from_belief_time_to_horizon()
                df = pd.concat(
                    [df, reference_df.reindex(df.index, method="pad")], axis=1
                )
                return df.convert_index_from_belief_horizon_to_time()
            else:
                return pd.concat(
                    [df, reference_df.reindex(df.index, method="pad")], axis=1
                )
        if "belief_time" in df.index.names:
            df = df.convert_index_from_belief_time_to_horizon()
            df = pd.concat([df, reference_df], axis=1)
            return df.convert_index_from_belief_horizon_to_time()
        return pd.concat([df, reference_df], axis=1)


def set_columns_and_indices_for_empty_frame(df, columns, indices, default_types):
    """ Set appropriate columns and indices for the empty BeliefsDataFrame. """
    if "belief_horizon" in df and "belief_time" not in df:
        indices = [
            "belief_horizon" if index == "belief_time" else index for index in indices
        ]
    if "event_end" in df and "event_start" not in df:
        indices = [
            "event_end" if index == "event_start" else index for index in indices
        ]
    for col in columns + indices:
        df[col] = None

        # Set the pandas types where needed
        if default_types[col] == datetime:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize("utc")
        elif default_types[col] == timedelta:
            df[col] = pd.to_timedelta(df[col])
        elif default_types[col] in (int, float):
            df[col] = pd.to_numeric(df[col])

    df.set_index(indices, inplace=True)  # todo: pandas GH30517


def assign_sensor_and_event_resolution(df, sensor, event_resolution):
    """ Set the Sensor metadata (including timing properties of the sensor). """
    df.sensor = sensor
    df.event_resolution = (
        event_resolution
        if event_resolution
        else sensor.event_resolution
        if sensor
        else None
    )


def downsample_beliefs_data_frame(
    df: BeliefsDataFrame, event_resolution: timedelta, col_att_dict: Dict[str, str]
) -> BeliefsDataFrame:
    """Because df.resample().agg() doesn't behave nicely for subclassed DataFrames,
    we aggregate each index level and column separately against the resampled event_start level,
    and then recombine them afterwards.
    """
    belief_timing_col = (
        "belief_time" if "belief_time" in df.index.names else "belief_horizon"
    )
    event_timing_col = "event_start" if "event_start" in df.index.names else "event_end"
    return pd.concat(
        [
            getattr(
                df.reset_index()
                .set_index(event_timing_col)[col]
                .to_frame()
                .resample(event_resolution),
                att,
            )()
            for col, att in col_att_dict.items()
        ],
        axis=1,
    ).set_index([belief_timing_col, "source", "cumulative_probability"], append=True)
