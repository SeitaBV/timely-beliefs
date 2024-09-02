from __future__ import annotations

import math
import types
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Type, Union

from packaging import version

if TYPE_CHECKING:
    import altair as alt
    from sktime.forecasting.base import BaseForecaster

import numpy as np
import pandas as pd
import pytz
from pandas.core.groupby import DataFrameGroupBy
from pandas.util._decorators import cache_readonly
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Interval,
    and_,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm import Session, backref, declarative_mixin, relationship
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.schema import Index
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.expression import Selectable

import timely_beliefs.utils as tb_utils
from timely_beliefs.beliefs import probabilistic_utils
from timely_beliefs.beliefs import utils as belief_utils
from timely_beliefs.beliefs.utils import is_pandas_structure, is_tb_structure, meta_repr
from timely_beliefs.db_base import Base
from timely_beliefs.sensors import utils as sensor_utils
from timely_beliefs.sensors.classes import DBSensor, Sensor, SensorDBMixin
from timely_beliefs.sensors.func_store.knowledge_horizons import ex_ante, ex_post
from timely_beliefs.sources import utils as source_utils
from timely_beliefs.sources.classes import BeliefSource, DBBeliefSource

METADATA = ["sensor", "event_resolution"]
DatetimeLike = Union[datetime, str, pd.Timestamp]
TimedeltaLike = Union[timedelta, str, pd.Timedelta]
JoinTarget = Union[
    Selectable,
    type,
    AliasedClass,
    types.FunctionType,
]


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
    are given, then this will be treated as a deterministic belief (cp=1). As an alternative to specifying a cumulative
    probability explicitly, you can specify an integer number of standard deviations which is translated
    into a cumulative probability assuming a normal distribution (e.g. sigma=-1 becomes cp=0.1587).
    """

    event_start: datetime
    belief_horizon: timedelta
    event_value: float  # todo: allow string to represent beliefs about labels? But what would nominal data mean for the interpretation of cp?
    sensor: Sensor
    source: BeliefSource
    cumulative_probability: float

    def __init__(
        self,
        sensor: Sensor,
        source: BeliefSource | str | int,
        event_value: float | None = None,
        cumulative_probability: float | None = None,
        cp: float | None = None,
        sigma: float | None = None,
        event_start: DatetimeLike | None = None,
        event_time: DatetimeLike | None = None,
        belief_horizon: TimedeltaLike | None = None,
        belief_time: DatetimeLike | None = None,
    ):
        self.sensor = sensor
        self.source = source_utils.ensure_source_exists(source)
        self.event_value = event_value

        if [cumulative_probability, cp, sigma].count(None) not in (2, 3):
            raise ValueError(
                "Must specify either cumulative_probability, cp, sigma or none of them (0.5 is the default value)."
            )
        if cumulative_probability is not None:
            self.cumulative_probability = cumulative_probability
        elif cp is not None:
            self.cumulative_probability = cp
        elif sigma is not None:
            self.cumulative_probability = 1 / 2 + (math.erf(sigma / 2**0.5)) / 2
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


@declarative_mixin
class TimedBeliefDBMixin(TimedBelief):
    """
    Mixin class for a table with beliefs.
    The fields source and sensor do not point to another table - overwrite them to make that happen.
    """

    @declared_attr
    def __table_args__(cls):
        return (
            Index(
                f"{cls.__tablename__}_search_session_idx",
                "event_start",
                "sensor_id",
                "source_id",
                postgresql_include=[
                    "belief_horizon",  # we use min() on this (most_recent_beliefs_only)
                ],
            ),
            Index(
                f"{cls.__tablename__}_search_session_singleevent_idx",
                "event_start",
                "sensor_id",
            ),
        )

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
        event_value: float | None = None,
        cumulative_probability: float | None = None,
        cp: float | None = None,
        sigma: float | None = None,
        event_start: DatetimeLike | None = None,
        event_time: DatetimeLike | None = None,
        belief_horizon: TimedeltaLike | None = None,
        belief_time: DatetimeLike | None = None,
    ):
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
        expunge_session: bool = False,
        allow_overwrite: bool = False,
        bulk_save_objects: bool = True,
        commit_transaction: bool = False,
    ):
        """Add a BeliefsDataFrame as timed beliefs to a database session.

        If you are adding lots of beliefs, it's most efficient to use expunge_session=True and allow_overwrite=False

        :param session:             the database session to use
        :param beliefs_data_frame:  the BeliefsDataFrame to be persisted
        :param expunge_session:     if True, all non-flushed instances are removed from the session before adding beliefs.
                                    Expunging can resolve problems you might encounter with states of objects in your session.
                                    When using this option, you might want to flush newly-created objects which are not beliefs
                                    (e.g. a sensor or data source object).
        :param allow_overwrite:     if True, new objects are merged
                                    if False, objects are added to the session or bulk saved
        :param bulk_save_objects:   if True, objects are bulk saved with session.bulk_save_objects(),
                                    which is quite fast but has several caveats, see:
                                    https://docs.sqlalchemy.org/orm/persistence_techniques.html#bulk-operations-caveats
                                    if False, objects are added to the session with session.add_all()
        :param commit_transaction:  if True, the session is committed
                                    if False, you can still add other data to the session
                                    and commit it all within an atomic transaction
        """
        if beliefs_data_frame.empty:
            return
        # Belief timing is stored as the belief horizon rather than as the belief time
        beliefs_data_frame = (
            beliefs_data_frame.convert_index_from_belief_time_to_horizon().reset_index()
        )
        beliefs = [
            cls(sensor=beliefs_data_frame.sensor, **d)
            for d in beliefs_data_frame.to_dict("records")
        ]

        if expunge_session:
            session.expunge_all()

        if bulk_save_objects:
            # serialize sources and sensor, while adding new sources

            # serialize sources
            beliefs_data_frame["source_id"] = beliefs_data_frame["source"].apply(
                lambda x: x.id
            )

            # Add new sources
            newbies = pd.isnull(beliefs_data_frame["source_id"])
            if any(newbies):
                session.add_all(beliefs_data_frame.loc[newbies, "source"])
                session.flush()  # assign IDs
                beliefs_data_frame.loc[newbies, "source_id"] = beliefs_data_frame.loc[
                    newbies, "source"
                ].apply(lambda x: x.id)

            # serialize sensor
            beliefs_data_frame["sensor_id"] = beliefs_data_frame.sensor.id
            beliefs_data_frame = beliefs_data_frame.drop(columns=["source"])

            smt = insert(cls).values(beliefs_data_frame.to_dict("records"))

            if allow_overwrite:
                smt = smt.on_conflict_do_update(
                    index_elements=[
                        "event_start",
                        "belief_horizon",
                        "source_id",
                        "sensor_id",
                        "cumulative_probability",
                    ],
                    set_=dict(event_value=smt.excluded.event_value),
                )

            session.execute(smt)

        else:
            if allow_overwrite:
                for belief in beliefs:
                    session.merge(belief)
            else:
                session.add_all(beliefs)

        if commit_transaction:
            session.commit()

    @classmethod
    def search_session(  # noqa: C901
        cls,
        session: Session,
        sensor: SensorDBMixin | int,
        sensor_class: Type[SensorDBMixin] | None = DBSensor,
        event_starts_after: datetime | None = None,
        event_ends_after: datetime | None = None,
        event_starts_before: datetime | None = None,
        event_ends_before: datetime | None = None,
        beliefs_after: datetime | None = None,
        beliefs_before: datetime | None = None,
        horizons_at_least: timedelta | None = None,
        horizons_at_most: timedelta | None = None,
        source: BeliefSource | list[BeliefSource] | None = None,
        most_recent_beliefs_only: bool = False,
        most_recent_events_only: bool = False,
        most_recent_only: bool = False,
        place_beliefs_in_sensor_timezone: bool = True,
        place_events_in_sensor_timezone: bool = True,
        custom_filter_criteria: list[BinaryExpression] | None = None,
        custom_join_targets: list[JoinTarget] | None = None,
    ) -> "BeliefsDataFrame":
        """Search a database session for beliefs about sensor events.

        The optional arguments represent optional filters, with two exceptions:
        - sensor_class makes it possible to create a query on sensor subclasses
        - custom_join_targets makes it possible to add custom filters using other (incl. subclassed) targets

        A word about query speed:
        As data sets can become quite large (and get wrapped into a BeliefsDataFrame), we recommend to filter by sensor and source, as well as a time range, in order to limit processing time.
        You should also consider selecting only the most recent beliefs (for each event, there might have been more than one).
        Also, look into selecting only the most recent events (per source).
        These two latter tips decrease the dataset and thus post-processing time. They use sub-queries, though, so be sure to use the main filtering, as well, like time range.
        Finally, if you only need one most recent value (one result row), there is a fast-track (most_recent_only).

        :param session: the database session to use
        :param sensor: sensor to which the beliefs pertain, or its unique sensor id
        :param sensor_class: optionally pass the sensor (sub)class explicitly (only needed if you pass a sensor id instead of a sensor, and your sensor class is not DBSensor); the class should be mapped to a database table
        :param event_starts_after: only return beliefs about events that start after this datetime (inclusive)
        :param event_ends_after: only return beliefs about events that end after this datetime (exclusive for non-instantaneous events, inclusive for instantaneous events)
                                 note that the first event may transpire partially before this datetime
        :param event_starts_before: only return beliefs about events that start before this datetime (exclusive for non-instantaneous events, inclusive for instantaneous events)
                                    note that the last event may transpire partially after this datetime
        :param event_ends_before: only return beliefs about events that end before this datetime (inclusive)
        :param beliefs_after: only return beliefs formed after this datetime (inclusive)
        :param beliefs_before: only return beliefs formed before this datetime (inclusive)
        :param horizons_at_least: only return beliefs with a belief horizon equal or greater than this timedelta (for example, use timedelta(0) to get ante knowledge time beliefs)
        :param horizons_at_most: only return beliefs with a belief horizon equal or less than this timedelta (for example, use timedelta(0) to get post knowledge time beliefs)
        :param source: only return beliefs formed by the given source or list of sources. This speeds up your query, so if you know the source, let the query know.
        :param most_recent_beliefs_only: only return the most recent beliefs for each event from each source (minimum belief horizon)
        :param most_recent_events_only: only return (post knowledge time) beliefs for the most recent event (maximum event start) for each source
        :param most_recent_only: only looks up one most recent event and then only returns the most recent belief about that event that it finds. This is a considerable fast-track if only one most recent belief is all you need.
        :param place_beliefs_in_sensor_timezone: if True (the default), belief times are converted to the timezone of the sensor
        :param place_events_in_sensor_timezone: if True (the default), event starts are converted to the timezone of the sensor
        :param custom_filter_criteria: additional filters, such as ones that rely on subclasses
        :param custom_join_targets: additional join targets, to accommodate filters that rely on other targets (e.g. subclasses)
        :returns: a multi-index DataFrame with all relevant beliefs
        """
        source_class = cls.source.property.mapper.class_

        # Check for timezone-aware datetime input
        if not pd.isnull(event_starts_after):
            event_starts_after = tb_utils.parse_datetime_like(
                event_starts_after, "event_starts_after"
            )
        if not pd.isnull(event_ends_after):
            event_ends_after = tb_utils.parse_datetime_like(
                event_ends_after, "event_ends_after"
            )
        if not pd.isnull(event_starts_before):
            event_starts_before = tb_utils.parse_datetime_like(
                event_starts_before, "event_starts_before"
            )
        if not pd.isnull(event_ends_before):
            event_ends_before = tb_utils.parse_datetime_like(
                event_ends_before, "event_ends_before"
            )
        if not pd.isnull(beliefs_after):
            beliefs_after = tb_utils.parse_datetime_like(
                beliefs_after, "belief_not_before"
            )
        if not pd.isnull(beliefs_before):
            beliefs_before = tb_utils.parse_datetime_like(
                beliefs_before, "belief_before"
            )

        # Query sensor, required for its timing properties
        if isinstance(sensor, int):
            # Check for proper sensor class
            if not issubclass(sensor_class, SensorDBMixin):
                raise ValueError(
                    f"sensor {sensor} is a {type(sensor)}, which is not a subclass of {SensorDBMixin}"
                )
            sensor = session.execute(
                select(sensor_class).filter(sensor_class.id == sensor)
            ).scalar_one_or_none()
            if sensor is None:
                raise ValueError("No such sensor")

        # Parse source parameter
        sources: list = []
        if source is not None:
            sources = [source] if not isinstance(source, list) else source
            # Fast-track empty list of sources
            if sources == []:
                return BeliefsDataFrame(sensor=sensor, beliefs=[])

        # Get bounds on the knowledge horizon (so we can already roughly filter by belief time)
        (
            knowledge_horizon_min,
            knowledge_horizon_max,
        ) = sensor_utils.eval_verified_knowledge_horizon_fnc(
            sensor.knowledge_horizon_fnc,
            sensor.knowledge_horizon_par,
            event_resolution=sensor.event_resolution,
            get_bounds=True,
        )

        def apply_event_timing_filters(q):
            """Apply filters that concern the event time.

            This includes any custom filters
            """
            if not pd.isnull(event_starts_after):
                q = q.filter(cls.event_start >= event_starts_after)
            if not pd.isnull(event_ends_after):
                if sensor.event_resolution == timedelta(0):
                    # inclusive
                    q = q.filter(cls.event_start >= event_ends_after)
                else:
                    # exclusive
                    q = q.filter(
                        cls.event_start > event_ends_after - sensor.event_resolution
                    )
            if not pd.isnull(event_starts_before):
                if sensor.event_resolution == timedelta(0):
                    # inclusive
                    q = q.filter(cls.event_start <= event_starts_before)
                else:
                    # exclusive
                    q = q.filter(cls.event_start < event_starts_before)
            if not pd.isnull(event_ends_before):
                q = q.filter(
                    cls.event_start <= event_ends_before - sensor.event_resolution
                )

            return q

        def apply_belief_timing_filters(q):
            """Apply filters that concern the belief timing.

            This includes any custom filters
            """

            # Apply rough belief time filter
            if not pd.isnull(
                beliefs_after
            ) and belief_utils.extreme_timedeltas_not_equal(
                knowledge_horizon_min, timedelta.min
            ):
                q = q.filter(
                    cls.event_start - cls.belief_horizon
                    >= beliefs_after + knowledge_horizon_min
                )
            if not pd.isnull(
                beliefs_before
            ) and belief_utils.extreme_timedeltas_not_equal(
                knowledge_horizon_max, timedelta.max
            ):
                q = q.filter(
                    cls.event_start - cls.belief_horizon
                    <= beliefs_before + knowledge_horizon_max
                )

            # Apply belief horizon filter
            if not pd.isnull(horizons_at_least):
                q = q.filter(cls.belief_horizon >= horizons_at_least)
            if not pd.isnull(horizons_at_most):
                q = q.filter(cls.belief_horizon <= horizons_at_most)

            # Apply custom filter criteria and join targets
            if custom_filter_criteria is not None:
                q = q.filter(*custom_filter_criteria)
            if custom_join_targets is not None:
                for target in custom_join_targets:
                    q = q.join(target)

            return q

        # Main query
        q = select(
            cls.event_start,
            cls.belief_horizon,
            cls.source_id,
            cls.cumulative_probability,
            cls.event_value,
        ).filter(cls.sensor_id == sensor.id)

        q = apply_event_timing_filters(q)
        q = apply_belief_timing_filters(q)

        # Apply source filter
        if len(sources) > 0:
            q = q.join(source_class).filter(cls.source_id.in_([s.id for s in sources]))

        # Switch to fast-track if user wants both most recent events & beliefs and one source is requested.
        # In this case, we know only one row will be returned. (If only one source exists in the to-be-returned dataset,
        #  we'd get one row without filtering for source, but we cannot know if that happens before we query.)
        if (most_recent_beliefs_only and most_recent_events_only) and len(sources) == 1:
            most_recent_only = True
            most_recent_events_only = False
            most_recent_beliefs_only = False
        # Otherwise, we should not allow to mix the two most-recent-X approaches, for clarity
        elif (most_recent_beliefs_only or most_recent_events_only) and (
            most_recent_only
        ):
            raise ValueError(
                "most_recent_events|beliefs_only can not be used with most_recent_only."
            )

        # Apply most recent beliefs filter as subquery
        most_recent_beliefs_only_incompatible_criteria = (
            beliefs_before is not None or beliefs_after is not None
        ) and sensor.knowledge_horizon_fnc not in (ex_ante.__name__, ex_post.__name__)
        if (
            most_recent_beliefs_only
            and not most_recent_beliefs_only_incompatible_criteria
        ):
            subq = select(
                cls.event_start,
                cls.source_id,
                func.min(cls.belief_horizon).label("most_recent_belief_horizon"),
            )
            # Apply event and belief timing filters to the subquery, too,
            # before taking the minimum horizon (the former is crucial for speed)
            subq = apply_event_timing_filters(subq)
            subq = apply_belief_timing_filters(subq)
            subq = (
                subq.filter(cls.sensor_id == sensor.id)
                .group_by(cls.event_start, cls.source_id)
                .subquery()
            )
            q = q.join(
                subq,
                and_(
                    cls.event_start == subq.c.event_start,
                    cls.source_id == subq.c.source_id,
                    cls.belief_horizon == subq.c.most_recent_belief_horizon,
                ),
            )

        # Apply most recent events filter as subquery
        if most_recent_events_only:
            subq_most_recent_events = select(
                cls.source_id,
                func.max(cls.event_start).label("most_recent_event_start"),
            )
            subq_most_recent_events = apply_event_timing_filters(
                subq_most_recent_events
            )
            subq_most_recent_events = apply_belief_timing_filters(
                subq_most_recent_events
            )
            subq_most_recent_events = (
                subq_most_recent_events.filter(cls.sensor_id == sensor.id)
                .group_by(cls.source_id)
                .subquery()
            )
            q = q.join(
                subq_most_recent_events,
                and_(
                    cls.source_id == subq_most_recent_events.c.source_id,
                    cls.event_start
                    == subq_most_recent_events.c.most_recent_event_start,
                ),
            )

        # Apply fast-track most-recent-only approach
        # Note that currently, this only works for a deterministic belief. A probabilistic belief would have multiple rows
        # (sharing the same event start and belief horizon).
        if most_recent_only:
            q = q.order_by(cls.event_start.desc(), cls.belief_horizon.asc()).limit(1)

        # Useful debugging code, let's keep it here
        # from sqlalchemy.dialects import postgresql
        # print(q.compile(dialect=postgresql.dialect()))

        # Build our DataFrame of beliefs
        df = pd.DataFrame(session.execute(q))

        if df.empty:
            return BeliefsDataFrame(sensor=sensor)
        df.columns = [
            "event_start",
            "belief_horizon",
            "source_id",
            "cumulative_probability",
            "event_value",
        ]

        # Fill in sources
        if source is None:
            source_ids = df["source_id"].unique().tolist()
            sources = session.scalars(
                select(source_class).filter(source_class.id.in_(source_ids))
            ).all()
        source_map = {source.id: source for source in sources}
        df["source_id"] = df["source_id"].map(source_map)
        df = df.rename(columns={"source_id": "source"})

        # Build our BeliefsDataFrame
        df = BeliefsDataFrame(df, sensor=sensor)
        df = df.convert_index_from_belief_horizon_to_time()

        # Actually filter by belief time
        if beliefs_after is not None:
            df = df[df.index.get_level_values("belief_time") >= beliefs_after]
        if beliefs_before is not None:
            df = df[df.index.get_level_values("belief_time") <= beliefs_before]

        # Select most recent beliefs using postprocessing in case of incompatible search criteria
        if most_recent_beliefs_only and most_recent_beliefs_only_incompatible_criteria:
            df = belief_utils.select_most_recent_belief(df)

        # Convert timezone of beliefs and events to sensor timezone
        if place_beliefs_in_sensor_timezone:
            df = df.convert_timezone_of_belief_timing_index(sensor.timezone)
        if place_events_in_sensor_timezone:
            df = df.convert_timezone_of_event_timing_index(sensor.timezone)

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
        event_value: float | None = None,
        cumulative_probability: float | None = None,
        cp: float | None = None,
        sigma: float | None = None,
        event_start: DatetimeLike | None = None,
        event_time: DatetimeLike | None = None,
        belief_horizon: TimedeltaLike | None = None,
        belief_time: DatetimeLike | None = None,
    ):
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

    # Pre-Pandas 2.0, call __finalize__() after construction to inherit metadata.
    if version.parse(pd.__version__) < version.parse("2.0.0"):

        @property
        def _constructor(self):
            def f(*args, **kwargs):
                return BeliefsSeries(*args, **kwargs).__finalize__(
                    self, method="inherit"
                )

            return f

    else:

        @property
        def _constructor(self):
            return partial(BeliefsSeries)

    @property
    def _constructor_expanddim(self):
        def f(*args, **kwargs):
            """Call __finalize__() after construction to inherit metadata."""
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
            object.__setattr__(self, "name", getattr(other, "name"))
        return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def __repr__(self):
        """Add the sensor and event resolution to the string representation of the BeliefsSeries."""
        return super().__repr__() + "\n" + meta_repr(self)

    @property
    def event_frequency(self) -> timedelta | None:
        """Duration between observations of events.

        :returns: a timedelta for regularly spaced observations
                  None for irregularly spaced observations
        """
        return pd.Timedelta(pd.infer_freq(self.index.unique("event_start")))


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

    :param beliefs: a list of TimedBelief objects used to initialize the BeliefsDataFrame
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
            """Call __finalize__() after construction to inherit metadata."""
            return BeliefsDataFrame(*args, **kwargs).__finalize__(
                self, method="inherit"
            )

        return f

    @property
    def _constructor_sliced(self):
        def f(*args, **kwargs):
            """Call __finalize__() after construction to inherit metadata."""
            # adapted from https://github.com/pandas-dev/pandas/issues/19850#issuecomment-367934440
            return BeliefsSeries(*args, **kwargs).__finalize__(self, method="inherit")

        return f

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self."""
        # merge operation: using metadata of the left object

        # Check if sources have unique names
        if hasattr(other, "objs"):
            sources = []
            for df in other.objs:
                if "source" in df.index:
                    sources.extend(
                        df.index.get_level_values(level="source")
                        .unique()
                        .to_numpy(dtype="object")
                    )
            sources = set(sources)
            source_names = set(source.name for source in sources)
            if len(source_names) != len(sources):
                raise ValueError(
                    "Source names must be unique. Cannot initialise BeliefsDataFrame given the following unique sources:\n%s"
                    % sources
                )

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
        source: BeliefSource | str | int = kwargs.pop("source", None)
        source: BeliefSource = source_utils.ensure_source_exists(
            source, allow_none=True
        )
        event_start: DatetimeLike = kwargs.pop("event_start", None)
        belief_time: DatetimeLike = kwargs.pop("belief_time", None)
        belief_horizon: datetime = kwargs.pop("belief_horizon", None)
        cumulative_probability: float = kwargs.pop("cumulative_probability", None)
        beliefs: list[TimedBelief] = kwargs.pop("beliefs", None)
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
            unique_sources = set(str(belief.source) for belief in beliefs)
            unique_source_string_representations = set(
                str(source) for source in unique_sources
            )
            if len(unique_source_string_representations) != len(unique_sources):
                raise ValueError(
                    "String representations of sources must be unique. Cannot initialise BeliefsDataFrame given the following unique sources:\n%s"
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
        belief_horizon: timedelta | pd.Series,
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

    def convert_index_from_event_end_to_start(self) -> "BeliefsDataFrame":
        return tb_utils.replace_multi_index_level(self, "event_end", self.event_starts)

    def convert_index_from_event_start_to_end(self) -> "BeliefsDataFrame":
        return tb_utils.replace_multi_index_level(self, "event_start", self.event_ends)

    def convert_timezone_of_belief_timing_index(
        self, timezone: str | pytz.timezone
    ) -> "BeliefsDataFrame":
        if "belief_horizon" in self.index.names:
            return self  # timedeltas don't have timezones
        elif "belief_time" in self.index.names:
            return tb_utils.replace_multi_index_level(
                self,
                "belief_time",
                pd.to_datetime(self.belief_times, utc=True).tz_convert(timezone),
            )
        else:
            raise ValueError(
                "Missing level 'belief_horizon' or 'belief_time' in index."
            )

    def convert_timezone_of_event_timing_index(
        self, timezone: str | pytz.timezone
    ) -> "BeliefsDataFrame":
        if "event_end" in self.index.names:
            return tb_utils.replace_multi_index_level(
                self,
                "event_end",
                pd.to_datetime(self.event_ends, utc=True).tz_convert(timezone),
            )
        elif "event_start" in self.index.names:
            return tb_utils.replace_multi_index_level(
                self,
                "event_start",
                pd.to_datetime(self.event_starts, utc=True).tz_convert(timezone),
            )
        else:
            raise ValueError("Missing level 'event_start' or 'event_end' in index.")

    def drop_belief_time_or_horizon_index_level(self) -> "BeliefsDataFrame":
        return self.droplevel(
            "belief_horizon" if "belief_horizon" in self.index.names else "belief_time"
        )

    def set_event_value_from_source(
        self, reference_source: BeliefSource
    ) -> "BeliefsDataFrame":
        return self.groupby(level=["event_start"], group_keys=False).apply(
            lambda x: x.groupby(level=["source"], group_keys=False).pipe(
                probabilistic_utils.set_truth, reference_source
            )
        )

    @cache_readonly
    def probabilistic_depth_per_belief(self):
        """Return the number of probabilistic values per belief."""
        return self.droplevel("cumulative_probability").index.value_counts()

    @cache_readonly
    def probabilistic_depth_count(self):
        """Return a count of the number of probabilistic values per belief.

        For example, this frame contains 8 beliefs described by 3 probabilistic values,
        and another 8 described by 2 probabilistic values.

        >>> from timely_beliefs.examples.beliefs_data_frames import sixteen_probabilistic_beliefs
        >>> sixteen_probabilistic_beliefs().probabilistic_depth_count
        count
        3    8
        2    8
        Name: count, dtype: int64
        """
        return self.probabilistic_depth_per_belief.value_counts()

    @property
    def event_frequency(self) -> timedelta | None:
        """Duration between observations of events.

        :returns: a timedelta for regularly spaced observations
                  None for irregularly spaced observations
        """
        return pd.Timedelta(pd.infer_freq(self.index.unique("event_start")))

    @property
    def knowledge_times(self) -> pd.DatetimeIndex:
        return self.sensor.knowledge_time(self.event_starts, self.event_resolution)

    @property
    def knowledge_horizons(self) -> pd.TimedeltaIndex:
        return self.sensor.knowledge_horizon(self.event_starts, self.event_resolution)

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
            return (
                self.knowledge_times.tz_convert("UTC")
                - self.belief_times.tz_convert("UTC")
            ).rename("belief_horizon")

    @property
    def event_starts(self) -> pd.DatetimeIndex:
        if "event_start" in self.index.names:
            return pd.DatetimeIndex(self.index.get_level_values("event_start"))
        else:
            return pd.DatetimeIndex(
                self.event_ends.to_series(name="event_start").apply(
                    lambda event_end: event_end - self.event_resolution
                )
            )

    @property
    def event_ends(self) -> pd.DatetimeIndex:
        if "event_end" in self.index.names:
            return pd.DatetimeIndex(self.index.get_level_values("event_end"))
        else:
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
    def event_time_window(self) -> tuple[datetime, datetime]:
        start, end = self.index.get_level_values("event_start")[[0, -1]]
        end = end + self.event_resolution
        return start, end

    @hybrid_method
    def for_each_belief(
        self, fnc: Callable = None, *args: Any, **kwargs: Any
    ) -> "BeliefsDataFrame" | DataFrameGroupBy:
        """Convenient function to apply a function to each belief in the BeliefsDataFrame.
        A belief is a group with unique event start, belief time and source.
        Each individual belief may be deterministic (defined by a single row), or probabilistic (multiple rows).
        If no function is given, return the GroupBy object.

        :Example:

        >>> # Apply some function that accepts a DataFrame, a positional argument and a keyword argument
        >>> df.for_each_belief(some_function, True, a=1)
        >>> # Pipe some other function that accepts a GroupBy object, a positional argument and a keyword argument
        >>> df.for_each_belief().pipe(some_other_function, True, a=1)
        >>> # If you want to call this method within another groupby function, pass the df group explicitly
        >>> df.for_each_belief(some_function, True, a=1, df=df)
        """
        return self._for_each_belief(fnc, False, *args, **kwargs)

    def _for_each_belief(
        self, fnc: Callable, collective_beliefs: bool, *args: Any, **kwargs: Any
    ) -> "BeliefsDataFrame" | DataFrameGroupBy:
        """
        If collective_beliefs is True, just group by event start and belief time.
        Otherwise, group beliefs by source, too.
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
        if collective_beliefs is False:
            index_names.append("source")
        gr = df.groupby(level=index_names, group_keys=False)
        if fnc is not None:
            return gr.apply(lambda x: fnc(x, *args, **kwargs))
        return gr

    @hybrid_method
    def for_each_collective_belief(
        self, fnc: Callable = None, *args: Any, **kwargs: Any
    ) -> "BeliefsDataFrame" | DataFrameGroupBy:
        """Convenient function to apply a function to each collective belief in the BeliefsDataFrame.
        A collective belief is a group with unique event start and belief time, which may contain multiple sources.
        Each individual belief may be deterministic (defined by a single row), or probabilistic (multiple rows).
        If no function is given, return the GroupBy object.

        :Example:

        >>> # Apply some function that accepts a DataFrame, a positional argument and a keyword argument
        >>> df.for_each_collective_belief(some_function, True, a=1)
        >>> # Pipe some other function that accepts a GroupBy object, a positional argument and a keyword argument
        >>> df.for_each_collective_belief().pipe(some_other_function, True, a=1)
        >>> # If you want to call this method within another groupby function, pass the df group explicitly
        >>> df.for_each_collective_belief(some_function, True, a=1, df=df)
        """
        return self._for_each_belief(fnc, True, *args, **kwargs)

    @hybrid_method
    def make_deterministic(self) -> "BeliefsDataFrame":
        if self.lineage.probabilistic_depth > 1:
            return self.for_each_belief(probabilistic_utils.get_expected_belief)
        return self

    @hybrid_method
    def belief_history(
        self,
        event_start: DatetimeLike,
        belief_time_window: tuple[DatetimeLike | None, DatetimeLike | None] = (
            None,
            None,
        ),
        belief_horizon_window: tuple[TimedeltaLike | None, TimedeltaLike | None] = (
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
        belief_time_window: tuple[DatetimeLike | None, DatetimeLike | None] = (
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
        belief_horizon_window: tuple[TimedeltaLike | None, TimedeltaLike | None] = (
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
        distribution: str | None = None,
        keep_only_most_recent_belief: bool = False,
        keep_nan_values: bool = False,
        boundary_policy: str = "first",
    ) -> "BeliefsDataFrame":
        """Aggregate over multiple events (downsample) or split events into multiple sub-events (upsample).

        Resampling events in a BeliefsDataFrames can be quite a slow operation, depending on the complexity of the data.
        In general, resampling events may need to deal with:
        - the distinction between event resolution (the duration of events) and event frequency (the duration between event starts)
          todo: this distinction was introduced in timely-beliefs==1.15.0 and still needs to be incorporated in code
        - upsampling or downsampling
          note: this function supports both
        - different resampling methods (e.g. 'mean', 'interpolate' or 'first')
          note: this function defaults to 'mean' for downsampling and 'pad' for upsampling
          todo: allow to set this explicitly, and derive a default from a sensor attribute
        - different event resolutions (e.g. instantaneous recordings vs. hourly averages)
          note: this function only supports few less complex cases of resampling instantaneous sensors
        - daylight savings time (DST) transitions
          note: this function resamples such that events coincide with midnight in both DST and non-DST
          note: only tested for instantaneous sensors
          todo: streamline how DST transitions are handled for instantaneous and non-instantaneous sensors
        - combining beliefs with different belief times
          note: for BeliefsDataFrames with multiple belief times per event, consider keep_only_most_recent_belief=True for a significant speed boost
        - combining beliefs from different sources
          note: resampling is currently done separately for each source
        - joining marginal probability distributions

        Each of the above aspects needs a carefully thought out and tested implementation.
        Quite a few cases have been implemented in detail already, such as:
        - a quite general (but slow) implementation for sensors recording average flows.
        - a much faster implementation for some less complex cases
        - a separate implementation for less complex BeliefsDataFrames with instantaneous recordings,
          which is robust against DST transitions.

        If you encounter a case that is not supported yet, we invite you to open a GitHub ticket and describe your case.

        Finally, a note on why we named this function 'resample_events'.
        BeliefsDataFrames record the timing of events, the timing of beliefs, sources and probabilities.
        It is conceivable to resample any of these, for example:
        - resample belief times to show how beliefs about an event change every day
        - resample sources to show how model versions improved accuracy
        - resample probabilities given some distribution to show how that affects extreme outcomes and risk

        Although the term, when applied to time series, usually is about resampling events,
        we wanted the function name to be explicit about what we resample.

        :param event_resolution: duration of events after resampling (except for instantaneous sensors, in which case
                                 it is the duration between events after resampling: the event frequency).
        :param distribution: Type of probability distribution to assume when taking the mean over probabilistic values.
                             Supported distributions are 'discrete', 'normal' and 'uniform'.
        :param keep_only_most_recent_belief: If True, assign the most recent belief time to each event after resampling.
                                             Only applies in case of multiple beliefs per event.
        :param keep_nan_values: If True, place back resampled NaN values. Drops NaN values by default.
        :param boundary_policy: When upsampling to instantaneous events,
                                take the 'max', 'min' or 'first' value at event boundaries.
        """

        if self.empty:
            return self
        event_resolution = tb_utils.parse_timedelta_like(event_resolution)
        if event_resolution == self.event_resolution:
            return self
        df = self

        # Resample instantaneous sensors
        # The event resolution stays zero, but the event frequency is updated
        if df.event_resolution == timedelta(0):
            if df.lineage.number_of_events != len(df):
                raise NotImplementedError("Please file a GitHub ticket.")
            return belief_utils.resample_instantaneous_events(df, event_resolution)

        belief_timing_col = (
            "belief_time" if "belief_time" in df.index.names else "belief_horizon"
        )

        # fast track a common case where each event has only one deterministic belief by the same source, and:
        # - only the most recent belief is needed, or
        # - we are upsampling, or
        # - all beliefs share a common belief time
        if (
            df.lineage.number_of_beliefs == df.lineage.number_of_events
            and (
                keep_only_most_recent_belief
                or event_resolution < self.event_resolution
                or df.lineage.number_of_belief_times == 1
            )
            and df.lineage.number_of_sources == 1
        ):
            if event_resolution > self.event_resolution:
                # downsample
                column_functions = {
                    "event_value": "mean",
                    "source": "first",  # keep the only source
                    belief_timing_col: (
                        "max" if belief_timing_col == "belief_time" else "min"
                    ),  # keep only most recent belief
                    "cumulative_probability": "mean",  # we just have one point on each CDF
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
                df = belief_utils.upsample_beliefs_data_frame(
                    df=df,
                    event_resolution=event_resolution,
                    keep_nan_values=keep_nan_values,
                    boundary_policy=boundary_policy,
                )
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

    @hybrid_method
    def form_beliefs(
        self,
        belief_time: datetime,
        source: BeliefSource,
        event_start: datetime = None,
        event_time_window: tuple[datetime, datetime]
        | None = (
            None,
            None,
        ),
        forecaster: "BaseForecaster" | None = None,
        concatenate: bool = False,
    ):
        """Form new beliefs by applying a given forecaster.

        Note that the result may contain NaN values, for example, when the BeliefsDataFrame contains insufficient data.

        :param belief_time:       Time at which the forecasts where made
                                  (any belief after this time will be inaccessible to the forecaster).
        :param source:            Source to assign the newly formed beliefs to.
        :param event_start:       Set this to forecast a single event with the given start time.
        :param event_time_window: Set this to forecast all events within the given time window.
        :param forecaster:        Forecasting model. Currently, only sktime models are supported.
                                  The default forecaster simply repeats the last known value.
                                  Hint: set a seasonal periodicity to obtain a more reasonable (baseline) forecast.
                                  For example:
                                      >>> periodicity = pd.Timedelta("PT1W")
                                      >>> forecaster = NaiveForecaster(sp=periodicity // df.event_resolution)
        :param concatenate:       If True, the new beliefs are concatenated with the original BeliefsDataFrame.
                                  If False, the new beliefs are returned (use pd.concat to add them yourself).
        """
        from sktime.forecasting.base import BaseForecaster

        if forecaster is None:
            from sktime.forecasting.naive import NaiveForecaster

            forecaster = NaiveForecaster(strategy="last")
        if event_start is not None:
            if event_time_window != (None, None):
                raise ValueError(
                    "Cannot pass both an event start and event start window."
                )
            event_time_window = (event_start, event_start + self.event_resolution)

        # Check assumptions
        if self.sensor.knowledge_time(event_time_window[0]) < belief_time:
            # No backcasting
            raise NotImplementedError("Backcasting is not implemented.")
        if self.lineage.number_of_sources > 1:
            # Single source only
            raise NotImplementedError(
                "Cannot form new beliefs given multi-sourced data. Consider picking 1 source, e.g. using: df = df[df.index.get_level_values('source') == df.lineage.sources[0]]"
            )
        if self.lineage.probabilistic_depth != 1:
            # Deterministic beliefs only
            raise NotImplementedError(
                "Cannot form new beliefs given probabilistic data. Consider pre-computing deterministic beliefs, e.g. using: df = df.make_deterministic()"
            )
        if self.lineage.number_of_beliefs != self.lineage.number_of_events:
            # One belief per event
            raise NotImplementedError(
                "Cannot form new beliefs given multiple beliefs about the same event. Consider picking a single belief per event, e.g. using: df = belief_utils.select_most_recent_belief(df)"
            )

        df = self

        # Do NOT use future data to predict NOR to fit
        belief_horizon_in_index = False
        if "belief_horizon" in df.index.names:
            belief_horizon_in_index = True
            df = df.convert_index_from_belief_horizon_to_time()
        df = df[df.index.get_level_values("belief_time") <= belief_time]

        if df.empty:
            # skip sktime, which expects a non-empty frame
            y_pred = pd.Series(
                data=np.nan,
                index=belief_utils.initialize_index(
                    start=event_time_window[0],
                    end=event_time_window[1],
                    resolution=df.event_resolution,
                ),
            )
        else:
            # Simplify index
            df = df.reset_index()[["event_start", "event_value"]].set_index(
                "event_start"
            )

            # Convert to local time in UTC, which sktime expects
            tz = event_time_window[0].tzinfo
            utc = pytz.utc
            event_time_window = tuple(
                dt.astimezone(utc).replace(tzinfo=None) for dt in event_time_window
            )
            df.index = df.index.tz_convert(utc).tz_localize(None)

            # Resample to the resolution the data already has, just to set the index frequency, which sktime expects
            # The new index ends just before the start of the first forecast (which may introduce trailing NaN values),
            # as sktime expects no gap between the indices of the input and forecasts when applying seasonal periodicity.
            df = df.reindex(
                belief_utils.initialize_index(
                    df.index[0],
                    event_time_window[0],
                    resolution=df.event_resolution,
                )
            )
            df = df.resample(df.event_resolution).mean()

            # todo: if forecaster does not handle missing values, impute intermediate missing values and forecast trailing missing values

            # Apply model
            if isinstance(forecaster, BaseForecaster):
                forecast_event_starts = belief_utils.initialize_index(
                    event_time_window[0],
                    event_time_window[1],
                    resolution=df.event_resolution,
                )
                forecaster.fit(
                    df["event_value"], df.loc[:, df.columns != "event_value"]
                )
                y_pred = forecaster.predict(forecast_event_starts)
            else:
                raise NotImplementedError("Consider opening a GitHub issue.")

            # Relocalize to requested timezone
            y_pred.index = y_pred.index.tz_localize("utc").tz_convert(tz)

        # Prepare results as BeliefsDataframe
        new_df = BeliefsDataFrame(
            y_pred, source=source, belief_time=belief_time, sensor=df.sensor
        )
        if belief_horizon_in_index:
            new_df = new_df.convert_index_from_belief_time_to_horizon()
        if concatenate:
            return pd.concat([self, new_df])
        return new_df

    def accuracy(
        self,
        t: datetime | timedelta = None,
        reference_belief_horizon: timedelta = None,
        reference_source: BeliefSource = None,
        lite_metrics: bool = False,
    ) -> "BeliefsDataFrame":
        """Simply get the accuracy of beliefs about events, at a given time (pass a datetime), at a given horizon
        (pass a timedelta), or as a function of horizon (the default).

        By default, the accuracy is determined with respect to the most recent beliefs held by the same source.
        Optionally, set a reference source to determine accuracy with respect to beliefs held by a specific source.

        By default, the accuracy is determined with respect to the most recent beliefs.

        By default, the mean absolute error (MAE), the mean absolute percentage error (MAPE) and
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
        belief_time_window: tuple[datetime | None, datetime | None] = (
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

        By default, the accuracy is determined with respect to the reference values in the `reference_value` column.
        This column is created if it does not exist, or if one of the following reference parameters is not None.
        By default, the reference values are the most recent beliefs held by the same source.
        - Optionally, set a reference belief time to determine accuracy with respect to beliefs at a specific time.
        - Alternatively, set a reference belief horizon instead of a reference belief time.
        - Optionally, set a reference source to determine accuracy with respect to beliefs held by a specific source.
        These options allow to define what is considered to be true at a certain time.

        By default, the mean absolute error (MAE), the mean absolute percentage error (MAPE) and
        the weighted absolute percentage error (WAPE) are returned.

        :Example:

        >>> from datetime import datetime
        >>> from pytz import utc
        >>> from timely_beliefs.examples import get_example_df
        >>> df = get_example_df()
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
        belief_horizon_window: tuple[timedelta | None, timedelta | None] = (
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

        By default, the accuracy is determined with respect to the reference values in the `reference_value` column.
        This column is created if it does not exist, or if one of the following reference parameters is not None.
        By default, the reference values are the most recent beliefs held by the same source.
        - Optionally, set a reference belief horizon to determine accuracy with respect to beliefs at a specific horizon.
        - Optionally, set a reference source to determine accuracy with respect to beliefs held by a specific source.
        These options allow defining what is considered to be true at a certain time after an event.

        By default, the mean absolute error (MAE), the mean absolute percentage error (MAPE) and
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
        event_value_range: tuple[float | None, float | None] = (None, None),
    ) -> "alt.LayerChart":
        """Visualize the BeliefsDataFrame in an interactive Altair chart.

        :param show_accuracy: Set to False to plot time series data only
        :param active_fixed_viewpoint_selector: If true, fixed viewpoint beliefs can be selected
        :param reference_source: BeliefSource to indicate that
               the accuracy should be determined with respect to the beliefs held by the given source
        :param intuitive_forecast_horizon: If true, horizons are shown with respect to event start rather than knowledge time
        :param interpolate: If True, the time series chart shows a user-friendly interpolated line
               rather than more accurate stripes indicating average values
        :param event_value_range: Optionally set explicit limits on the range of event values (for axis scaling).
               For example:
               (0, 3)  # lower limit is 0, upper limit is 3
               (None, 3)  # lower limit is taken from the plotted data, upper limit is 3
               (0, None)  # lower limit is 0, upper limit is taken from the plotted data
               (bdf["event_value"].min(), bdf["event_value"].max())  # limits are taken from the event value range of bdf
               (None, None)  # default, limits are taken from the plotted data
        :returns: Altair chart object with a vega-lite representation (for more information, see reference below).

        >>> chart = df.plot(df.lineage.sources[0])
        >>> chart.save("chart.json")
        >>> chart.serve()

        References
        ----------
        Altair: Declarative Visualization in Python.
            https://altair-viz.github.io
        """
        from timely_beliefs.visualization import utils as visualization_utils

        return visualization_utils.plot(
            self,
            show_accuracy=show_accuracy,
            active_fixed_viewpoint_selector=active_fixed_viewpoint_selector,
            reference_source=reference_source,
            intuitive_forecast_horizon=intuitive_forecast_horizon,
            interpolate=interpolate,
            event_value_range=event_value_range,
        )

    @staticmethod
    def plot_ridgeline_fixed_viewpoint(
        reference_time: datetime,
        df: "BeliefsDataFrame",
        future_only: bool = False,
        distribution: str = "uniform",
        event_value_window: tuple[float, float] = None,
    ) -> "alt.FacetChart":
        """Create a ridgeline plot of the latest beliefs held at a certain reference time.

        :param reference_time: datetime, reference to determine belief horizons
        :param df: BeliefsDataFrame
        :param future_only: if True mask the past
        :param distribution: string, distribution name to use (discrete, normal or uniform)
        :param event_value_window: optional tuple specifying an event value window for the x-axis
               (e.g. plot temperatures between -1 and 21 degrees Celsius)
        """
        from timely_beliefs.visualization import utils as visualization_utils

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
        event_value_window: tuple[float, float] = None,
    ) -> "alt.FacetChart":
        """Create a ridgeline plot of the belief history about a specific event.

        :param event_start: datetime, indicating the start time of the event for which to plot the belief history
        :param df: BeliefsDataFrame
        :param past_only: if True mask the future (i.e. mask any updates of beliefs after knowledge time)
        :param distribution: string, distribution name to use (discrete, normal or uniform)
        :param event_value_window: optional tuple specifying an event value window for the x-axis
               (e.g. plot temperatures between -1 and 21 degrees Celsius)
        """
        from timely_beliefs.visualization import utils as visualization_utils

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
        return_reference_type: str = "full",
    ) -> "BeliefsDataFrame":
        """Add a column with reference values.
        By default, the reference will be the probabilistic value of the most recent belief held by the same source.
        To set a deterministic reference, use either return_expected_value or return_middle_value.
        Optionally, set a reference belief horizon.
        Optionally, set a reference source present in the BeliefsDataFrame.
        These options allow defining what is considered to be true at a certain time after an event.

        :param reference_belief_time: optional datetime to indicate that
               the accuracy should be determined with respect to the latest belief held at this time
        :param reference_belief_horizon: optional timedelta to indicate that
               the accuracy should be determined with respect to the latest belief at this duration past knowledge time
        :param reference_source: optional BeliefSource to indicate that
               the accuracy should be determined with respect to the beliefs held by the given source
        :param return_reference_type: valid strings are:
               - "full": a probabilistic reference using the full distribution
               - "mean": a deterministic reference using the mean value
               - "median": a deterministic reference using the median value
        """

        df = self

        reference_df = df.groupby(level="event_start", group_keys=True).apply(
            lambda x: belief_utils.set_reference(
                x,
                reference_belief_time=reference_belief_time,
                reference_belief_horizon=reference_belief_horizon,
                reference_source=reference_source,
                return_reference_type=return_reference_type,
            )
        )

        # Concat to add a column while keeping original cp index level
        if return_reference_type != "full":
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

    def __repr__(self):
        """Add the sensor and event resolution to the string representation of the BeliefsDataFrame."""
        return super().__repr__() + "\n" + meta_repr(self)


def set_columns_and_indices_for_empty_frame(df, columns, indices, default_types):
    """Set appropriate columns and indices for the empty BeliefsDataFrame."""
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
    """Set the Sensor metadata (including timing properties of the sensor)."""

    # https://github.com/SeitaBV/timely-beliefs/issues/145
    # if not isinstance(sensor, Sensor):
    #     import warnings
    #
    #     warnings.warn(
    #         "'sensor' field needs to be of type 'Sensor'. This constraint will be enforced in an upcoming version.",
    #         FutureWarning,
    #         # stacklevel=6,
    #     )
    #
    #     # TODO: raise ValueError(...)

    df.sensor = sensor
    df.event_resolution = (
        event_resolution
        if event_resolution
        else sensor.event_resolution
        if sensor
        else None
    )


def downsample_beliefs_data_frame(
    df: BeliefsDataFrame, event_resolution: timedelta, col_att_dict: dict[str, str]
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
