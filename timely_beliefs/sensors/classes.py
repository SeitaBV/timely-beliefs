from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Tuple, Union

from sqlalchemy import JSON, Column, Integer, Interval, String
from sqlalchemy.ext.hybrid import hybrid_method

from timely_beliefs.db_base import Base
from timely_beliefs.sensors.func_store import knowledge_horizons
from timely_beliefs.sensors.utils import (
    eval_verified_knowledge_horizon_fnc,
    jsonify_time_dict,
)
from timely_beliefs.utils import enforce_tz


class Sensor(object):
    """Sensors of physical or economical events, e.g. a thermometer or price index.

    Todo: describe init parameters
    Todo: describe default sensor
    """

    name: str
    unit: str
    timezone: str
    event_resolution: timedelta
    knowledge_horizon_fnc: str
    knowledge_horizon_par: dict

    def __init__(
        self,
        name: str = "",
        unit: str = "",
        timezone: str = "UTC",
        event_resolution: Optional[timedelta] = None,
        knowledge_horizon: Optional[
            Union[timedelta, Tuple[Callable[[datetime, Any], timedelta], dict]]
        ] = None,
    ):
        if name == "":
            raise Exception("Please give this sensor a name to be identifiable.")
        self.name = name
        self.unit = unit
        self.timezone = timezone
        if event_resolution is None:
            event_resolution = timedelta(hours=0)
        self.event_resolution = event_resolution
        if knowledge_horizon is None:
            # Set an appropriate knowledge horizon for physical sensors, representing ex-post knowledge time.
            self.knowledge_horizon_fnc = knowledge_horizons.ex_post.__name__
            knowledge_horizon_par = {
                knowledge_horizons.ex_post.__code__.co_varnames[1]: timedelta(hours=0)
            }
        elif isinstance(knowledge_horizon, timedelta):
            self.knowledge_horizon_fnc = knowledge_horizons.ex_ante.__name__
            knowledge_horizon_par = {
                knowledge_horizons.ex_ante.__code__.co_varnames[0]: knowledge_horizon
            }
        elif isinstance(knowledge_horizon, Tuple):
            self.knowledge_horizon_fnc = knowledge_horizon[0].__name__
            knowledge_horizon_par = knowledge_horizon[1]
        else:
            raise ValueError(
                "Knowledge horizon should be a timedelta or a tuple containing a knowledge horizon function (import from function store) and a kwargs dict."
            )
        self.knowledge_horizon_par = jsonify_time_dict(knowledge_horizon_par)

    @hybrid_method
    def knowledge_horizon(
        self, event_start: datetime, event_resolution: Optional[timedelta] = None
    ) -> timedelta:
        event_start = enforce_tz(event_start, "event_start")
        if event_resolution is None:
            event_resolution = self.event_resolution
        return eval_verified_knowledge_horizon_fnc(
            self.knowledge_horizon_fnc,
            self.knowledge_horizon_par,
            event_start,
            event_resolution,
        )

    @hybrid_method
    def knowledge_time(
        self, event_start: datetime, event_resolution: Optional[timedelta] = None
    ) -> datetime:
        event_start = enforce_tz(event_start, "event_start")
        if event_resolution is None:
            event_resolution = self.event_resolution
        return event_start - self.knowledge_horizon(event_start, event_resolution)

    def __repr__(self):
        return "<Sensor: %s>" % self.name


class SensorDBMixin(Sensor):
    """
    Mixin class for a table with sensors.
    """

    id = Column(Integer, primary_key=True)
    # overwriting name as db field
    name = Column(String(120), nullable=False, default="")
    unit = Column(String(80), nullable=False, default="")
    timezone = Column(String(80), nullable=False, default="UTC")
    event_resolution = Column(Interval(), nullable=False, default=timedelta(hours=0))
    knowledge_horizon_fnc = Column(
        String(80), nullable=False, default=knowledge_horizons.ex_post.__name__
    )
    knowledge_horizon_par = Column(
        JSON(),
        nullable=False,
        default=jsonify_time_dict(
            {knowledge_horizons.ex_post.__code__.co_varnames[1]: timedelta(hours=0)}
        ),
    )

    def __init__(
        self,
        name: str,
        unit: str = "",
        timezone: str = "UTC",
        event_resolution: Optional[timedelta] = None,
        knowledge_horizon: Optional[
            Union[timedelta, Tuple[Callable[[datetime, Any], timedelta], dict]]
        ] = None,
    ):
        Sensor.__init__(self, name, unit, timezone, event_resolution, knowledge_horizon)


class DBSensor(Base, SensorDBMixin):
    """Database class for a table with sensors."""

    __tablename__ = "sensor"

    def __init__(
        self,
        name: str = "",
        unit: str = "",
        timezone: str = "UTC",
        event_resolution: Optional[timedelta] = None,
        knowledge_horizon: Optional[
            Union[timedelta, Tuple[Callable[[datetime, Any], timedelta], dict]]
        ] = None,
    ):
        SensorDBMixin.__init__(
            self, name, unit, timezone, event_resolution, knowledge_horizon
        )
        Base.__init__(self)

    def __repr__(self):
        return "<DBSensor: %s (%s)>" % (self.id, self.name)
