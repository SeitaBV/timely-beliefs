from typing import Any, Callable, Optional, Tuple, Union
from datetime import datetime, timedelta

from isodate import duration_isoformat
from sqlalchemy import Column, Integer, Interval, JSON, String
from sqlalchemy.ext.hybrid import hybrid_method

from timely_beliefs.base import Base
from timely_beliefs.utils import enforce_utc
from timely_beliefs.sensors.func_store.knowledge_horizons import constant_timedelta
from timely_beliefs.sensors.utils import (
    jsonify_time_dict,
    eval_verified_knowledge_horizon_fnc,
)


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
        self.name = name
        self.unit = unit
        self.timezone = timezone
        if event_resolution is None:
            event_resolution = timedelta(hours=0)
        self.event_resolution = event_resolution
        if knowledge_horizon is None:
            knowledge_horizon = -event_resolution
        if isinstance(knowledge_horizon, timedelta):
            self.knowledge_horizon_fnc = constant_timedelta.__name__
            self.knowledge_horizon_par = {
                constant_timedelta.__code__.co_varnames[-1]: duration_isoformat(
                    knowledge_horizon
                )
            }
        if isinstance(knowledge_horizon, Tuple):
            self.knowledge_horizon_fnc = knowledge_horizon[0].__name__
            self.knowledge_horizon_par = jsonify_time_dict(knowledge_horizon[1])

    @hybrid_method
    def knowledge_horizon(self, event_start: datetime = None) -> timedelta:
        event_start = enforce_utc(event_start)
        return eval_verified_knowledge_horizon_fnc(
            self.knowledge_horizon_fnc, self.knowledge_horizon_par, event_start
        )

    @hybrid_method
    def knowledge_time(self, event_start: datetime) -> datetime:
        event_start = enforce_utc(event_start)
        return event_start - self.knowledge_horizon(event_start)

    def __repr__(self):
        return "<Sensor: %s>" % self.name


class DBSensor(Base, Sensor):
    """Mixin class for a table with sensors.
    """

    __tablename__ = "sensor"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False, default="")
    unit = Column(String(80), nullable=False, default="")
    timezone = Column(String(80), nullable=False, default="UTC")
    event_resolution = Column(Interval(), nullable=False, default=timedelta(hours=0))
    knowledge_horizon_fnc = Column(String(80), nullable=False)
    knowledge_horizon_par = Column(JSON(), default={}, nullable=False)

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
        Sensor.__init__(self, name, unit, timezone, event_resolution, knowledge_horizon)
        Base.__init__(self)

    def __repr__(self):
        return "<DBSensor: %s (%s)>" % (self.id, self.name)
