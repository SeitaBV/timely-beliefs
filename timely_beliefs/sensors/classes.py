from typing import Any, Callable, Optional, Tuple, Union
from datetime import datetime, timedelta

from sqlalchemy import Column, Integer, Interval, JSON, String
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy.ext.declarative import declared_attr

from timely_beliefs.db_base import Base
from timely_beliefs.utils import enforce_tz
from timely_beliefs.sensors.func_store.knowledge_horizons import (
    determine_ex_ante_knowledge_horizon,
    determine_ex_post_knowledge_horizon,
)
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
            self.knowledge_horizon_fnc = determine_ex_post_knowledge_horizon.__name__
            knowledge_horizon_par = {
                determine_ex_post_knowledge_horizon.__code__.co_varnames[1]: timedelta(
                    hours=0
                )
            }
        elif isinstance(knowledge_horizon, timedelta):
            self.knowledge_horizon_fnc = determine_ex_ante_knowledge_horizon.__name__
            knowledge_horizon_par = {
                determine_ex_ante_knowledge_horizon.__code__.co_varnames[
                    0
                ]: knowledge_horizon
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


class DBSensor(Base, Sensor):
    """Mixin class for a table with sensors.
    """

    __tablename__ = "sensor"

    # two columns for db purposes: id is a row identifier
    id = Column(Integer, primary_key=True)
    # type is useful so we can use polymorphic inheritance
    # (https://docs.sqlalchemy.org/en/13/orm/inheritance.html#single-table-inheritance)
    type = Column(String(50), nullable=False)

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

    @declared_attr
    def __mapper_args__(self):
        if self.__name__ == "DBSensor":
            return {"polymorphic_on": self.type, "polymorphic_identity": "sensor"}
        else:
            return {"polymorphic_identity": self.__name__}
