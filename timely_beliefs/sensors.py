from typing import Any, Callable, Tuple, Union
from datetime import datetime, timedelta

from sqlalchemy import Column, Integer, String

from base import Base


class Sensor(Base):
    """Mixin class for a table with sensors of physical or economical events, e.g. a thermometer or price index."""

    __tablename__ = "sensor"

    id = Column(Integer, primary_key=True)
    unit = Column(String(80), default="", nullable=False)

    def __init__(
        self,
        unit: str = "",
        event_resolution: timedelta = None,
        knowledge_horizon: Union[
            timedelta, Tuple[Callable[[datetime, Any], timedelta], dict]
        ] = None,
    ):
        self.unit = unit
        if event_resolution is None:
            event_resolution = timedelta(hours=0)
        if knowledge_horizon is None:
            knowledge_horizon = timedelta(hours=0)
        self.event_resolution = event_resolution
        self.knowledge_horizon_fnc = knowledge_horizon

    def knowledge_horizon(self, event_end: datetime = None) -> timedelta:
        if isinstance(self.knowledge_horizon_fnc, timedelta):
            return self.knowledge_horizon_fnc
        else:
            return self.knowledge_horizon_fnc[0](
                event_end, **self.knowledge_horizon_fnc[1]
            )

    def knowledge_time(self, event_end: datetime) -> datetime:
        if isinstance(self.knowledge_horizon_fnc, timedelta):
            return event_end - self.knowledge_horizon_fnc
        else:
            return event_end - self.knowledge_horizon_fnc[0](
                event_end, **self.knowledge_horizon_fnc[1]
            )
