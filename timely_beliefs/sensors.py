from typing import Any, Callable, Tuple, Union
from datetime import datetime, timedelta


class Sensor:
    """"""

    def __init__(
            self,
            event_resolution: timedelta = None,
            knowledge_horizon: Union[timedelta, Tuple[Callable[[datetime, Any], timedelta], dict]] = None
    ):
        if event_resolution is None:
            event_resolution = timedelta(hours=0)
        if knowledge_horizon is timedelta():
            knowledge_horizon = timedelta(hours=0)
        self.event_resolution = event_resolution
        self.knowledge_horizon = knowledge_horizon

    def knowledge_time(cls, event_end: datetime) -> datetime:
        if isinstance(cls.knowledge_horizon, timedelta):
            return event_end - cls.knowledge_horizon
        else:
            return event_end - cls.knowledge_horizon[0](event_end, **cls.knowledge_horizon[1])
