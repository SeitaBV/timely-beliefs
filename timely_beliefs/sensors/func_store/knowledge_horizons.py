"""Function store for computing knowledge horizons given a certain event start and resolution.
When passed an event_start = None, these functions return bounds on the knowledge horizon,  # todo: how do the bounds change when event_resolution != sensor.event_resolution
i.e. a duration window in which the knowledge horizon must lie (e.g. between 0 and 2 days before the event start)."""
from typing import Optional, Tuple, Union
from datetime import datetime, timedelta

from timely_beliefs.sensors.utils import datetime_x_days_ago_at_y_oclock


def timedelta_for_end_of_event(
    event_start: Optional[datetime], event_resolution: timedelta
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Knowledge horizon always matches the negative event_resolution.
    This knowledge horizon is anchored to the event_end.
    """
    if event_start is None:
        return -event_resolution, -event_resolution
    return -event_resolution


def constant_timedelta(
    event_start: Optional[datetime], knowledge_horizon: timedelta,
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Knowledge horizon is a constant timedelta.
    This knowledge horizon is anchored to the event_start.
    """
    if event_start is None:
        return knowledge_horizon, knowledge_horizon
    return knowledge_horizon


def timedelta_for_x_days_ago_at_y_oclock(
    event_start: Optional[datetime], x: int, y: Union[int, float], z: str,
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Knowledge horizon is with respect to a previous day at some hour.
    This knowledge horizon is anchored to the event_start.
    """
    if event_start is None:
        return (
            timedelta(days=x, hours=-y - 2),
            timedelta(days=x + 1, hours=-y + 2),
        )  # The 2's account for possible hour differences for double daylight saving time w.r.t. standard time
    return event_start - datetime_x_days_ago_at_y_oclock(event_start, x, y, z)
