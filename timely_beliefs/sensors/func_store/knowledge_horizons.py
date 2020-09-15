"""Function store for computing knowledge horizons given a certain event start and resolution.
When passed get_bounds=True, these functions return bounds on the knowledge horizon,
i.e. a duration window in which the knowledge horizon must lie (e.g. between 0 and 2 days before the event start)."""
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

from timely_beliefs.sensors.utils import datetime_x_days_ago_at_y_oclock


def determine_ex_post_knowledge_horizon(
    event_resolution: timedelta, ex_post_horizon: timedelta, get_bounds: bool = False,
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Compute the sensor's knowledge horizon to represent the event can be known some length of time after it ends.

    Since we define a knowledge horizon as the time it takes before (the event starts) the event could be known,
    the knowledge horizon in this case is equal to minus the event resolution.

    :param event_resolution: resolution of the event, needed to re-anchor from event_end to event_start.
    :param ex_post_horizon: length of time after (the end of) the event.
    :param get_bounds: if True, this function returns bounds on the possible return value.
    These bounds are useful for creating more efficient database queries when filtering by belief time.

    """
    if get_bounds:
        return -event_resolution - ex_post_horizon, -event_resolution - ex_post_horizon
    return -event_resolution - ex_post_horizon


def determine_ex_ante_knowledge_horizon(
    ex_ante_horizon: timedelta, get_bounds: bool = False,
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Compute the sensor's knowledge horizon to represent the event can be known some length of time before it starts.

    :param ex_ante_horizon: length of time before (the start of) the event.
    :param get_bounds: if True, this function returns bounds on the possible return value.
    These bounds are useful for creating more efficient database queries when filtering by belief time.

    """
    if get_bounds:
        return ex_ante_horizon, ex_ante_horizon
    return ex_ante_horizon


def determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
    event_start: Optional[datetime],
    x: int,
    y: Union[int, float],
    z: str,
    get_bounds: bool = False,
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Compute the sensor's knowledge horizon to represent the event can be known some previous day at some hour.

    :param event_start: start of the event, used as an anchor for determining the knowledge time.
    :param x: number of days before the day the event starts.
    :param y: hour of the day.
    :param z: timezone string.
    :param get_bounds: if True, this function returns bounds on the possible return value.
    These bounds are useful for creating more efficient database queries when filtering by belief time.

    """
    if get_bounds:
        return (
            timedelta(days=x, hours=-y - 2),
            timedelta(days=x + 1, hours=-y + 2),
        )  # The 2's account for possible hour differences for double daylight saving time w.r.t. standard time
    return event_start - datetime_x_days_ago_at_y_oclock(event_start, x, y, z)
