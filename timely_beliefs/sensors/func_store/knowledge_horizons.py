"""Function store for computing knowledge horizons given a certain event start and resolution.
When passed get_bounds=True, these functions return bounds on the knowledge horizon,
i.e. a duration window in which the knowledge horizon must lie (e.g. between 0 and 2 days before the event start)."""
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

from timely_beliefs.sensors.utils import datetime_x_days_ago_at_y_oclock


def at_date(
    event_start: Optional[datetime],
    knowledge_time: datetime,
    get_bounds: bool = False,
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Compute the sensor's knowledge horizon to represent the event could be known since some fixed date
    (knowledge time).

    For example, can be used for a tariff, in which case the knowledge time is the contract date.

    :param event_start: start of the event, used as an anchor for determining the knowledge horizon.
    :param knowledge_time: datetime since which all events for the sensor could be known.
    :param get_bounds: if True, this function returns bounds on the possible return value.
    These bounds are normally useful for creating more efficient database queries when filtering by belief time.
    In this case, the knowledge horizon is unbounded.
    """
    if get_bounds:
        return timedelta.min, timedelta.max
    return event_start - knowledge_time


def ex_post(
    event_resolution: timedelta,
    ex_post_horizon: timedelta,
    get_bounds: bool = False,
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Compute the sensor's knowledge horizon to represent the event can be known some length of time after it ends.

    For example, for most physical events, events can be known when they end.
    Since we define a knowledge horizon as the duration from knowledge time to event start
    (i.e. how long before the event starts could the event be known),
    the knowledge horizon in this case is equal to minus the event resolution.

    :param event_resolution: resolution of the event, needed to re-anchor from event_end to event_start.
    :param ex_post_horizon: length of time after (the end of) the event.
    :param get_bounds: if True, this function returns bounds on the possible return value.
    These bounds are useful for creating more efficient database queries when filtering by belief time.

    """
    if get_bounds:
        return -event_resolution - ex_post_horizon, -event_resolution - ex_post_horizon
    return -event_resolution - ex_post_horizon


def ex_ante(
    ex_ante_horizon: timedelta,
    get_bounds: bool = False,
) -> Union[timedelta, Tuple[timedelta, timedelta]]:
    """Compute the sensor's knowledge horizon to represent the event can be known some length of time before it starts.

    :param ex_ante_horizon: length of time before (the start of) the event.
    :param get_bounds: if True, this function returns bounds on the possible return value.
    These bounds are useful for creating more efficient database queries when filtering by belief time.

    """
    if get_bounds:
        return ex_ante_horizon, ex_ante_horizon
    return ex_ante_horizon


def x_days_ago_at_y_oclock(
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


# todo: deprecate this function
def determine_knowledge_horizon_for_fixed_knowledge_time(
    event_start: Optional[datetime],
    *args,
    **kwargs,
):
    import warnings

    warnings.warn(
        "Function name will be replaced by shorthand. Replace with 'at_date' to suppress this warning.",
        FutureWarning,
    )
    return at_date(event_start, *args, **kwargs)


# todo: deprecate this function
def determine_ex_post_knowledge_horizon(
    event_resolution: timedelta,
    *args,
    **kwargs,
):
    import warnings

    warnings.warn(
        "Function name will be replaced by shorthand. Replace with 'ex_post' to suppress this warning.",
        FutureWarning,
    )
    return ex_post(event_resolution, *args, **kwargs)


# todo: deprecate this function
def determine_ex_ante_knowledge_horizon(
    *args,
    **kwargs,
):
    import warnings

    warnings.warn(
        "Function name will be replaced by shorthand. Replace with 'ex_ante' to suppress this warning.",
        FutureWarning,
    )
    return ex_ante(*args, **kwargs)


# todo: deprecate this function
def determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
    event_start: Optional[datetime],
    *args,
    **kwargs,
):
    import warnings

    warnings.warn(
        "Function name will be replaced by shorthand. Replace with 'x_days_ago_at_y_oclock' to suppress this warning.",
        FutureWarning,
    )
    return x_days_ago_at_y_oclock(event_start, *args, **kwargs)
