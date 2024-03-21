"""Function store for computing knowledge horizons given a certain event start and resolution.
When passed get_bounds=True, these functions return bounds on the knowledge horizon,
i.e. a duration window in which the knowledge horizon must lie (e.g. between 0 and 2 days before the event start)."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from timely_beliefs.sensors.func_store.utils import (
    datetime_x_days_ago_at_y_oclock,
    x_years_ago_at_date_datetime,
)


def at_date(
    event_start: datetime | pd.DatetimeIndex | None,
    knowledge_time: datetime,
    get_bounds: bool = False,
) -> timedelta | pd.TimedeltaIndex | tuple[timedelta, timedelta]:
    """Compute the sensor's knowledge horizon to represent the event could be known since some fixed date
    (knowledge time).

    For example, can be used for a tariff, in which case the knowledge time is the contract date.

    :param event_start:     Start of the event, used as an anchor for determining the knowledge horizon.
    :param knowledge_time:  Datetime since which all events for the sensor could be known.
    :param get_bounds:      If True, this function returns bounds on the possible return value.
                            These bounds are normally useful for creating more efficient database queries when filtering by belief time.
                            In this case, the knowledge horizon is unbounded.
    """
    if get_bounds:
        return timedelta.min, timedelta.max
    return event_start - knowledge_time.astimezone(event_start.tzinfo)


def x_years_ago_at_date(
    event_start: datetime | pd.DatetimeIndex,
    day: int,
    month: int,
    x: int = 1,
    get_bounds: bool = False,
    z: str | None = None,
) -> timedelta | pd.TimedeltaIndex | tuple[timedelta, timedelta]:
    """Compute the sensor's knowledge horizon to represent the event could be known since some date, `x` years ago.

    For example, it can be used for a tax rate that changes annually and with a known publication date.

    :param event_start:     Start of the event, used as an anchor for determining the knowledge horizon.
    :param day:             Reference day of the month of the annual date to compare against.
    :param month:           The month of the annual date to compare against.
    :param x:               The number of years to shift the reference date to.
    :param z:           Timezone string.
    :param get_bounds:      If True, this function returns bounds on the possible return value.
                            These bounds are normally useful for creating more efficient database queries when filtering by belief time.
    """

    if x < 0:
        raise ValueError("Negative value for `x` not supported.")

    if get_bounds:
        return timedelta(days=(x - 1) * 366 - 3), timedelta(days=(x + 1) * 366 + 1)

    if isinstance(event_start, datetime):
        return x_years_ago_at_date_datetime(event_start, day, month, x, z)
    else:
        return event_start.map(
            lambda _event_start: x_years_ago_at_date_datetime(
                _event_start, day, month, x, z
            )
        )


def ex_post(
    event_resolution: timedelta,
    ex_post_horizon: timedelta,
    get_bounds: bool = False,
) -> timedelta | tuple[timedelta, timedelta]:
    """Compute the sensor's knowledge horizon to represent the event can be known some length of time after it ends.

    For example, for most physical events, events can be known when they end.
    Since we define a knowledge horizon as the duration from knowledge time to event start
    (i.e. how long before the event starts could the event be known),
    the knowledge horizon in this case is equal to minus the event resolution.

    :param event_resolution:    Resolution of the event, needed to re-anchor from event_end to event_start.
    :param ex_post_horizon:     Length of time after (the end of) the event.
    :param get_bounds:          If True, this function returns bounds on the possible return value.
                                These bounds are useful for creating more efficient database queries when filtering by belief time.
    """
    if get_bounds:
        return -event_resolution - ex_post_horizon, -event_resolution - ex_post_horizon
    return -event_resolution - ex_post_horizon


def ex_ante(
    ex_ante_horizon: timedelta,
    get_bounds: bool = False,
) -> timedelta | tuple[timedelta, timedelta]:
    """Compute the sensor's knowledge horizon to represent the event can be known some length of time before it starts.

    :param ex_ante_horizon:     Length of time before (the start of) the event.
    :param get_bounds:          If True, this function returns bounds on the possible return value.
                                These bounds are useful for creating more efficient database queries when filtering by belief time.

    """
    if get_bounds:
        return ex_ante_horizon, ex_ante_horizon
    return ex_ante_horizon


def x_days_ago_at_y_oclock(
    event_start: datetime | pd.DatetimeIndex | None,
    x: int,
    y: int | float,
    z: str,
    get_bounds: bool = False,
) -> timedelta | pd.TimedeltaIndex | tuple[timedelta, timedelta]:
    """Compute the sensor's knowledge horizon to represent the event can be known some previous day at some hour.

    :param event_start: Start of the event, used as an anchor for determining the knowledge time.
    :param x:           Number of days before the day the event starts.
    :param y:           Hour of the day.
    :param z:           Timezone string.
    :param get_bounds:  If True, this function returns bounds on the possible return value.
                        These bounds are useful for creating more efficient database queries when filtering by belief time.

    """
    if get_bounds:
        return (
            timedelta(days=x, hours=-y - 2),
            timedelta(days=x + 1, hours=-y + 2),
        )  # The 2's account for possible hour differences for double daylight saving time w.r.t. standard time
    return event_start - datetime_x_days_ago_at_y_oclock(event_start, x, y, z)


# aliases
x_days_ago_at_y_o_clock = x_days_ago_at_y_oclock  # forgive a common typo
