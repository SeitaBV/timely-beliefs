from datetime import datetime, timedelta

from pytz import utc

from timely_beliefs.sensors.func_store.knowledge_horizons import (
    timedelta_x_days_ago_at_y_oclock
)


def test_dst():
    tz_str = "Europe/Amsterdam"

    # Before daylight saving time starts
    event_start = datetime(2018, 3, 25, 0, tzinfo=utc)
    assert timedelta_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=13
    )  # 12 + 1 hour difference of Amsterdam with UTC

    # Transition to daylight saving time
    event_start = datetime(2018, 3, 25, 6, tzinfo=utc)
    assert timedelta_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=19
    )  # 18 + 1 hour difference of Amsterdam with UTC

    # After daylight saving time started
    event_start = datetime(2018, 3, 26, 0, tzinfo=utc)
    assert timedelta_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=14
    )  # 12 + 2 hour difference of Amsterdam with UTC

    # Before daylight saving time ends
    event_start = datetime(2018, 10, 28, 0, tzinfo=utc)
    assert timedelta_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=14
    )  # 12 + 2 hour difference of Amsterdam with UTC

    # Transition from daylight saving time
    event_start = datetime(2018, 10, 28, 6, tzinfo=utc)
    assert timedelta_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=20
    )  # 18 + 2 hour difference of Amsterdam with UTC

    # After daylight saving time ended
    event_start = datetime(2018, 10, 29, 0, tzinfo=utc)
    assert timedelta_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=13
    )  # 12 + 1 hour difference of Amsterdam with UTC


def test_dst_bounds():
    tz_str = "Europe/London"

    # Test bounds for Europe/London for double daylight saving time with respect to standard time
    timedelta_bounds = timedelta_x_days_ago_at_y_oclock(
        None, x=180, y=23.9999999999, z=tz_str
    )
    event_start = datetime(
        1947, 5, 12, 22, 0, 0, tzinfo=utc
    )  # Date with double daylight saving
    assert (
        timedelta_bounds[0]
        < timedelta_x_days_ago_at_y_oclock(
            event_start, x=180, y=23.9999999999, z=tz_str
        )
        < timedelta_bounds[1]
    )

    # Test bounds for Europe/London for standard time with respect to double daylight saving time
    timedelta_bounds = timedelta_x_days_ago_at_y_oclock(None, x=210, y=0, z=tz_str)
    event_start = datetime(1947, 11, 10, 23, 59, 59, tzinfo=utc)
    assert (
        timedelta_bounds[0]
        < timedelta_x_days_ago_at_y_oclock(event_start, x=210, y=0, z=tz_str)
        < timedelta_bounds[1]
    )
