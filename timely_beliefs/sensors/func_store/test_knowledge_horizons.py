from datetime import datetime, timedelta

import pandas as pd
from pandas.testing import assert_index_equal
from pytz import utc

from timely_beliefs.sensors.func_store.knowledge_horizons import (
    at_date,
    ex_ante,
    ex_post,
    x_days_ago_at_y_oclock,
)


def test_ex_ante_knowledge_horizon():
    """Check definition of knowledge horizon for ex-ante knowledge."""
    assert ex_ante(
        ex_ante_horizon=timedelta(minutes=15),
    ) == timedelta(minutes=15), "The event should be known 15 minutes before it starts."


def test_ex_post_knowledge_horizon():
    """Check definition of knowledge horizon for ex-post knowledge."""
    assert ex_post(
        event_resolution=timedelta(minutes=15),
        ex_post_horizon=timedelta(minutes=-5),
    ) == timedelta(minutes=-10), "The event should be known 10 minutes after it starts."


def test_fixed_knowledge_time():
    """Check definition of knowledge horizon for events known at a fixed date."""
    knowledge_time = datetime(2020, 11, 20, 0, tzinfo=utc)
    assert at_date(
        event_start=datetime(2020, 11, 19, 0, tzinfo=utc),
        knowledge_time=knowledge_time,
    ) == timedelta(-1)
    assert at_date(
        event_start=datetime(2020, 11, 20, 0, tzinfo=utc),
        knowledge_time=knowledge_time,
    ) == timedelta(0)
    assert at_date(
        event_start=datetime(2020, 11, 21, 0, tzinfo=utc),
        knowledge_time=knowledge_time,
    ) == timedelta(1)

    # Repeat test with pd.DatetimeIndex instead
    event_start = pd.date_range("2020-11-19", "2020-11-21", tz="utc")
    assert_index_equal(
        at_date(
            event_start=event_start,
            knowledge_time=knowledge_time,
        ),
        pd.TimedeltaIndex([timedelta(-1), timedelta(0), timedelta(1)]),
    )


def test_dst():
    """Check definition of knowledge horizon for events known x days ago at y o'clock in some timezone z,
    especially around daylight savings time (DST) transitions."""
    tz_str = "Europe/Amsterdam"

    # Before daylight saving time starts
    event_start = datetime(2018, 3, 25, 0, tzinfo=utc)
    assert x_days_ago_at_y_oclock(event_start, x=1, y=12, z=tz_str) == timedelta(
        hours=13
    )  # 12 + 1 hour difference of Amsterdam with UTC

    # Transition to daylight saving time
    event_start = datetime(2018, 3, 25, 6, tzinfo=utc)
    assert x_days_ago_at_y_oclock(event_start, x=1, y=12, z=tz_str) == timedelta(
        hours=19
    )  # 18 + 1 hour difference of Amsterdam with UTC

    # After daylight saving time started
    event_start = datetime(2018, 3, 26, 0, tzinfo=utc)
    assert x_days_ago_at_y_oclock(event_start, x=1, y=12, z=tz_str) == timedelta(
        hours=14
    )  # 12 + 2 hour difference of Amsterdam with UTC

    # Before daylight saving time ends
    event_start = datetime(2018, 10, 28, 0, tzinfo=utc)
    assert x_days_ago_at_y_oclock(event_start, x=1, y=12, z=tz_str) == timedelta(
        hours=14
    )  # 12 + 2 hour difference of Amsterdam with UTC

    # Transition from daylight saving time
    event_start = datetime(2018, 10, 28, 6, tzinfo=utc)
    assert x_days_ago_at_y_oclock(event_start, x=1, y=12, z=tz_str) == timedelta(
        hours=20
    )  # 18 + 2 hour difference of Amsterdam with UTC

    # After daylight saving time ended
    event_start = datetime(2018, 10, 29, 0, tzinfo=utc)
    assert x_days_ago_at_y_oclock(event_start, x=1, y=12, z=tz_str) == timedelta(
        hours=13
    )  # 12 + 1 hour difference of Amsterdam with UTC

    # Repeat test with pd.DatetimeIndex instead
    event_start = pd.DatetimeIndex(
        [
            "2018-03-25T00:00",
            "2018-03-25T06:00",
            "2018-03-26T00:00",
            "2018-10-28T00:00",
            "2018-10-28T06:00",
            "2018-10-29T00:00",
        ],
        tz="utc",
    )
    assert_index_equal(
        x_days_ago_at_y_oclock(
            event_start=event_start,
            x=1,
            y=12,
            z=tz_str,
        ),
        pd.TimedeltaIndex(
            [
                timedelta(hours=13),
                timedelta(hours=19),
                timedelta(hours=14),
                timedelta(hours=14),
                timedelta(hours=20),
                timedelta(hours=13),
            ]
        ),
    )


def test_dst_bounds():
    """Check definition of bounds on the knowledge horizon for events known x days ago at y o'clock in some timezone z,
    especially around daylight savings time (DST) transitions."""
    tz_str = "Europe/London"

    # Test bounds for Europe/London for double daylight saving time with respect to standard time
    timedelta_bounds = x_days_ago_at_y_oclock(
        None, x=180, y=23.9999999999, z=tz_str, get_bounds=True
    )
    event_start = datetime(
        1947, 5, 12, 22, 0, 0, tzinfo=utc
    )  # Date with double daylight saving
    assert (
        timedelta_bounds[0]
        < x_days_ago_at_y_oclock(event_start, x=180, y=23.9999999999, z=tz_str)
        < timedelta_bounds[1]
    )

    # Test bounds for Europe/London for standard time with respect to double daylight saving time
    timedelta_bounds = x_days_ago_at_y_oclock(
        None, x=210, y=0, z=tz_str, get_bounds=True
    )
    event_start = datetime(1947, 11, 10, 23, 59, 59, tzinfo=utc)
    assert (
        timedelta_bounds[0]
        < x_days_ago_at_y_oclock(event_start, x=210, y=0, z=tz_str)
        < timedelta_bounds[1]
    )
