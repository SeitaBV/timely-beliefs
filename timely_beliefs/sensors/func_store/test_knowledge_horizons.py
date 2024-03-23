from datetime import datetime, timedelta

import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from pytz import timezone, utc

from timely_beliefs.sensors.func_store.knowledge_horizons import (
    at_date,
    ex_ante,
    ex_post,
    x_days_ago_at_y_oclock,
    x_years_ago_at_date,
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


def test_x_years_ago_at_date():
    """Check definition of knowledge horizon for events known at a fixed date annually."""

    knowledge_func_params = dict(x=1, month=11, day=20, z="UTC")

    # Events that occur before the reference
    # year 2024 is leap
    assert x_years_ago_at_date(
        event_start=datetime(2024, 11, 19, 1, tzinfo=utc), **knowledge_func_params
    ) == timedelta(
        days=365, hours=1
    )  # 366 days - 1

    # year 2025 is not leap, but 2024 is
    assert x_years_ago_at_date(
        event_start=datetime(2025, 11, 19, 2, tzinfo=utc), **knowledge_func_params
    ) == timedelta(
        days=364, hours=2
    )  # 365 - 1

    # year 2023 is not leap and 2022 neither
    assert x_years_ago_at_date(
        event_start=datetime(2022, 11, 19, 2, tzinfo=utc), **knowledge_func_params
    ) == timedelta(
        days=364, hours=2
    )  # 365 - 1

    # Events that occur after the reference
    assert x_years_ago_at_date(
        event_start=datetime(2021, 11, 21, 3, tzinfo=utc), **knowledge_func_params
    ) == timedelta(
        days=366, hours=3
    )  # 365 + 1

    assert x_years_ago_at_date(
        event_start=datetime(2021, 11, 21, 4, tzinfo=utc), **knowledge_func_params
    ) == timedelta(
        days=366, hours=4
    )  # 365 + 1

    assert x_years_ago_at_date(
        event_start=datetime(2020, 11, 21, 4, tzinfo=utc), **knowledge_func_params
    ) == timedelta(
        days=367, hours=4
    )  # 366 (leap year) + 1

    # Repeat test with pd.DatetimeIndex instead
    event_start = pd.DatetimeIndex(
        [
            "2024-11-19T01:00:00",
            "2025-11-19T02:00:00",
            "2022-11-19T02:00:00",
            "2021-11-21T03:00:00",
            "2021-11-21T04:00:00",
        ],
        tz="utc",
    )
    assert_index_equal(
        x_years_ago_at_date(event_start=event_start, **knowledge_func_params),
        pd.TimedeltaIndex(
            [
                timedelta(days=365, hours=1),
                timedelta(days=364, hours=2),
                timedelta(days=364, hours=2),
                timedelta(days=366, hours=3),
                timedelta(days=366, hours=4),
            ]
        ),
    )

    knowledge_func_params_2_years = dict(x=2, month=11, day=20, z="UTC")

    # Check years parameter
    assert x_years_ago_at_date(
        event_start=datetime(2024, 11, 19, 1, tzinfo=utc),
        **knowledge_func_params_2_years,
    ) == timedelta(
        days=2 * 365, hours=1
    )  # 365 days + 366 days - 1 day


def test_x_years_ago_at_date_with_dst():
    """Check x_years_ago_at_date specifically against Daylight Savings Transition.

    Note that 2023-03-28 lies after the spring DST transition, and 2024-03-28 lies before the spring DST transition.
    - 2023-03-26
    - 2023-10-29
    - 2024-03-30
    """

    knowledge_func_params = dict(
        x=1, month=3, day=28, z="Europe/Amsterdam"
    )  # before first DST transition 2024
    assert x_years_ago_at_date(
        event_start=timezone("Europe/Amsterdam").localize(datetime(2024, 3, 28, 0)),
        **knowledge_func_params,
    ) == timedelta(
        days=366, hours=1
    )  # 365 + 1 day (because of the leap day on 2024-02-29) + 1 hour (fall transition)

    # Try 4 days later, at which we crossed the spring DST transition
    assert x_years_ago_at_date(
        event_start=timezone("Europe/Amsterdam").localize(datetime(2024, 4, 1, 0)),
        **knowledge_func_params,
    ) == timedelta(
        days=370, hours=0
    )  # 0 hours (fall and spring transitions cancelled each other out)

    # Repeat test with pd.DatetimeIndex instead
    event_start = pd.DatetimeIndex(
        [
            "2024-03-28T00:00:00",
            "2024-04-01T00:00:00",
        ],
        tz="Europe/Amsterdam",
    )
    assert_index_equal(
        x_years_ago_at_date(event_start=event_start, **knowledge_func_params),
        pd.TimedeltaIndex(
            [
                timedelta(days=366, hours=1),
                timedelta(days=370, hours=0),
            ]
        ),
    )


@pytest.mark.parametrize(
    "event_start",
    [
        timezone("Europe/Amsterdam").localize(datetime(2024, 1, 1, 0)),
        timezone("Europe/Amsterdam").localize(datetime(2024, 12, 31, 23, 59, 59)),
    ],
)
@pytest.mark.parametrize("years", list(range(1, 6)))
def test_x_years_ago_at_date_bounds(event_start, years):
    knowledge_func_params = dict(x=years, month=12, day=31, z="Europe/Amsterdam")

    timedelta_bounds = x_years_ago_at_date(
        event_start, get_bounds=True, **knowledge_func_params
    )

    assert (
        timedelta_bounds[0]
        <= x_years_ago_at_date(event_start, **knowledge_func_params)
        <= timedelta_bounds[1]
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
