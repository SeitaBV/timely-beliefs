from datetime import datetime, timedelta

from pytz import utc

from timely_beliefs.sensors.func_store.knowledge_horizons import (
    determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock,
    determine_knowledge_horizon_for_fixed_knowledge_time,
)


def test_fixed_knowledge_time():
    knowledge_time = datetime(2020, 11, 20, 0, tzinfo=utc)
    assert (
        determine_knowledge_horizon_for_fixed_knowledge_time(
            event_start=datetime(2020, 11, 19, 0, tzinfo=utc),
            knowledge_time=knowledge_time,
        )
        == timedelta(-1)
    )
    assert (
        determine_knowledge_horizon_for_fixed_knowledge_time(
            event_start=datetime(2020, 11, 20, 0, tzinfo=utc),
            knowledge_time=knowledge_time,
        )
        == timedelta(0)
    )
    assert (
        determine_knowledge_horizon_for_fixed_knowledge_time(
            event_start=datetime(2020, 11, 21, 0, tzinfo=utc),
            knowledge_time=knowledge_time,
        )
        == timedelta(1)
    )


def test_dst():
    tz_str = "Europe/Amsterdam"

    # Before daylight saving time starts
    event_start = datetime(2018, 3, 25, 0, tzinfo=utc)
    assert determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=13
    )  # 12 + 1 hour difference of Amsterdam with UTC

    # Transition to daylight saving time
    event_start = datetime(2018, 3, 25, 6, tzinfo=utc)
    assert determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=19
    )  # 18 + 1 hour difference of Amsterdam with UTC

    # After daylight saving time started
    event_start = datetime(2018, 3, 26, 0, tzinfo=utc)
    assert determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=14
    )  # 12 + 2 hour difference of Amsterdam with UTC

    # Before daylight saving time ends
    event_start = datetime(2018, 10, 28, 0, tzinfo=utc)
    assert determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=14
    )  # 12 + 2 hour difference of Amsterdam with UTC

    # Transition from daylight saving time
    event_start = datetime(2018, 10, 28, 6, tzinfo=utc)
    assert determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=20
    )  # 18 + 2 hour difference of Amsterdam with UTC

    # After daylight saving time ended
    event_start = datetime(2018, 10, 29, 0, tzinfo=utc)
    assert determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
        event_start, x=1, y=12, z=tz_str
    ) == timedelta(
        hours=13
    )  # 12 + 1 hour difference of Amsterdam with UTC


def test_dst_bounds():
    tz_str = "Europe/London"

    # Test bounds for Europe/London for double daylight saving time with respect to standard time
    timedelta_bounds = determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
        None, x=180, y=23.9999999999, z=tz_str, get_bounds=True
    )
    event_start = datetime(
        1947, 5, 12, 22, 0, 0, tzinfo=utc
    )  # Date with double daylight saving
    assert (
        timedelta_bounds[0]
        < determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
            event_start, x=180, y=23.9999999999, z=tz_str
        )
        < timedelta_bounds[1]
    )

    # Test bounds for Europe/London for standard time with respect to double daylight saving time
    timedelta_bounds = determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
        None, x=210, y=0, z=tz_str, get_bounds=True
    )
    event_start = datetime(1947, 11, 10, 23, 59, 59, tzinfo=utc)
    assert (
        timedelta_bounds[0]
        < determine_ex_ante_knowledge_horizon_for_x_days_ago_at_y_oclock(
            event_start, x=210, y=0, z=tz_str
        )
        < timedelta_bounds[1]
    )
