from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from pytz import timezone


def datetime_x_days_ago_at_y_oclock(
    tz_aware_original_time: datetime | pd.DatetimeIndex,
    x: int,
    y: int | float,
    z: str,
) -> datetime | pd.DatetimeIndex:
    """Returns the datetime x days ago a y o'clock as determined from the perspective of timezone z."""
    if isinstance(y, float):
        micros = int(y * 60 * 60 * 10**6)
        s, micros = divmod(micros, 10**6)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
    else:
        micros = 0
        s = 0
        m = 0
        h = y
    tz = timezone(z)
    original_tz = tz_aware_original_time.tzinfo
    if isinstance(tz_aware_original_time, datetime):
        tz_naive_original_time = tz_aware_original_time.astimezone(tz).replace(
            tzinfo=None
        )
        tz_naive_earlier_time = (tz_naive_original_time - pd.Timedelta(days=x)).replace(
            hour=h, minute=m, second=s, microsecond=micros
        )
        tz_aware_earlier_time = tz.localize(tz_naive_earlier_time).astimezone(
            original_tz
        )
    else:
        tz_naive_original_time = tz_aware_original_time.tz_convert(tz).tz_localize(None)
        tz_naive_earlier_time = (tz_naive_original_time - pd.Timedelta(days=x)).floor(
            "D"
        ) + pd.Timedelta(hours=h, minutes=m, seconds=s, microseconds=micros)
        tz_aware_earlier_time = tz_naive_earlier_time.tz_localize(tz).tz_convert(
            original_tz
        )

    return tz_aware_earlier_time


def datetime_x_years_ago_at_date(
    tz_aware_original_time: datetime | pd.DatetimeIndex,
    x: int,
    day: int,
    month: int,
    z: str,
) -> timedelta:
    """Returns the datetime x years ago at the midnight start of the given date, from the perspective of timezone z."""
    tz = timezone(z)
    original_tz = tz_aware_original_time.tzinfo
    micros = 0
    s = 0
    m = 0
    h = 0
    if isinstance(tz_aware_original_time, datetime):
        tz_naive_original_time = tz_aware_original_time.astimezone(tz).replace(
            tzinfo=None
        )
        tz_naive_earlier_time = (
            pd.Timestamp(tz_naive_original_time).to_period("1Y").to_timestamp()
            - pd.DateOffset(years=x)
        ).replace(month=month, day=day, hour=h, minute=m, second=s, microsecond=micros)
        tz_aware_earlier_time = tz.localize(tz_naive_earlier_time).astimezone(
            original_tz
        )
    else:
        tz_aware_earlier_time = tz_aware_original_time.to_period(
            "1Y"
        ).to_timestamp().tz_localize(tz_aware_original_time.tz) + pd.DateOffset(
            month=month, day=day, years=-x
        )

    return tz_aware_earlier_time
