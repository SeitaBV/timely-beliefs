from datetime import datetime, timedelta
from typing import Union

from pytz import timezone


def datetime_x_days_ago_at_y_oclock(
    tz_aware_original_time: datetime, x: int, y: Union[int, float], z: str
) -> datetime:
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
    tz_naive_original_time = tz_aware_original_time.astimezone(tz).replace(tzinfo=None)
    tz_naive_earlier_time = tz_naive_original_time.replace(
        hour=h, minute=m, second=s, microsecond=micros
    ) - timedelta(days=x)
    tz_aware_earlier_time = tz.localize(tz_naive_earlier_time).astimezone(original_tz)
    return tz_aware_earlier_time
