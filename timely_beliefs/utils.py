from datetime import datetime

from pytz import utc
from pandas import Timestamp


def enforce_utc(dt: datetime) -> Timestamp:
    if dt.tzinfo is None:
        raise Exception(
            "The timely-beliefs package does not work with timezone-naive datetimes. Please localize your datetime."
        )
    return Timestamp(dt.astimezone(utc))
