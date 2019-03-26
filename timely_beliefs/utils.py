from datetime import datetime

from pytz import utc
import pandas as pd


def enforce_utc(dt: datetime) -> pd.Timestamp:
    if dt.tzinfo is None:
        raise Exception(
            "The timely-beliefs package does not work with timezone-naive datetimes. Please localize your datetime."
        )
    return pd.Timestamp(dt.astimezone(utc))
