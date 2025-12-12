import pandas as pd
import pytest

from timely_beliefs.examples import get_example_df
from timely_beliefs.utils import (
    enforce_tz,
    parse_datetime_like,
    remove_class_init_kwargs,
)


def test_remove_used_kwargs():
    class MyCls:
        def __init__(self, a, b):
            pass

    kwargs = dict(a=1, b=2, c=3, d=4, self=5)
    remaining_kwargs = remove_class_init_kwargs(MyCls, kwargs)
    assert remaining_kwargs == dict(c=3, d=4, self=5)


def test_parse_datetime_like():
    dt = pd.Series(["2000-01-03 09:00:00+01:00", "2000-01-03 10:00:00+01:00"])
    parse_datetime_like(dt)
    with pytest.raises(TypeError) as e_info:
        dt = pd.Series(["2000-01-03 09:00:00", "2000-01-03 10:00:00"])
        parse_datetime_like(dt)
    assert (
        str(e_info.value)
        == "The timely-beliefs package does not work with timezone-naive datetimes. Please localize your Series starting with 2000-01-03 09:00:00."
    )


def test_enforce_tz():
    df = get_example_df()
    enforce_tz(df.index.get_level_values("event_start"))
    with pytest.raises(TypeError) as e_info:
        enforce_tz(df.index.get_level_values("event_start").tz_localize(None))
    assert (
        str(e_info.value)
        == "The timely-beliefs package does not work with timezone-naive datetimes. Please localize your DatetimeIndex starting with 2000-01-03 09:00:00."
    )
