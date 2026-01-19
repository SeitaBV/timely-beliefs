from datetime import datetime, timedelta

import pandas as pd
import pytz

from timely_beliefs import BeliefsDataFrame, TimedBelief


def test_most_common_event_frequency_with_gaps(time_slot_sensor, test_source_a):
    """
    Gaps are allowed as long as they are integer multiples of the base resolution.
    event_frequency should be None, but most_common_event_frequency should still
    infer the base resolution.
    """
    start = pytz.timezone("utc").localize(datetime(2000, 1, 3, 9))

    # Construct beliefs with a 1-hour gap (15-min base resolution)
    beliefs = [
        TimedBelief(
            sensor=time_slot_sensor,
            source=test_source_a,
            event_start=start + timedelta(minutes=m),
            belief_time=start,
            event_value=i,
        )
        for i, m in enumerate([0, 15, 30, 90, 105])
    ]

    bdf = BeliefsDataFrame(sensor=time_slot_sensor, beliefs=beliefs)

    # Gappy data means event_frequency cannot be determined
    assert bdf.event_frequency is None or pd.isna(bdf.event_frequency)

    # But base resolution is still clear
    assert bdf.most_common_event_frequency == timedelta(minutes=15)
