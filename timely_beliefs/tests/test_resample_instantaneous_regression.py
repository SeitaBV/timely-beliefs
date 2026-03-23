#!/usr/bin/env python3
"""
Test case for the fix of ZeroDivisionError when resampling with timedelta(0) and multiple sources/belief times.

This test reproduces the issue from:
https://github.com/SeitaBV/timely-beliefs/issues/228

The bug occurred when:
1. BeliefsDataFrame had multiple sources
2. BeliefsDataFrame had multiple belief times
3. Attempting to resample to timedelta(0) (instantaneous events)

The error was: ZeroDivisionError: integer modulo by zero in join_beliefs()
"""
from datetime import datetime, timedelta

import pandas as pd
import pytz

from timely_beliefs.beliefs.classes import BeliefsDataFrame
from timely_beliefs.sensors.classes import Sensor


def test_resample_to_instantaneous_with_multiple_sources_and_belief_times():
    """Test that resampling to timedelta(0) works with multiple sources and belief times.

    Regression test for: ZeroDivisionError when resampling to instantaneous with multiple sources
    """
    # Create a sensor
    timezone_obj = pytz.timezone("Europe/Amsterdam")
    sensor = Sensor(
        name="test sensor",
        unit="MWh",
        event_resolution=timedelta(minutes=15),
        timezone=timezone_obj,
    )

    # Create data with TWO sources and TWO belief times
    data = {
        "event_start": [
            datetime(2026, 3, 24, 12, 15, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 12, 30, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 12, 45, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 13, 0, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 13, 15, tzinfo=timezone_obj),
            # Second source with different belief time
            datetime(2026, 3, 24, 12, 15, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 12, 30, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 12, 45, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 13, 0, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 13, 15, tzinfo=timezone_obj),
        ],
        "belief_time": [
            datetime(2026, 3, 23, 13, 15, 0, 320382, tzinfo=timezone_obj),
            datetime(2026, 3, 23, 13, 15, 0, 320382, tzinfo=timezone_obj),
            datetime(2026, 3, 23, 13, 15, 0, 320382, tzinfo=timezone_obj),
            datetime(2026, 3, 23, 13, 15, 0, 320382, tzinfo=timezone_obj),
            datetime(2026, 3, 23, 13, 15, 0, 320382, tzinfo=timezone_obj),
            # Older belief time for second source
            datetime(2025, 3, 31, 9, 27, 30, 783382, tzinfo=timezone_obj),
            datetime(2025, 3, 31, 9, 27, 30, 783382, tzinfo=timezone_obj),
            datetime(2025, 3, 31, 9, 27, 30, 783382, tzinfo=timezone_obj),
            datetime(2025, 3, 31, 9, 27, 30, 783382, tzinfo=timezone_obj),
            datetime(2025, 3, 31, 9, 27, 30, 783382, tzinfo=timezone_obj),
        ],
        "source": [
            "source1",
            "source1",
            "source1",
            "source1",
            "source1",
            "source2",
            "source2",
            "source2",
            "source2",
            "source2",
        ],
        "cumulative_probability": [0.5] * 10,
        "event_value": [10.0] * 5 + [20.0] * 5,
    }

    df = pd.DataFrame(data)
    bdf = BeliefsDataFrame(df, sensor=sensor)

    # Verify the setup
    assert bdf.lineage.number_of_sources == 2
    assert bdf.lineage.number_of_belief_times == 2
    assert bdf.event_resolution == timedelta(minutes=15)

    # This should not raise ZeroDivisionError
    result = bdf.resample_events(
        timedelta(0), boundary_policy="first", keep_only_most_recent_belief=True
    )

    # Verify the result
    assert result.event_resolution == timedelta(0)
    assert len(result) > 0
    # With keep_only_most_recent_belief=True, should keep only the most recent belief time
    assert len(result.index.get_level_values("belief_time").unique()) == 1


def test_resample_to_instantaneous_without_keep_most_recent():
    """Test resampling to timedelta(0) without keep_only_most_recent_belief."""
    timezone_obj = pytz.timezone("Europe/Amsterdam")
    sensor = Sensor(
        name="test sensor",
        unit="MWh",
        event_resolution=timedelta(minutes=15),
        timezone=timezone_obj,
    )

    data = {
        "event_start": [
            datetime(2026, 3, 24, 12, 15, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 12, 30, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 12, 15, tzinfo=timezone_obj),
            datetime(2026, 3, 24, 12, 30, tzinfo=timezone_obj),
        ],
        "belief_time": [
            datetime(2026, 3, 23, 13, 15, 0, tzinfo=timezone_obj),
            datetime(2026, 3, 23, 13, 15, 0, tzinfo=timezone_obj),
            datetime(2025, 3, 31, 9, 27, 30, tzinfo=timezone_obj),
            datetime(2025, 3, 31, 9, 27, 30, tzinfo=timezone_obj),
        ],
        "source": ["source1", "source1", "source2", "source2"],
        "cumulative_probability": [0.5] * 4,
        "event_value": [10.0, 10.0, 20.0, 20.0],
    }

    df = pd.DataFrame(data)
    bdf = BeliefsDataFrame(df, sensor=sensor)

    # Should work without keep_only_most_recent_belief
    result = bdf.resample_events(timedelta(0), boundary_policy="first")

    assert result.event_resolution == timedelta(0)
    assert len(result) > 0


if __name__ == "__main__":
    test_resample_to_instantaneous_with_multiple_sources_and_belief_times()
    test_resample_to_instantaneous_without_keep_most_recent()
    print("All tests passed!")
