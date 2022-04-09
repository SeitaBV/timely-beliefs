from datetime import datetime

import pytz

from timely_beliefs.examples import get_example_df, get_temperature_df


def test_ridgeline_example():
    df = get_example_df()
    df = df[df.index.get_level_values("source") == df.lineage.sources[0]]
    chart = df.plot_ridgeline_fixed_viewpoint(
        datetime(2000, 1, 3, 9, 0, tzinfo=pytz.utc),
        df,
        future_only=True,
        distribution="normal",
    )
    # chart.serve()
    assert chart
    chart = df.plot_ridgeline_belief_history(
        datetime(2000, 1, 3, 9, 0, tzinfo=pytz.utc),
        df,
        past_only=True,
        distribution="normal",
    )
    # chart.serve()
    assert chart


def test_ridgeline_temperature():
    df = get_temperature_df()
    df = df[
        df.index.get_level_values("event_start")
        <= datetime(2015, 3, 3, 14, tzinfo=pytz.utc)
    ]
    chart = df.plot_ridgeline_fixed_viewpoint(
        datetime(2015, 3, 1, 13, 0, tzinfo=pytz.utc),
        df,
        future_only=True,
        distribution="normal",
        event_value_window=(-1, 16),
    )
    # chart.serve()
    assert chart
    chart = df.plot_ridgeline_belief_history(
        datetime(2015, 3, 1, 13, 0, tzinfo=pytz.utc),
        df,
        past_only=True,
        distribution="normal",
        event_value_window=(-1, 16),
    )
    # chart.serve()
    assert chart
