import pandas as pd

from timely_beliefs.beliefs.probabilistic_utils import get_median_belief
from timely_beliefs.beliefs.utils import propagate_beliefs
from timely_beliefs.examples import get_example_df


def test_propagate_multi_sourced_deterministic_beliefs():
    # Start with a deterministic example frame (4 events, 2 sources and 2 belief times)
    df = get_example_df().for_each_belief(get_median_belief)

    # Set the four later beliefs to an unknown value
    df[
        df.index.get_level_values("belief_time")
        == pd.Timestamp("2000-01-01 01:00:00+00:00")
    ] = None
    assert df["event_value"].isnull().sum() == 8

    # Propagate the four earlier beliefs
    df = propagate_beliefs(df)
    assert df["event_value"].isnull().sum() == 0

    # After propagating, the four later beliefs should be equal to the four earlier beliefs
    pd.testing.assert_frame_equal(
        df[
            df.index.get_level_values("belief_time")
            == pd.Timestamp("2000-01-01 00:00:00+00:00")
        ].droplevel("belief_time"),
        df[
            df.index.get_level_values("belief_time")
            == pd.Timestamp("2000-01-01 01:00:00+00:00")
        ].droplevel("belief_time"),
    )
