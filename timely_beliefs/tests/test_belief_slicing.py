from datetime import datetime

import pytz

from timely_beliefs.examples import example_df
from timely_beliefs.beliefs import utils


def test_belief_most_recent():
    """First make sure there is at least some diversity of cumulative probabilities
    amongst probabilistic beliefs by the same source about the same event.
    Then check whether only the latest belief time survived after selecting the most recent beliefs,
    also checking specifically that the other belief time did not survive.
    """
    df = example_df
    df = df.reset_index()
    df["cumulative_probability"].iloc[0] = 0.2
    df = df.set_index(["event_start", "belief_time", "source", "cumulative_probability"])
    df = utils.select_most_recent_belief(df)
    assert datetime(2000, 1, 1, 1, tzinfo=pytz.utc) in df.lineage.belief_times
    assert datetime(2000, 1, 1, 0, tzinfo=pytz.utc) not in df.lineage.belief_times
