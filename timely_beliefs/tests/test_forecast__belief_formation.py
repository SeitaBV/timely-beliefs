from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz
from sktime.forecasting.naive import NaiveForecaster

from timely_beliefs import BeliefsDataFrame, BeliefSource, Sensor
from timely_beliefs.beliefs import utils as belief_utils
from timely_beliefs.examples import get_example_df


@pytest.mark.parametrize(
    "forecaster, forecast",
    [
        # by default, most recent belief by Source A about most recent event
        (None, 400),
        # mean value of most recent beliefs by Source A about previous events
        (NaiveForecaster(strategy="mean"), 250),
    ],
)
def test_form_single_belief(forecaster, forecast):
    df = get_example_df()
    print(df)
    df_in = df[df.index.get_level_values("source") == df.lineage.sources[0]]
    df_in = df_in.make_deterministic()
    df_in = belief_utils.select_most_recent_belief(df_in)
    df_in_copy = df_in.copy()
    source = BeliefSource("Source C")
    df_out = df_in.form_beliefs(
        belief_time=datetime(2000, 1, 1, 2, tzinfo=pytz.utc),
        source=source,
        event_start=datetime(2000, 1, 3, 13, tzinfo=pytz.utc),
        forecaster=forecaster,
    )

    # Operation should not affect original DataFrame
    pd.testing.assert_frame_equal(df_in, df_in_copy)

    # Check expected forecast
    assert len(df_out) == 1
    assert df_out.values[0] == forecast

    # Check concatenation
    df_concat_check = pd.concat([df_in, df_out])
    df_concat = df_in.form_beliefs(
        belief_time=datetime(2000, 1, 1, 2, tzinfo=pytz.utc),
        source=source,
        event_start=datetime(2000, 1, 3, 13, tzinfo=pytz.utc),
        forecaster=forecaster,
        concatenate=True,
    )
    assert len(df_concat) == len(df_in) + 1
    assert df_concat.values[-1] == forecast
    pd.testing.assert_frame_equal(df_concat, df_concat_check)


def test_form_nan_belief_on_empty_frame():
    """Check whether forming beliefs on an empty frame leads to NaN values for the requested event time window."""
    sensor = Sensor(
        name="Availability",
        unit="%",
        event_resolution=timedelta(hours=1),
    )
    df_in = BeliefsDataFrame(sensor=sensor)
    source = BeliefSource("Source C")
    df_out = df_in.form_beliefs(
        belief_time=datetime(2000, 1, 1, 2, tzinfo=pytz.utc),
        source=source,
        event_time_window=(
            datetime(2000, 1, 3, tzinfo=pytz.utc),
            datetime(2000, 1, 4, tzinfo=pytz.utc),
        ),
    )
    assert len(df_out) == len(df_in) + 24
    assert all(df_out.isnull())
