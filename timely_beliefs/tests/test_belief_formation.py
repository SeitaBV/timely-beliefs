from datetime import datetime

import pandas as pd
import pytest
import pytz
from sktime.forecasting.naive import NaiveForecaster

from timely_beliefs import BeliefSource
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
    df_in = df_in.rolling_viewpoint()
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
