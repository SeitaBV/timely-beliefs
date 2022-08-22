import pytest

from datetime import datetime
import pytz

import pandas as pd
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
    print(df_in)
    df_out = df_in.form_beliefs(
        belief_time=datetime(2000, 1, 1, 2, tzinfo=pytz.utc),
        source=BeliefSource("Source C"),
        event_start=datetime(2000, 1, 3, 13, tzinfo=pytz.utc),
        forecaster=forecaster,
    )
    assert df_out.values[0] == forecast
    df = pd.concat([df, df_out])
    print(df)
