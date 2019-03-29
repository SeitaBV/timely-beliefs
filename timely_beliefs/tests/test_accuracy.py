from datetime import datetime, timedelta

import numpy as np
from pytz import utc

from timely_beliefs.tests import example_df


def test_mae():
    df = example_df
    mae = df.rolling_horizon_accuracy(timedelta(days=2, hours=9), source_id_anchor=1)
    assert (mae.values == np.array([0, (100+200+300+400)/4])).all()
    mae = df.rolling_horizon_accuracy(timedelta(days=2, hours=10), source_id_anchor=1)
    assert (mae.values == np.array([0, (200 + 300 + 400) / 3])).all()  # No forecast yet for the first event
    mae = df.rolling_horizon_accuracy(timedelta(days=2, hours=10), source_id_anchor=2)
    assert (mae.values == np.array([(200 + 300 + 400) / 3, 0])).all()  # Same error, but by the other source
    mae = df.fixed_horizon_accuracy(datetime(2000, 1, 2, tzinfo=utc), source_id_anchor=1)
    assert (mae.values == np.array([0, (100 + 200 + 300 + 400) / 4])).all()
    mae = df.fixed_horizon_accuracy(datetime(2000, 1, 1, tzinfo=utc), source_id_anchor=1)
    assert (mae.values == np.array([0, (100 + 200 + 300 + 400) / 4])).all()
    mae = df.accuracy(timedelta(days=2, hours=9))
    assert (mae.values == np.zeros(2)).all()
    mae = df.accuracy(datetime(2000, 1, 2, tzinfo=utc))
    assert (mae.values == np.zeros(2)).all()
