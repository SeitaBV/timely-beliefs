from datetime import datetime, timedelta

import numpy as np
from pytz import utc

from timely_beliefs import BeliefsDataFrame
from timely_beliefs.beliefs.probabilistic_utils import partial_cdf
from timely_beliefs.examples import get_example_df
from timely_beliefs.tests.utils import equal_lists


def test_setting_reference():
    """Set a column with reference values."""

    # Deterministic reference values
    example_df = get_example_df()
    df = example_df.set_reference_values(
        reference_source=example_df.lineage.sources[0],
        return_expected_value=True,
        # todo: deprecate the 'return_expected_value' argument in favor of 'return_reference_type' (announced v1.9.0)
        # return_reference_type = "mean",
    )
    assert isinstance(df, BeliefsDataFrame)  # Check if type is maintained
    assert df.sensor == example_df.sensor  # Check if sensor is kept
    assert equal_lists(
        df["reference_value"].values, [100] * 10 + [200] * 10 + [300] * 10 + [400] * 10
    )
    assert df.index.names == example_df.index.names

    # Probabilistic reference values
    df = example_df.set_reference_values(
        reference_source=example_df.lineage.sources[0],
        return_expected_value=False,
        # todo: deprecate the 'return_expected_value' argument in favor of 'return_reference_type' (announced v1.9.0)
        # return_reference_type="full",
    )
    assert equal_lists(df["reference_value"].head(6).values, [99, 100, 101] * 2)

    # Todo: Test event with missing reference belief


def test_mae():
    """For deterministic forecasts, our scoring rule (continuous ranked probability score) should equal the mean
    absolute error."""
    df = get_example_df().xs(0.5, level="cumulative_probability", drop_level=False)
    mae = df.rolling_viewpoint_accuracy(
        timedelta(days=2, hours=9), reference_source=df.lineage.sources[0]
    )["mae"]
    assert (mae.values == np.array([0, (100 + 200 + 300 + 400) / 4])).all()
    mae = df.rolling_viewpoint_accuracy(
        timedelta(days=2, hours=10), reference_source=df.lineage.sources[0]
    )["mae"]
    assert (
        mae.values == np.array([0, (200 + 300 + 400) / 3])
    ).all()  # No forecast yet for the first event
    mae = df.rolling_viewpoint_accuracy(
        timedelta(days=2, hours=10), reference_source=df.lineage.sources[1]
    )["mae"]
    assert (
        mae.values == np.array([(200 + 300 + 400) / 3, 0])
    ).all()  # Same error, but by the other source
    mae = df.fixed_viewpoint_accuracy(
        datetime(2000, 1, 2, tzinfo=utc), reference_source=df.lineage.sources[0]
    )["mae"]
    assert (mae.values == np.array([0, (100 + 200 + 300 + 400) / 4])).all()
    mae = df.fixed_viewpoint_accuracy(
        datetime(2000, 1, 1, tzinfo=utc), reference_source=df.lineage.sources[0]
    )["mae"]
    assert (mae.values == np.array([0, (100 + 200 + 300 + 400) / 4])).all()
    mae = df.accuracy(timedelta(days=2, hours=9))["mae"]
    assert (mae.values == np.zeros(2)).all()
    mae = df.accuracy(datetime(2000, 1, 2, tzinfo=utc))["mae"]
    assert (mae.values == np.zeros(2)).all()


def test_crps():
    """For probabilistic forecasts, our scoring rule (continuous ranked probability score) tells more than just whether
    you got the expected value right."""
    df = get_example_df()
    crps = df.rolling_viewpoint_accuracy(
        timedelta(days=2, hours=9), reference_source=df.lineage.sources[0]
    )["mae"]
    assert (
        crps[df.lineage.sources[0]] > 0
    )  # Expected value was exactly right, but source 1 still forecast probabilistically
    assert (
        crps[df.lineage.sources[0]] == (0.1587 * 9 + 0.5 * 9) / 4
    )  # Actually, only the first out of 4 forecasts was off
    assert (
        125 < crps[df.lineage.sources[1]] < 250
    )  # Expected value was wrong by 250, but source 2 still forecast about 50% right


def test_partial_cdf():

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1, 0.5, 1]), np.array([10, 20, 30]), (0, 0.2)
    )
    assert (cdf_p == np.array([0.5, 1])).all()
    assert (cdf_v == np.array([10, 20])).all()

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1587, 0.5, 1]), np.array([10, 20, 30]), (0, 0.1587)
    )
    assert (cdf_p == np.array([1])).all()
    assert (cdf_v == np.array([10])).all()

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1, 0.3, 0.5, 1]), np.array([10, 20, 30, 40]), (0.2, 0.7)
    )
    assert equal_lists(cdf_p, [0.2, 0.6, 1])
    assert equal_lists(cdf_v, [20, 30, 40])

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1, 0.3, 0.5, 1]), np.array([10, 20, 30, 40]), (0.2, 0.5)
    )
    assert equal_lists(cdf_p, [1 / 3, 1])
    assert equal_lists(cdf_v, [20, 30])

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1, 0.3, 0.5, 1]), np.array([10, 20, 30, 40]), (0.3, 0.7)
    )
    assert equal_lists(cdf_p, [0.5, 1])
    assert equal_lists(cdf_v, [30, 40])

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1, 0.3, 0.5, 1]), np.array([10, 20, 30, 40]), (0.5, 1)
    )
    assert equal_lists(cdf_p, [1])
    assert equal_lists(cdf_v, [40])

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1, 0.3, 0.5, 1]), np.array([10, 20, 30, 40]), (0.4, 1)
    )
    assert equal_lists(cdf_p, [1 / 6, 1])
    assert equal_lists(cdf_v, [30, 40])

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1, 0.3, 0.5, 1]), np.array([10, 20, 30, 40]), (0, 0.3)
    )
    assert equal_lists(cdf_p, [1 / 3, 1])
    assert equal_lists(cdf_v, [10, 20])

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1, 0.3, 0.5, 1]), np.array([10, 20, 30, 40]), (0, 0.2)
    )
    assert equal_lists(cdf_p, [0.5, 1])
    assert equal_lists(cdf_v, [10, 20])

    cdf_p, cdf_v = partial_cdf(
        np.array([0.1, 0.3, 0.5, 1]), np.array([10, 20, 30, 40]), (0, 0.1)
    )
    assert equal_lists(cdf_p, [1])
    assert equal_lists(cdf_v, [10])
