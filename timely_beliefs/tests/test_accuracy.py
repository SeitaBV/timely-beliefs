from datetime import datetime, timedelta
from statistics import NormalDist

import numpy as np
import pytest
from pytz import utc

from timely_beliefs import BeliefsDataFrame
from timely_beliefs.beliefs.probabilistic_utils import (
    crps_ensemble,
    interpret_complete_cdf,
    partial_cdf,
)
from timely_beliefs.examples import get_example_df
from timely_beliefs.tests.utils import equal_lists


def test_setting_reference():
    """Set a column with reference values."""

    # Deterministic reference values
    example_df = get_example_df()
    df = example_df.set_reference_values(
        reference_source=example_df.lineage.sources[0],
        return_reference_type="mean",
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
        return_reference_type="full",
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


@pytest.mark.parametrize(
    "observation, values, probabilities, expected",
    [
        # A single member scores its absolute error.
        (3, [5], None, 2),
        # E|X - x| = 0.5 and E|X - X'| = 0.5, so the score is 0.5 - 0.25.
        (0, [0, 1], None, 0.25),
        # E|X - x| = 0.75 and E|X - X'| = 2 * 0.25 * 0.75, so the score is 0.75 - 0.1875.
        (0, [0, 1], [0.25, 0.75], 0.5625),
        # Probabilities are normalized, so only their ratio matters.
        (0, [0, 1], [1, 3], 0.5625),
    ],
)
def test_crps_ensemble(observation, values, probabilities, expected):
    """Check the CRPS of small ensembles against scores worked out by hand."""
    assert crps_ensemble(observation, values, probabilities) == pytest.approx(expected)


def test_interpret_complete_cdf_as_normal():
    """Fit a normal distribution through two points on its CDF: one standard deviation below the mean, and the mean."""
    (distribution,) = interpret_complete_cdf(
        [[NormalDist().cdf(-1), 0.5]], [[90, 100]], distribution="normal"
    )
    assert distribution.getMean()[0] == pytest.approx(100)
    assert distribution.getStandardDeviation()[0] == pytest.approx(10)


def test_interpret_complete_cdf_as_normal_needs_inner_probabilities():
    """A cumulative probability of 0 or 1 lies at an infinite distance from the mean, so it cannot pin down a normal distribution."""
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        interpret_complete_cdf([[0.5, 1]], [[100, 110]], distribution="normal")


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
