import pytest
from pytest import approx
from typing import Tuple

import numpy as np
import openturns as ot

from timely_beliefs.beliefs.probabilistic_utils import (
    multivariate_marginal_to_univariate_joint_cdf,
    equalize_bins,
)
from timely_beliefs.tests.utils import equal_lists


@pytest.fixture(scope="module")
def multivariate_test_cdfs() -> Tuple[np.ndarray, np.ndarray]:
    """Convenient CDF input for stress tests:
    - How many random variables can we handle? Change dim
    - What depth of probabilistic accuracy can we handle? Change n_outcomes.
    """
    # Todo: speed up by serialising the work
    # Todo: test unequal number of outcomes for the variables

    dim = 300
    n_outcomes = 100
    min_v = 10  # Lowest possible outcome
    max_v = 100  # Highest possible outcome
    p_tail = 0.00  # Residual probability of outcomes higher than max_v

    marginal_cdf_v = (max_v - min_v) * np.sort(
        np.random.random_sample((dim, n_outcomes))
    ) + min_v
    marginal_pdf_p = np.random.random_sample((dim, n_outcomes))
    marginal_pdf_p = (
        marginal_pdf_p / marginal_pdf_p.sum(axis=1, keepdims=True) - p_tail / n_outcomes
    )  # Normalize
    marginal_cdf_p = np.cumsum(marginal_pdf_p, axis=1)

    return marginal_cdf_p, marginal_cdf_v


def test_bivariate_aggregation():
    """Check defaults for bivariate aggregation."""

    # For independent variables, sum of outcomes, possible outcomes are P(v=0)=0 and P(v=1)=1
    cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf([[0, 1], [0, 1]])
    assert all(np.diff(cdf_v) >= 0) and all(
        np.diff(cdf_p) >= 0
    )  # Check for non-decreasing cdf
    assert equal_lists(
        cdf_v, [2]
    )  # Check aggregated outcomes (no chance of sum = 0 or 1)
    assert equal_lists(cdf_p, [1])  # Check cumulative probabilities
    # And for possible outcomes P(v=0)=0.5 and P(v=1)=0.5
    cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf([[0.5, 1], [0.5, 1]])
    assert equal_lists(
        cdf_v, [0, 1, 2]
    )  # Check aggregated outcomes (25% chance of sum = 0 or 2, 50% of sum = 1)
    assert equal_lists(cdf_p, [0.25, 0.75, 1])  # Check cumulative probabilities


def test_multivariate_aggregation():
    """Check different aggregation functions for multivariate aggregation, as well as a copula."""

    # For sum
    marginal_pdfs = [
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 4, 1 / 4, 2 / 4],
        [1 / 6, 1 / 6, 2 / 3],
    ]
    marginal_cdfs = np.cumsum(marginal_pdfs, axis=1)
    cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdfs, a=10, b=100
    )
    assert all(np.diff(cdf_v) >= 0) and all(
        np.diff(cdf_p) >= 0
    )  # Check for non-decreasing cdf
    assert cdf_v[0] == 30 and cdf_v[-1] == 300  # Check range of aggregated outcomes
    assert (
        cdf_p[0] == 1 / 3 * 1 / 4 * 1 / 6 and cdf_p[-1] == 1
    )  # Check range of cumulative probabilities

    # For mean
    cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdfs, agg_function=np.mean, a=10, b=100
    )
    assert all(np.diff(cdf_v) >= 0) and all(
        np.diff(cdf_p) >= 0
    )  # Check for non-decreasing cdf
    assert cdf_v[0] == 10 and cdf_v[-1] == 100  # Check range of aggregated outcomes
    assert (
        cdf_p[0] == 1 / 3 * 1 / 4 * 1 / 6 and cdf_p[-1] == 1
    )  # Check range of cumulative probabilities

    # For a normal copula with a correlation matrix with positive correlation between 1st and 2nd variable
    R = ot.CorrelationMatrix(len(marginal_cdfs))
    R[0, 1] = 0.25
    cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdfs, copula=ot.NormalCopula(R), a=10, b=100
    )
    assert all(np.diff(cdf_v) >= 0) and all(
        np.diff(cdf_p) >= 0
    )  # Check for non-decreasing cdf
    assert cdf_v[0] == 30 and cdf_v[-1] == 300  # Check range of aggregated outcomes
    assert (
        cdf_p[0] > 1 / 3 * 1 / 4 * 1 / 6 and cdf_p[-1] == 1
    )  # Check range of cumulative probabilities

    # For a normal copula with a correlation matrix with negative correlation between 1st and 2nd variable
    R = ot.CorrelationMatrix(len(marginal_cdfs))
    R[0, 1] = -0.25
    cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdfs, copula=ot.NormalCopula(R), a=10, b=100
    )
    assert all(np.diff(cdf_v) >= 0) and all(
        np.diff(cdf_p) >= 0
    )  # Check for non-decreasing cdf
    assert cdf_v[0] == 30 and cdf_v[-1] == 300  # Check range of aggregated outcomes
    assert (
        cdf_p[0] < 1 / 3 * 1 / 4 * 1 / 6 and cdf_p[-1] == 1
    )  # Check range of cumulative probabilities


def test_bivariate_aggregation_with_unmatched_bins():
    """Check bivariate aggregation where the outcomes of the first and second variables are completely different."""

    marginal_cdfs_v = [[5, 6.5, 7], [1.2, 2.02, 3]]
    marginal_cdfs_p = [[1 / 3, 2 / 3, 1], [1 / 4, 2 / 3, 3 / 4]]

    cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdfs_p, marginal_cdfs_v=marginal_cdfs_v
    )
    assert all(np.diff(cdf_v) >= 0) and all(
        np.diff(cdf_p) >= 0
    )  # Check for non-decreasing cdf
    assert (
        cdf_v[0] == 5 + 1.2 and cdf_v[-1] == 7 + 3
    )  # Check range of aggregated outcomes
    assert (
        cdf_p[0] == 1 / 3 * 1 / 4 and cdf_p[-1] == 3 / 4
    )  # Check range of cumulative probabilities

    # Show equalising bins is irrelevant
    marginal_cdfs_v, marginal_cdfs_p = equalize_bins(marginal_cdfs_v, marginal_cdfs_p)
    cdf_p_2, cdf_v_2 = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdfs_p, marginal_cdfs_v=marginal_cdfs_v
    )
    assert equal_lists(cdf_p, cdf_p_2)
    assert equal_lists(cdf_v, cdf_v_2)


def test_marginal_distributions_with_residual_probability():
    """Aggregate three time slots with CDFs that are not fully specified.
    That means each has a residual probability that their outcome is higher than the highest value given.
    """

    # Make sure incomplete cdf functions can still be transformed (a higher outcome with cp=1 can be assumed to exist)
    marginal_cdfs = [
        [1 / 3, 2 / 4, 3 / 4],
        [1 / 4, 4 / 6, 5 / 6],
        [1 / 6, 5 / 9, 8 / 9],
    ]
    cdf_p, _ = multivariate_marginal_to_univariate_joint_cdf(marginal_cdfs)
    assert all(np.diff(cdf_p) >= 0)  # Check for non-decreasing cdf
    assert (
        cdf_p[-1] < 1
    )  # Check that the assumed outcome with cp=1 is not actually returned
    a = (
        cdf_p[-1] - cdf_p[-2]
    )  # The probability of the highest outcome for each of the three variables
    b = 1 / 4 * 1 / 6 * 3 / 9  # The probability for independent random variables
    assert a == approx(b)  # Check the expected outcome
    # Make a correlation matrix with negative correlation between the first and second variable
    R = ot.CorrelationMatrix(len(marginal_cdfs))
    R[0, 1] = -0.25
    cdf_p, _ = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdfs, copula=ot.NormalCopula(R)
    )
    assert all(np.diff(cdf_p) >= 0)  # Check for non-decreasing cdf
    a = (
        cdf_p[-1] - cdf_p[-2]
    )  # The probability of the highest outcome for each of the three variables
    assert (
        a < b
    )  # Check the expected outcome is now lower (if x1 is high, than x2 is less likely to be high)
    # Make a correlation matrix with positive correlation between the first and second variable
    R = ot.CorrelationMatrix(len(marginal_cdfs))
    R[0, 1] = 0.25
    cdf_p, _ = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdfs, copula=ot.NormalCopula(R)
    )
    assert all(np.diff(cdf_p) >= 0)  # Check for non-decreasing cdf
    a = (
        cdf_p[-1] - cdf_p[-2]
    )  # The probability of the highest outcome for each of the three variables
    assert (
        a > b
    )  # Check the expected outcome is now higher (if x1 is high, than x2 is likely to be high, too)


def test_multivariate_aggregation_with_unmatched_bins(multivariate_test_cdfs):
    """Check multivariate aggregation where the outcomes of each variable are completely different."""

    marginal_cdf_p, marginal_cdf_v = multivariate_test_cdfs
    dim = len(marginal_cdf_p)

    cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdf_p, marginal_cdfs_v=marginal_cdf_v
    )

    assert all(np.diff(cdf_v) >= 0) and all(
        np.diff(cdf_p) >= 0
    )  # Check for non-decreasing cdf
    assert (
        cdf_v[0] >= 10 * dim and cdf_v[-1] <= 100 * dim
    )  # Check range of aggregated outcomes
    assert cdf_p[0] >= 0 and (
        cdf_p[-1] < 1 or cdf_p[-1] == approx(1)
    )  # Check range of cumulative probabilities


def test_multivariate_aggregation_with_unmatched_bins_and_dependence(
    multivariate_test_cdfs
):
    """Check multivariate aggregation where the outcomes of each variable are completely different,
    and the variables are correlated."""

    marginal_cdf_p, marginal_cdf_v = multivariate_test_cdfs
    dim = len(marginal_cdf_p)

    # Make a correlation matrix with positive correlation between each pair of adjacent variables (needs at least 2D)
    R = ot.CorrelationMatrix(dim)
    for d in range(1, dim):
        R[d - 1, d] = 0.25
    cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdf_p, marginal_cdfs_v=marginal_cdf_v, copula=ot.NormalCopula(R)
    )
    assert all(np.diff(cdf_v) >= 0) and all(
        np.diff(cdf_p) >= 0
    )  # Check for non-decreasing cdf
    assert (
        cdf_v[0] >= 10 * dim and cdf_v[-1] <= 100 * dim
    )  # Check range of aggregated outcomes
    assert cdf_p[0] >= 0 and (
        cdf_p[-1] < 1 or cdf_p[-1] == approx(1)
    )  # Check range of cumulative probabilities

    cdf_p_2, cdf_v_2 = multivariate_marginal_to_univariate_joint_cdf(
        marginal_cdf_p,
        marginal_cdfs_v=marginal_cdf_v,
        copula=ot.NormalCopula(R),
        n_draws=1000,
    )
    assert len(cdf_p_2) == 1000
