"""openturns-backed probabilistic machinery.

Import this module lazily, inside the function that needs it, so importing timely_beliefs doesn't
pay openturns's ~66 MB import cost for consumers who never touch probabilistic beliefs.

joint_cdf_with_openturns_copula is the general engine: pass a custom copula or openturns-distribution
marginals. For the common independent-copula, plain-array case, use independent_joint_cdf instead.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

try:
    import openturns as ot
except ImportError as e:
    raise ImportError(
        "This module requires the 'openturns' package to work with probabilistic beliefs. "
        "Install it with: pip install timely-beliefs[probabilistic]"
    ) from e

from timely_beliefs.beliefs.probabilistic_math import cp_to_p, erfinv


def interpret_complete_cdf_as_distribution(
    cdfs_p: list[list | np.ndarray],
    cdfs_v: list[list | np.ndarray],
    distribution: str,
) -> list[ot.DistributionImplementation]:
    """Interpret the given points on the cumulative distribution function as an openturns distribution.
    - discrete: all residual probability is attributed to the highest given value
    - normal or gaussian: derived from the first two points only
    - uniform: interpolates linearly between points, with residual probability attributed to the min and max values
    """
    cdfs = []
    if distribution == "discrete":
        for cdf_p, cdf_v in zip(cdfs_p, cdfs_v):
            cdf_p[-1] = 1  # Last value is the highest
            cdfs.append(ot.UserDefined([[v] for v in cdf_v], cp_to_p(cdf_p)))
    elif distribution in ["normal", "gaussian"]:
        for cdf_p, cdf_v in zip(cdfs_p, cdfs_v):
            if len(cdf_v) > 1:
                x1 = cdf_v[0]
                x2 = cdf_v[1]
                y1 = cdf_p[0]
                y2 = cdf_p[1]
                mu = (x1 * erfinv(1 - 2 * y2) - x2 * erfinv(1 - 2 * y1)) / (
                    erfinv(1 - 2 * y2) - erfinv(1 - 2 * y1)
                )
                sigma = (2**0.5 * x1 - 2**0.5 * x2) / (
                    2 * erfinv(1 - 2 * y2) - 2 * erfinv(1 - 2 * y1)
                )
                cdfs.append(ot.Normal(mu, sigma))
            else:
                cdfs.append(ot.UserDefined([[v] for v in cdf_v], cdf_p))
    elif distribution == "uniform":
        for cdf_p, cdf_v in zip(cdfs_p, cdfs_v):
            if len(cdf_v) == 1:
                cdfs.append(ot.UserDefined([cdf_v]))
            elif len(cdf_v) > 1:
                coll = (
                    [ot.UserDefined([[cdf_v[0]]])]
                    + [
                        ot.Uniform(float(cdf_v[i]), float(cdf_v[i + 1]))
                        for i in range(len(cdf_v) - 1)
                    ]
                    + [ot.UserDefined([[cdf_v[-1]]])]
                )
                weights = np.append(cp_to_p(cdf_p), 1 - cdf_p[-1])
                cdfs.append(ot.Mixture(coll, weights))
    else:
        return NotImplementedError
    return cdfs


def joint_cdf_with_openturns_copula(  # noqa: C901
    marginal_cdfs_p: (
        list[list[float] | np.ndarray | ot.DistributionImplementation] | np.ndarray
    ),
    marginal_cdfs_v: list[list[float] | np.ndarray] | np.ndarray = None,
    a: float = 0,
    b: float = 1,
    copula: ot.DistributionImplementation = None,
    agg_function=None,
    simplify: bool = True,
    n_draws: int = 100,
    empirical: bool = False,
) -> tuple[np.array, np.array]:
    """Calculate univariate joint CDF given a list of multivariate marginal CDFs and a copula,
    returning both the cumulative probabilities and the aggregated outcome of the random variables.

    :param marginal_cdfs_p: Each marginal CDF is a list (or 2darray) with cumulative probabilities up to cp=1.
    If it doesn't reach cp=1 and there are few cdfs (low dimension), the remaining probability is attributed
    to some higher (unknown) outcome, and the empirical method can't be used.
    :param marginal_cdfs_v: Values of possible outcomes for each random variable, i.e. the bins of the marginal CDFs.
    If just one set of bins is given, we assume the CDFs share it. If none is given, we assume equal-sized bins between a and b.

    "All bins are equal, but some bins are more equal than others." (because they have a higher probability)

    :param a: The lowest outcome (0 by default, ignored if CDF values are given explicitly)
    :param b: The highest outcome (1 by default, ignored if CDF values are given explicitly)
    :param copula: The default copula is the independence copula (i.e. we assume independent random variables).
    :param agg_function: The default aggregation function is to take the sum of the outcomes.
    :param simplify: Simplify the resulting cdf by removing outcomes with zero probability (True by default)
    :param n_draws: Number of draws (sample size) to compute the empirical CDF when aggregating >3 random variables.
    :param empirical: Compute the empirical CDF regardless of number of random variables (default is False)
    """

    dim = len(marginal_cdfs_p)
    n_outcomes = 99  # Todo: refactor to avoid having to set this above our threshold for computing exact probabilities

    # Set up marginal distributions
    empirical_method_possible = True
    if isinstance(marginal_cdfs_p[0], ot.DistributionImplementation):
        marginals = marginal_cdfs_p
        shared_bins = False
        empirical = True
    else:
        # Set up marginal cdf values
        n_outcomes = len(marginal_cdfs_p[0])
        shared_bins = True
        if marginal_cdfs_v is None:
            values = np.linspace(a, b, n_outcomes)
        elif isinstance(marginal_cdfs_v[0], (list, np.ndarray)):
            shared_bins = False
            values = marginal_cdfs_v
        else:
            values = marginal_cdfs_v

        marginals = []
        for i in range(dim):
            marginal_cdf = marginal_cdfs_p[i]
            if shared_bins is True:
                values_for_cdf = values
            else:
                values_for_cdf = marginal_cdfs_v[i]
            if not math.isclose(marginal_cdf[-1], 1, rel_tol=1e-7):
                empirical_method_possible = False
                # We can assume some higher outcome exists with cp=1
                values_for_cdf = np.append(
                    values_for_cdf, values_for_cdf[-1] + 1
                )  # Add a higher outcome (+1 suffices)
                marginal_pdf = np.clip(
                    np.concatenate(
                        (
                            [marginal_cdf[0]],
                            np.diff(marginal_cdf),
                            [1.0 - marginal_cdf[-1]],
                        )
                    ),
                    0,
                    1,
                )
                marginals.append(
                    ot.UserDefined([[v] for v in values_for_cdf], marginal_pdf)
                )
            else:
                marginal_pdf = np.clip(cp_to_p(marginal_cdf), 0, 1)
                marginals.append(
                    ot.UserDefined([[v] for v in values_for_cdf], marginal_pdf)
                )

    # If not specified, pick the independent copula as a default (i.e. assume independent random variables)
    if copula is None:
        copula = ot.IndependentCopula(dim)
    elif not copula.isCopula():
        raise ValueError(f"Copula {copula} doesn't seem to be a valid copula")

    # If not specified, pick the sum function as a default for joining values
    if agg_function is None:
        agg_function = np.sum

    # Evaluate exact probabilities only for small bivariate and tri-variate joint distributions
    if dim <= 3 and n_outcomes <= 10 and empirical is False:

        # Determine joint distribution (too slow for high dimensions)
        d = ot.JointDistribution(marginals, copula)

        # Compute acceptable margin to prevent floating point errors (we'll evaluate a little on the right side of each marginal point)
        if shared_bins is True:
            smallest_marginal_point_distance = (
                np.diff(values).min() if n_outcomes > 1 else 1
            )
        elif dim > 1:
            smallest_marginal_point_distance = (
                np.diff(values, axis=1).min() if n_outcomes > 1 else 1
            )
        else:
            smallest_marginal_point_distance = (
                1  # With just 1 point, an arbitrary positive distance suffices (e.g. 1)
            )
        margin = smallest_marginal_point_distance / 2

        # Construct an n-dimensional matrix with all possible points (i.e. combinations of outcomes of our random variables)
        if shared_bins is True:
            marginal_points = list(product(values, repeat=dim))
            shape = (n_outcomes,) * dim

            # Marginal points for the cdf evaluation are slightly higher to ensure we are on the right side of the discrete jump in cumulative probability
            marginal_points_for_cdf_evaluation = list(
                product([v + margin for v in values], repeat=dim)
            )
        else:
            marginal_points = list(product(*marginal_cdfs_v))
            shape = [len(m) for m in marginal_cdfs_v]

            # Marginal points for the cdf evaluation
            marginal_points_for_cdf_evaluation = list(
                product(*[v + margin for v in marginal_cdfs_v])
            )

        # Evaluate exact probabilities at each point (too slow for high dimensions)
        joint_multivariate_cdf = np.reshape(
            d.computeCDF(marginal_points_for_cdf_evaluation), shape
        )
        joint_multivariate_pdf = _joint_cdf_to_pdf(joint_multivariate_cdf)

        # Sort the probabilities ascending, keeping track of the corresponding values
        p, v = zip(
            *sorted(
                zip(joint_multivariate_pdf.flatten(), agg_function(marginal_points, 1))
            )
        )

        # Calculate total probability of each unique value (by adding probability of cases that yield the same value)
        cdf_v = np.unique(v)
        pdf_p = np.array([sum(np.array(p)[np.where(v == i)[0]]) for i in cdf_v])
    elif (
        empirical_method_possible is True
    ):  # Otherwise, compute the empirical cdf from a sample generated directly from the copula
        uniform_points = np.array(
            copula.getSample(n_draws)
        )  # Much faster than sampling from the joint cdf
        aggregated_points = np.zeros(n_draws)
        for i, point in enumerate(uniform_points):
            aggregated_points[i] = agg_function(
                list(
                    marginal_cdf.computeQuantile(marginal_cdf_p)[0]
                    for marginal_cdf_p, marginal_cdf in zip(point, marginals)
                )
            )
        empirical_cdf = ot.UserDefined([[v] for v in aggregated_points])
        pdf_p = np.array(empirical_cdf.getP())
        cdf_v = np.array(empirical_cdf.getX()).flatten()
    else:
        raise ValueError(
            "Empirical method not possible given incomplete marginal CDF. Make sure all CDFs go up to 1."
        )

    # Simplify resulting pdf
    if simplify is True:
        cdf_v = cdf_v[np.nonzero(pdf_p)]
        pdf_p = pdf_p[np.nonzero(pdf_p)]

    # Return the univariate joint cumulative probability function
    cdf_p = pdf_p.cumsum()

    return cdf_p, cdf_v


def _joint_cdf_to_pdf(cdf: np.ndarray) -> np.ndarray:
    """Recursive function to determine the joint multivariate pdf from a given joint multivariate cdf."""

    if len(cdf.shape) > 1:
        pdf = cdf.copy()
        for i, cdf_i in enumerate(cdf):
            if i != 0:
                pdf[i] = _joint_cdf_to_pdf(cdf_i) - _joint_cdf_to_pdf(cdf[i - 1])
            else:
                pdf[i] = _joint_cdf_to_pdf(cdf_i)
        return pdf
    else:
        return cp_to_p(cdf)
