"""Joint CDF for the independent copula, in plain numpy (no openturns).

Under independence, the joint pdf is just the product of the marginal pdfs, so we don't need a
copula/joint-distribution object or an openturns dependency. Only handles the independent case;
for a custom copula or openturns-distribution marginals, use
probabilistic_backend.joint_cdf_with_openturns_copula.
"""

from __future__ import annotations

import functools
import math
from itertools import product
from typing import Callable

import numpy as np

from timely_beliefs.beliefs.probabilistic_math import cp_to_p


def independent_joint_cdf(
    marginal_cdfs_p: list[list[float] | np.ndarray] | np.ndarray,
    marginal_cdfs_v: list[list[float] | np.ndarray] | np.ndarray = None,
    a: float = 0,
    b: float = 1,
    agg_function: Callable[[np.ndarray], np.ndarray] = None,
    simplify: bool = True,
    n_draws: int = 100,
    empirical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent-copula counterpart of probabilistic_backend.joint_cdf_with_openturns_copula, in plain numpy."""
    if hasattr(marginal_cdfs_p[0], "computeQuantile"):
        raise TypeError(
            "independent_joint_cdf does not accept openturns distribution objects as marginals. "
            "Use probabilistic_backend.joint_cdf_with_openturns_copula instead."
        )

    dim = len(marginal_cdfs_p)
    n_outcomes = len(marginal_cdfs_p[0])

    shared_bins = True
    if marginal_cdfs_v is None:
        values = np.linspace(a, b, n_outcomes)
    elif isinstance(marginal_cdfs_v[0], (list, np.ndarray)):
        shared_bins = False
        values = marginal_cdfs_v
    else:
        values = marginal_cdfs_v

    if agg_function is None:
        agg_function = np.sum

    # Each marginal's known probability mass, as plain (values, pdf) pairs -- no padding to a
    # complete (sums-to-1) distribution needed; see module docstring.
    marginal_values = []
    marginal_pdfs = []
    empirical_method_possible = True
    for i in range(dim):
        marginal_cdf = marginal_cdfs_p[i]
        marginal_values.append(values if shared_bins else marginal_cdfs_v[i])
        marginal_pdfs.append(np.clip(cp_to_p(marginal_cdf), 0, 1))
        if not math.isclose(marginal_cdf[-1], 1, rel_tol=1e-7):
            empirical_method_possible = False

    if dim <= 3 and n_outcomes <= 10 and empirical is False:
        cdf_v, pdf_p = _exact(marginal_values, marginal_pdfs, shared_bins, agg_function)
    elif empirical_method_possible is True:
        cdf_v, pdf_p = _empirical(marginal_values, marginal_pdfs, n_draws, agg_function)
    else:
        raise ValueError(
            "Empirical method not possible given incomplete marginal CDF. Make sure all CDFs go up to 1."
        )

    if simplify is True:
        cdf_v, pdf_p = _drop_zero_probability_outcomes(cdf_v, pdf_p)

    cdf_p = pdf_p.cumsum()
    if empirical_method_possible and len(cdf_p):
        # All marginals were complete (each cdf[-1] isclose to 1), so the total probability mass is
        # exactly 1 mathematically; pin the last point there rather than leave it at whatever
        # floating-point remainder cumsum happened to accumulate.
        cdf_p[-1] = 1.0
    return cdf_p, cdf_v


def _exact(marginal_values, marginal_pdfs, shared_bins, agg_function):
    """Exact joint pdf as the outer product of independent marginal pdfs (dim <= 3)."""
    dim = len(marginal_values)

    # Joint pdf at each grid point is simply the product of marginal pdfs (independence). No detour
    # through a joint CDF (unlike the openturns implementation, which has to evaluate that way
    # because ot.JointDistribution only exposes computeCDF, not a direct joint pdf on a grid).
    joint_pdf = functools.reduce(np.multiply.outer, marginal_pdfs)

    if shared_bins:
        marginal_points = list(product(marginal_values[0], repeat=dim))
    else:
        marginal_points = list(product(*marginal_values))

    p = joint_pdf.flatten()
    v = agg_function(np.array(marginal_points), 1)

    # Calculate total probability of each unique value (by adding probability of cases that yield
    # the same value), same as probabilistic_backend.
    cdf_v = np.unique(v)
    pdf_p = np.array([p[np.where(v == i)[0]].sum() for i in cdf_v])
    return cdf_v, pdf_p


def _empirical(marginal_values, marginal_pdfs, n_draws, agg_function):
    """Monte Carlo joint cdf: independent uniform draws per dimension, quantile lookup per marginal."""
    dim = len(marginal_values)
    marginal_cumulative = [np.cumsum(pdf) for pdf in marginal_pdfs]

    uniform_points = np.random.uniform(size=(n_draws, dim))
    aggregated_points = np.empty(n_draws)
    for row, point in enumerate(uniform_points):
        quantiles = [
            marginal_values[i][
                min(
                    np.searchsorted(marginal_cumulative[i], point[i], side="left"),
                    len(marginal_values[i]) - 1,
                )
            ]
            for i in range(dim)
        ]
        aggregated_points[row] = agg_function(np.array(quantiles))

    cdf_v = np.unique(aggregated_points)
    pdf_p = np.array(
        [np.count_nonzero(aggregated_points == v) / len(aggregated_points) for v in cdf_v]
    )
    return cdf_v, pdf_p


def _drop_zero_probability_outcomes(
    cdf_v: np.ndarray, pdf_p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    nonzero = np.nonzero(pdf_p)
    return cdf_v[nonzero], pdf_p[nonzero]
