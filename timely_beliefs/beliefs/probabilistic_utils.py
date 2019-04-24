import math
from itertools import product
from typing import Union, List, Callable, Optional, Tuple

import numpy as np
import pyerf
import openturns as ot
import properscoring as ps
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from timely_beliefs.beliefs import classes  # noqa: F401
from timely_beliefs import utils as tb_utils


def interpret_complete_cdf(
    cdfs_p: List[Union[list, np.ndarray]],
    cdfs_v: List[Union[list, np.ndarray]],
    distribution: str = None,
) -> Union[
    List[Union[list, np.ndarray]],
    Tuple[List[Union[list, np.ndarray]], List[Union[list, np.ndarray]]],
    ot.DistributionImplementation,
]:
    """Interpret the given points on the cumulative distribution function to represent a complete CDF. The default
    policy is to assume discrete probabilities.
    If a distribution name is specified, the CDF is returned as an openturns distribution object.
    Supported openturns distributions are the following:
    - discrete: all residual probability is attributed to the highest given value
    - normal or gaussian: derived from the first two point only
    - uniform: derived from the first and last points and extended to range from cp=0 to cp=1
    """
    # Todo: refactor, currently too many possible types of output

    if distribution is None:
        for cdf_p in cdfs_p:
            cdf_p[-1] = 1  # Last value is the highest
        return cdfs_p, cdfs_v
    cdfs = []
    if distribution == "discrete":
        for cdf_p, cdf_v in zip(cdfs_p, cdfs_v):
            cdf_p[-1] = 1  # Last value is the highest
            cdfs.append(ot.UserDefined([[v] for v in cdf_v], cdf_p))
    elif distribution in ["normal", "gaussian"]:
        for cdf_p, cdf_v in zip(cdfs_p, cdfs_v):
            x1 = cdf_v[0]
            x2 = cdf_v[1]
            y1 = cdf_p[0]
            y2 = cdf_p[1]
            mu = (x1 * pyerf.erfinv(1 - 2 * y2) - x2 * pyerf.erfinv(1 - 2 * y1)) / (
                pyerf.erfinv(1 - 2 * y2) - pyerf.erfinv(1 - 2 * y1)
            )
            sigma = (2 ** 0.5 * x1 - 2 ** 0.5 * x2) / (
                2 * pyerf.erfinv(1 - 2 * y2) - 2 * pyerf.erfinv(1 - 2 * y1)
            )
            cdfs.append(ot.Normal(mu, sigma))
    elif distribution is "uniform":
        for cdf_p, cdf_v in zip(cdfs_p, cdfs_v):
            x1 = cdf_v[0]
            x2 = cdf_v[-1]
            y1 = cdf_p[0]
            y2 = cdf_p[-1]
            dydx = (y2 - y1) / (x2 - x1)
            a = x1 - y1 / dydx
            b = x2 + (1 - y2) / dydx
            cdfs.append(ot.Uniform(a, b))
    else:
        return NotImplementedError
    return cdfs


def probabilistic_nan_mean(
    df: "classes.BeliefsDataFrame",
    output_resolution,
    input_resolution,
    distribution: Optional[str] = None,
) -> "classes.BeliefsDataFrame":
    """Calculate the mean value while ignoring nan values."""

    if output_resolution < input_resolution:
        raise ValueError(
            "Cannot use a downsampling policy to upsample from %s to %s."
            % (input_resolution, output_resolution)
        )

    # Extract the probabilistic values that will serve as marginal distributions
    event_starts = df.groupby(["event_start"]).groups.keys()
    cdf_v = []
    cdf_p = []
    for e, event_start in enumerate(event_starts):
        vp = df.xs(event_start, level="event_start")  # value probability pair
        cdf_v.append(vp.values.flatten())
        cdf_p.append(vp.index.get_level_values("cumulative_probability").values)

    # Interpret cumulative probabilities as a description of the complete cdf, and calculate univariate joint cdf
    if distribution is None:
        cdf_p, cdf_v = interpret_complete_cdf(cdf_p, cdf_v)
        cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf(
            cdf_p, cdf_v, agg_function=np.nanmean
        )
    else:
        cdfs = interpret_complete_cdf(cdf_p, cdf_v, distribution=distribution)
        # Todo: allow passing a copula to this function
        cdf_p, cdf_v = multivariate_marginal_to_univariate_joint_cdf(
            cdfs, agg_function=np.nanmean
        )

    # Build up new BeliefsDataFrame slice with the new probabilistic values
    first_row = df.iloc[0:1]
    first_row = first_row.reset_index()
    df = pd.concat([first_row] * len(cdf_p), ignore_index=True)
    df["event_value"] = cdf_v
    df["cumulative_probability"] = cdf_p
    return df.set_index(
        ["event_start", "belief_time", "source", "cumulative_probability"]
    )


def multivariate_marginal_to_univariate_joint_cdf(  # noqa: C901
    marginal_cdfs_p: Union[
        List[Union[List[float], np.ndarray, ot.DistributionImplementation]], np.ndarray
    ],
    marginal_cdfs_v: Union[List[Union[List[float], np.ndarray]], np.ndarray] = None,
    a: float = 0,
    b: float = 1,
    copula: ot.CopulaImplementation = None,
    agg_function: Callable[[np.ndarray], np.ndarray] = None,
    simplify: bool = True,
    n_draws: int = 100,
    empirical: bool = False,
) -> Tuple[np.array, np.array]:
    """Calculate univariate joint CDF given a list of multivariate marginal CDFs and a copula,
    returning both the cumulative probabilities and the aggregated outcome of the random variables.

    :param: marginal_cdfs_p: Each marginal CDF is a list (or 2darray) with cumulative probabilities up to cp=1.
    If a cdf does not go up to cp=1 and there are few cdfs (low dimension), we can still evaluate possible combinations
    for each marginal cp given. That is, the remaining probability is attributed to some higher (but unknown) outcome.
    However, the empirical method can't be used.
    :param: marginal_cdfs_v: Values of possible outcomes for each random variable, i.e. the bins of the marginal CDFs.
    If just one set of bins is given, we assume the CDFs share the same set of bins.
    If no bins are specified (the default), we assume the CDFs share a set of equal-sized bins between a and b.

    "All bins are equal, but some bins are more equal than others." (because they have a higher probability)

    :param: a: The lowest outcome (0 by default, and ignored if CDF values are given explicitly)
    :param: b: The highest outcome (1 by default, and ignored if CDF values are given explicitly)
    :param: copula: The default copula is the independence copula (i.e. we assume independent random variables).
    :param: agg_function: The default aggregation function is to take the sum of the outcomes of the random variables.
    :param: simplify: Simplify the resulting cdf by removing possible outcomes with zero probability (True by default)
    :param: n_draws: Number of draws (sample size) to compute the empirical CDF when aggregating >3 random variables.
    :param: empirical: Compute the empirical CDF regardless of number of random variables (default is False)
    """

    dim = len(marginal_cdfs_p)
    n_outcomes = (
        99
    )  # Todo: refactor to avoid having to set this above our threshold for computing exact probabilities

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
                marginal_pdf = np.clip(
                    np.concatenate(([marginal_cdf[0]], np.diff(marginal_cdf))), 0, 1
                )
                marginals.append(
                    ot.UserDefined([[v] for v in values_for_cdf], marginal_pdf)
                )

    # If not specified, pick the independent copula as a default (i.e. assume independent random variables)
    if copula is None:
        copula = ot.IndependentCopula(dim)

    # If not specified, pick the sum function as a default for joining values
    if agg_function is None:
        agg_function = np.sum

    # Evaluate exact probabilities only for small bivariate and tri-variate joint distributions
    if dim <= 3 and n_outcomes <= 10 and empirical is False:

        # Determine joint distribution (too slow for high dimensions)
        d = ot.ComposedDistribution(marginals, copula)

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
                1
            )  # With just 1 point, an arbitrary positive distance suffices (e.g. 1)
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
        joint_multivariate_pdf = joint_cdf_to_pdf(joint_multivariate_cdf)

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


def joint_cdf_to_pdf(cdf: np.ndarray) -> np.ndarray:
    """Recursive function to determine the joint multivariate pdf from a given joint multivariate cdf."""

    if len(cdf.shape) > 1:
        pdf = cdf.copy()
        for i, cdf_i in enumerate(cdf):
            if i is not 0:
                pdf[i] = joint_cdf_to_pdf(cdf_i) - joint_cdf_to_pdf(cdf[i - 1])
            else:
                pdf[i] = joint_cdf_to_pdf(cdf_i)
        return pdf
    else:
        return np.concatenate(([cdf[0]], np.diff(cdf)))


def fill_zeros_with_last(arr):
    """Forward fill, e.g. [0, 0, 1, 0, 0, 2, 0] becomes [0, 0, 1, 1, 1, 2, 2]."""
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


def bin_it(
    binned_marginal_cdf_v: Union[List[float], np.ndarray],
    marginal_cdf_v: Union[np.ndarray, List[float]],
    marginal_cdf_p: Union[np.ndarray, List[float]],
):
    """Given outcome bins, and a marginal cdf (outcomes and probabilities), determine the binned marginal cdf."""
    binned_marginal_cdf_p = np.zeros(len(binned_marginal_cdf_v))
    for v, cp in zip(marginal_cdf_v, marginal_cdf_p):
        # Find nearest rather than an exact match
        binned_marginal_cdf_p[np.abs(binned_marginal_cdf_v - v).argmin()] = cp
    binned_marginal_cdf_p = fill_zeros_with_last(binned_marginal_cdf_p)
    return binned_marginal_cdf_p


def equalize_bins(
    cdf_values: Union[List[List[float]], np.ndarray],
    cdf_probabilities: List[List[float]],
    equal_bin_size: bool = False,
):
    """Define bins that cover all unique marginal outcomes, and compute each marginal cdf for these bins.
    Note that the bins do not necessarily have the same bin size. If this is needed, set equal_bin_size to True."""
    if equal_bin_size is False:
        values = np.unique(cdf_values)  # Also flattens and sorts
    else:
        import Fraction
        import functools
        import math

        values = np.array(cdf_values).flatten()
        v_min = np.min(values)
        v_max = np.max(values)
        v = [Fraction(x).limit_denominator().denominator for x in values]
        dv = 1 / functools.reduce(lambda a, b: a * b // math.gcd(a, b), v)
        values = np.linspace(v_min, v_max, int((v_max - v_min) // dv))
    return (
        values,
        np.array(
            [
                bin_it(values, cdf_v, cdf_p)
                for cdf_v, cdf_p in zip(cdf_values, cdf_probabilities)
            ]
        ),
    )


def set_truth(
    grouped: DataFrameGroupBy, right_source: "classes.BeliefSource"
) -> "classes.BeliefsDataFrame":
    """Overwrite the beliefs of each source by those of the given source.
    Terminology-wise, we say the given source is considered to be right,
    so it's beliefs contain the truth to be used as a reference for accuracy calculations.
    """

    # Pick out the group that is considered to contain the true observations
    gr_dict = dict(grouped.__iter__())
    if right_source in gr_dict:
        truth_group = gr_dict[right_source]
    else:
        raise KeyError("Source %s not found in BeliefsDataFrame." % right_source)

    # Replace each original group with the truth group, while adding back the source id for each original group
    gr_list = [
        tb_utils.replace_multi_index_level(
            truth_group, "source", pd.Index([key] * len(truth_group))
        )
        for key, group in grouped
    ]

    return pd.concat(gr_list)


def calculate_crps(df: "classes.BeliefsDataFrame") -> "classes.BeliefsDataFrame":
    """Compute the continuous ranked probability score for a BeliefsDataFrame with a probabilistic (or deterministic)
    forecast and observation. This function supports a probabilistic observation, too.

    References
    ----------
    Hans Hersbach. Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems
        in Weather and Forecasting, Volume 15, No. 5, pages 559-570, 2000.
        https://journals.ametsoc.org/doi/pdf/10.1175/1520-0434%282000%29015%3C0559%3ADOTCRP%3E2.0.CO%3B2
    """

    if len(df.groupby(level=["event_start", "source"])) > 1:
        raise ValueError(
            "Expected BeliefsDataFrame must describe a single observation and forecast."
            "BeliefsDataFrame cannot contain multiple events or sources."
        )

    # Split DataFrame into forecast and observation
    df_forecast = df.dropna(subset=["forecast"])["forecast"]
    df_observation = df.dropna(subset=["observation"])["observation"]

    # Obtain the distributions
    pdf_p_forecast, pdf_v_forecast = get_pdfs_from_beliefsdataframe(df_forecast)
    pdf_p_observation, pdf_v_observation = get_pdfs_from_beliefsdataframe(
        df_observation
    )

    # Check if we have both a forecast and an observation
    if pdf_p_forecast.size == 0 or pdf_p_observation.size == 0:
        crps = np.nan
    else:
        cdf_p_observation = pdf_p_observation.cumsum()
        cdf_p_forecast = pdf_p_forecast.cumsum()
        crpss = []

        # Loop over steps in cumulative probability (in case of a deterministic observation, this is a single step)
        previous_cp_observation = 0
        for p_observation, cp_observation, v_observation in zip(
            pdf_p_observation, cdf_p_observation, pdf_v_observation
        ):

            # Obtain the normalized pdf for this step
            cdf_p_forecast_i, cdf_v_forecast_i = partial_cdf(
                cdf_p_forecast,
                pdf_v_forecast,
                (previous_cp_observation, cp_observation),
            )
            pdf_p_forecast_i = np.concatenate(
                ([cdf_p_forecast_i[0]], np.diff(cdf_p_forecast_i))
            )

            # Calculate the continuous ranked profile score for this step (i.e. how well does the forecast descibe this possible outcome for the observation)
            crpss.append(
                ps.crps_ensemble(v_observation, cdf_v_forecast_i, pdf_p_forecast_i)
            )

            # Set the left cp bound for the next step
            previous_cp_observation = cp_observation

        # Calculate the weighted sum of scores over all possible outcomes for the observation.
        crps = np.dot(crpss, pdf_p_observation)

    # List the expected observation as the reference for determining percentage scores
    df_score = get_expected_belief(df_observation.to_frame())
    df_score.index = df_score.index.droplevel("cumulative_probability")

    # And of course return the score as well
    df_score["crps"] = crps

    return df_score


def partial_cdf(cdf_p: np.ndarray, cdf_v: np.ndarray, cp_range: Tuple[float, float]):
    """Calculate partial cdf within the given cumulative probability range."""

    # Select relevant probabilities within the given range
    left = np.searchsorted(cdf_p, cp_range[0], side="right")
    right = np.searchsorted(cdf_p, cp_range[1], side="left")
    cdf_p_to_consider = cdf_p[left : right + 1]

    # Transform (normalize cdf to range from 0 to 1)
    cdf_p_to_become = (cdf_p_to_consider - cp_range[0]) / (cp_range[1] - cp_range[0])
    cdf_p_to_become[-1] = 1

    return cdf_p_to_become, cdf_v[left : right + 1]


def get_cdfs_from_beliefsdataframe(
    df: "classes.BeliefsDataFrame"
) -> Tuple[np.ndarray, np.ndarray]:
    """From a BeliefsDataFrame with a single belief, get the cumulative distribution functions."""
    if df.empty:
        return np.empty(0), np.empty(0)

    pdf_v = df.values
    cdf_p = df.index.get_level_values("cumulative_probability").values

    # Todo: support interpretation as non-discrete distribution, e.g. uniform
    cdfs_p, cdfs_v = interpret_complete_cdf([cdf_p], [pdf_v])
    return cdfs_p[0], cdfs_v[0]


def get_pdfs_from_beliefsdataframe(
    df: "classes.BeliefsDataFrame"
) -> Tuple[np.ndarray, np.ndarray]:
    """From a BeliefsDataFrame with a single belief, get the probability distribution functions."""
    cdf_p, pdf_v = get_cdfs_from_beliefsdataframe(df)
    pdf_p = (
        np.concatenate(([cdf_p[0]], np.diff(cdf_p))) if cdf_p.size != 0 else np.empty(0)
    )
    return pdf_p, pdf_v


def get_belief_at_cumulative_probability(
    df: "classes.BeliefsDataFrame", cumulative_probability: float
) -> "classes.BeliefsDataFrame":
    """Take the first value with cumulative probability equal or higher than the probability given.
    This selects the right value assuming a discrete probability distribution."""
    df2 = df[
        df.index.get_level_values("cumulative_probability") >= cumulative_probability
    ]
    if df2.empty:
        # Take the value with the highest cumulative probability from the original DataFrame
        return df.tail(1)
    else:
        # Take the first value with a higher cumulative probability than given
        return df2.head(1)


def get_mean_belief(df: "classes.BeliefsDataFrame") -> "classes.BeliefsDataFrame":
    """Convenience function to select the expected value."""
    return get_belief_at_cumulative_probability(df, 0.5)


def get_nth_percentile_belief(
    df: "classes.BeliefsDataFrame", n: float
) -> "classes.BeliefsDataFrame":
    """Convenience function to select the value at the nth percentile."""
    return get_belief_at_cumulative_probability(df, n / 100)


get_expected_belief = get_mean_belief  # Define alias
