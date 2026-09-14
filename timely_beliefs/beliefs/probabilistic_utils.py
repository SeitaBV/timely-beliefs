from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from timely_beliefs import utils as tb_utils
from timely_beliefs.beliefs import classes  # noqa: F401
from timely_beliefs.beliefs.crps import crps_ensemble
from timely_beliefs.beliefs.probabilistic_math import cp_to_p  # noqa: F401 (re-exported)

# Two ways to compute a joint cdf over several (marginal) beliefs: independent_joint_cdf below
# (independent beliefs, plain numpy) or probabilistic_backend.joint_cdf_with_openturns_copula
# (custom copula or openturns-distribution marginals). Caller picks explicitly.
from timely_beliefs.beliefs.independent_joint_cdf import (  # noqa: E402, F401
    independent_joint_cdf,
)

if TYPE_CHECKING:
    # Only for type hints below; never imported at runtime. See probabilistic_backend.py.
    import openturns as ot  # noqa: F401


def interpret_complete_cdf(
    cdfs_p: list[list | np.ndarray],
    cdfs_v: list[list | np.ndarray],
    distribution: str | None = None,
) -> (
    tuple[list[list | np.ndarray], list[list | np.ndarray]]
    | list[ot.DistributionImplementation]
):
    """Interpret the given points on the cumulative distribution function to represent a complete
    CDF, by assuming all residual probability is attributed to the highest given value.
    Passing `distribution` is deprecated; call probabilistic_backend.interpret_complete_cdf_as_distribution instead.
    """
    if distribution is not None:
        warnings.warn(
            "interpret_complete_cdf(..., distribution=...) is deprecated and will be removed in a "
            "future version. Call probabilistic_backend.interpret_complete_cdf_as_distribution "
            "directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from timely_beliefs.beliefs import probabilistic_backend as backend

        return backend.interpret_complete_cdf_as_distribution(cdfs_p, cdfs_v, distribution)

    for cdf_p in cdfs_p:
        cdf_p[-1] = 1  # Last value is the highest
    return cdfs_p, cdfs_v


def joint_cdf_with_copula(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated. Use probabilistic_backend.joint_cdf_with_openturns_copula or independent_joint_cdf instead."""
    warnings.warn(
        "joint_cdf_with_copula is deprecated and will be removed in a future version. Use "
        "probabilistic_backend.joint_cdf_with_openturns_copula or independent_joint_cdf instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from timely_beliefs.beliefs import probabilistic_backend as backend

    return backend.joint_cdf_with_openturns_copula(
        marginal_cdfs_p,
        marginal_cdfs_v,
        a,
        b,
        copula,
        agg_function,
        simplify,
        n_draws,
        empirical,
    )


def probabilistic_nan_mean(
    df: "classes.BeliefsDataFrame",
    output_resolution,
    input_resolution,
    distribution: str | None = None,
) -> "classes.BeliefsDataFrame":
    """Calculate the mean value while ignoring nan values."""

    if output_resolution < input_resolution:
        raise ValueError(
            "Cannot use a downsampling policy to upsample from %s to %s."
            % (input_resolution, output_resolution)
        )

    # Extract the probabilistic values that will serve as marginal distributions
    event_starts = df.groupby(["event_start"], group_keys=False).groups.keys()
    cdf_v = []
    cdf_p = []
    all_deterministic = True
    for event_start in event_starts:
        vp = df.xs(event_start, level="event_start")  # value probability pair
        values = vp.values.flatten()
        if len(values) > 1:
            all_deterministic = False
        cdf_v.append(values)
        cdf_p.append(vp.index.get_level_values("cumulative_probability").values)

    if all_deterministic:
        # Every event in this slice is a single deterministic belief (one row, cp=1), so the mean
        # across them is a plain arithmetic mean. Short-circuit to avoid building marginal
        # distributions, a joint distribution and a product-space quantile search for data that never
        # needed any of that (~1.6x faster on resample_events over deterministic multi-source data).
        mean_value = np.nanmean([v[0] for v in cdf_v])
        first_row = df.iloc[0:1]
        first_row = first_row.reset_index()
        df = pd.concat([first_row], ignore_index=True)
        df["event_value"] = [mean_value]
        df["cumulative_probability"] = [1.0]
        return df.set_index(
            ["event_start", "belief_time", "source", "cumulative_probability"]
        )

    # Interpret cumulative probabilities as a description of the complete cdf, and calculate univariate joint cdf
    if distribution is None:
        # Plain array marginals, independent copula: the common case, needs no openturns.
        cdf_p, cdf_v = interpret_complete_cdf(cdf_p, cdf_v)
        cdf_p, cdf_v = independent_joint_cdf(cdf_p, cdf_v, agg_function=np.nanmean)
    else:
        # distribution="discrete"/"normal"/"uniform" builds openturns distribution objects as
        # marginals, so this path always needs openturns, even though the copula is still the
        # (implicit) independent one.
        # Todo: allow passing a copula to this function
        from timely_beliefs.beliefs import probabilistic_backend as backend

        cdfs = backend.interpret_complete_cdf_as_distribution(cdf_p, cdf_v, distribution)
        cdf_p, cdf_v = backend.joint_cdf_with_openturns_copula(cdfs, agg_function=np.nanmean)

    # Build up new BeliefsDataFrame slice with the new probabilistic values
    first_row = df.iloc[0:1]
    first_row = first_row.reset_index()
    df = pd.concat([first_row] * len(cdf_p), ignore_index=True)
    df["event_value"] = cdf_v
    df["cumulative_probability"] = cdf_p
    return df.set_index(
        ["event_start", "belief_time", "source", "cumulative_probability"]
    )


def fill_zeros_with_last(arr):
    """Forward fill, e.g. [0, 0, 1, 0, 0, 2, 0] becomes [0, 0, 1, 1, 1, 2, 2]."""
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


def bin_it(
    binned_marginal_cdf_v: list[float] | np.ndarray,
    marginal_cdf_v: np.ndarray | list[float],
    marginal_cdf_p: np.ndarray | list[float],
):
    """Given outcome bins, and a marginal cdf (outcomes and probabilities), determine the binned marginal cdf."""
    binned_marginal_cdf_p = np.zeros(len(binned_marginal_cdf_v))
    for v, cp in zip(marginal_cdf_v, marginal_cdf_p):
        # Find nearest rather than an exact match
        binned_marginal_cdf_p[np.abs(binned_marginal_cdf_v - v).argmin()] = cp
    binned_marginal_cdf_p = fill_zeros_with_last(binned_marginal_cdf_p)
    return binned_marginal_cdf_p


def equalize_bins(
    cdf_values: list[list[float]] | np.ndarray,
    cdf_probabilities: list[list[float]],
    equal_bin_size: bool = False,
):
    """Define bins that cover all unique marginal outcomes, and compute each marginal cdf for these bins.
    Note that the bins do not necessarily have the same bin size. If this is needed, set equal_bin_size to True.
    """
    if equal_bin_size is False:
        values = np.unique(cdf_values)  # Also flattens and sorts
    else:
        import functools
        import math

        import Fraction

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
    so its beliefs contain the truth to be used as a reference for accuracy calculations.
    """

    # Pick out the group that is considered to contain the true observations
    gr_dict = dict(grouped.__iter__())
    if right_source in gr_dict:
        truth_group = gr_dict[right_source]
    else:
        raise KeyError("Source %s not found in BeliefsDataFrame." % right_source)

    # Replace each original group with the truth group, while adding back the source for each original group
    gr_list = [
        tb_utils.replace_multi_index_level(
            truth_group, "source", pd.Index([key] * len(truth_group))
        )
        for key, group in grouped
    ]

    return pd.concat(gr_list)


def calculate_crps(df: "classes.BeliefsDataFrame") -> "classes.BeliefsDataFrame":
    """Compute the continuous ranked probability score for a BeliefsDataFrame with a probabilistic (or deterministic)
    forecast (event_value column) and observation (reference_value column).
    This function supports a probabilistic observation, too.

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

    # Split DataFrame into forecast (event_value) and observation (reference_value)
    df_forecast = df.dropna(subset=["event_value"])["event_value"]
    df_observation = df.dropna(subset=["reference_value"])["reference_value"]

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
        for cp_observation, v_observation in zip(cdf_p_observation, pdf_v_observation):

            # Obtain the normalized pdf for this step
            cdf_p_forecast_i, cdf_v_forecast_i = partial_cdf(
                cdf_p_forecast,
                pdf_v_forecast,
                (previous_cp_observation, cp_observation),
            )
            pdf_p_forecast_i = np.concatenate(
                ([cdf_p_forecast_i[0]], np.diff(cdf_p_forecast_i))
            )

            # Calculate the continuous ranked profile score for this step (i.e. how well does the forecast describe this possible outcome for the observation)
            crpss.append(
                crps_ensemble(v_observation, cdf_v_forecast_i, pdf_p_forecast_i)
            )

            # Set the left cp bound for the next step
            previous_cp_observation = cp_observation

        # Calculate the weighted sum of scores over all possible outcomes for the observation.
        crps = np.dot(crpss, pdf_p_observation)

    # List the middle observation as the reference for determining percentage scores
    df_score = get_median_belief(df_observation.to_frame())
    df_score = df_score.droplevel("cumulative_probability")

    # And of course return the score as well
    df_score["crps"] = crps

    return df_score


def partial_cdf(cdf_p: np.ndarray, cdf_v: np.ndarray, cp_range: tuple[float, float]):
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
    df: "classes.BeliefsDataFrame",
) -> tuple[np.ndarray, np.ndarray]:
    """From a BeliefsDataFrame with a single belief, get the cumulative distribution functions."""
    if df.empty:
        return np.empty(0), np.empty(0)

    pdf_v = df.values
    cdf_p = df.index.get_level_values("cumulative_probability").values

    # Todo: support interpretation as non-discrete distribution, e.g. uniform
    cdfs_p, cdfs_v = interpret_complete_cdf([cdf_p], [pdf_v])
    return cdfs_p[0], cdfs_v[0]


def get_pdfs_from_beliefsdataframe(
    df: "classes.BeliefsDataFrame",
) -> tuple[np.ndarray, np.ndarray]:
    """From a BeliefsDataFrame with a single belief, get the probability distribution functions."""
    cdf_p, pdf_v = get_cdfs_from_beliefsdataframe(df)
    pdf_p = cp_to_p(cdf_p)
    return pdf_p, pdf_v


def get_belief_at_cumulative_probability(
    df: "classes.BeliefsDataFrame", cumulative_probability: float
) -> "classes.BeliefsDataFrame":
    """Take the first value with cumulative probability equal or higher than the probability given.
    This selects the right value assuming a discrete probability distribution."""
    if not len(df) > 1:
        return df
    df2 = df[
        df.index.get_level_values("cumulative_probability") >= cumulative_probability
    ]
    if df2.empty:
        # Take the value with the highest cumulative probability from the original DataFrame
        return df.tail(1)
    else:
        # Take the first value with a higher cumulative probability than given
        return df2.head(1)


def get_mean_belief(
    df: "classes.BeliefsDataFrame", distribution: str = "uniform"
) -> "classes.BeliefsDataFrame":
    """Convenience function to select the expected value.
    Assumes the data frame contains a single belief per event.
    """
    event_starts = df.groupby(["event_start"]).groups.keys()
    cdf_v = []
    cdf_p = []
    for event_start in event_starts:
        vp = df.xs(event_start, level="event_start")  # value probability pair
        cdf_v.append(vp.values.flatten())
        cdf_p.append(vp.index.get_level_values("cumulative_probability").values)

    # Interpret cumulative probabilities as a description of the complete cdf, and calculate means
    from timely_beliefs.beliefs import probabilistic_backend as backend

    cdfs: list[ot.DistributionImplementation] = (
        backend.interpret_complete_cdf_as_distribution(cdf_p, cdf_v, distribution)
    )
    means = [cdf.getMean()[0] for cdf in cdfs]
    # Get the cumulative probability at the mean value, which may differ from 0.5 for asymmetric distributions
    cp_at_means = [cdf.computeCDF(mean) for cdf, mean in zip(cdfs, means)]

    # Convert from probabilistic to deterministic beliefs, assigning the mean
    df = df.groupby(level=["event_start"], group_keys=False).apply(lambda x: x.head(1))
    df = tb_utils.replace_multi_index_level(
        df,
        "cumulative_probability",
        pd.Index(data=cp_at_means),
    )
    df["event_value"] = means
    return df


def get_median_belief(df: "classes.BeliefsDataFrame") -> "classes.BeliefsDataFrame":
    """Convenience function to select the middle value (50th percentile).
    Assumes the data frame contains a single belief.
    """
    return get_belief_at_cumulative_probability(df, 0.5) if len(df) > 1 else df


def get_nth_percentile_belief(
    df: "classes.BeliefsDataFrame", n: float
) -> "classes.BeliefsDataFrame":
    """Convenience function to select the value at the nth percentile."""
    return get_belief_at_cumulative_probability(df, n / 100) if len(df) > 1 else df


get_expected_belief = get_mean_belief  # Define alias
