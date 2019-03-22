from typing import Callable, List, Optional, Tuple, Union
from datetime import timedelta
from itertools import product

import numpy as np
import pandas as pd
from pandas import Index
from pandas.tseries.frequencies import to_offset
from pandas.core.groupby import DataFrameGroupBy
import openturns as ot

from timely_beliefs.beliefs import classes
from timely_beliefs.sensors.func_store.event_values import nan_mean


def select_most_recent_belief(df: "classes.BeliefsDataFrame") -> "classes.BeliefsDataFrame":

    # Remember original index levels
    indices = df.index.names

    # Convert index levels to columns
    df = df.reset_index()

    # Drop all but most recent belief
    if "belief_horizon" in indices:
        df = df.sort_values(by=["belief_horizon"], ascending=True).drop_duplicates(subset=["event_start"], keep="first").sort_values(by=["event_start"])
    elif "belief_time" in indices:
        df = df.sort_values(by=["belief_time"], ascending=True).drop_duplicates(subset=["event_start"], keep="last").sort_values(by=["event_start"])
    else:
        raise KeyError("No belief_horizon or belief_time index level found in DataFrame.")

    # Convert columns to index levels (only columns that represent index levels)
    return df.set_index(indices)


def replace_multi_index_level(df: "classes.BeliefsDataFrame", level: str, index: Index, intersection: bool = False) -> "classes.BeliefsDataFrame":
    """Replace one of the index levels of the multi-indexed DataFrame.
    :param: df: a BeliefsDataFrame (or just a multi-indexed DataFrame).
    :param: level: the name of the index level to replace.
    :param: index: the new index.
    :param: intersection: policy for replacing the index level.
    If intersection is False then simply replace (note that the new index should have the same length as the old index).
    If intersection is True then add indices not contained in the old index and delete indices not contained in the new
    index. New rows have nan columns values and copies of the first row for other index levels (note that the resulting
    index is usually longer and contains values that were both in the old and new index, i.e. the intersection).
    """
    # Todo: check whether timezone information is copied over correctly

    if intersection is False and len(index) != len(df.index):
        raise ValueError(
            "Cannot simply replace multi-index level with an index of different length than the original. "
            "Use intersection instead?"
        )
    if index.name is None:
        index.name = level

    new_index_values = []
    new_index_names = []
    if intersection is True:
        contained_in_old = index.isin(df.index.get_level_values(level))
        new_index_not_in_old = index[~contained_in_old]
        contained_in_new = df.index.get_level_values(level).isin(index)
        for i in df.index.names:
            if i == level:  # For the index level that should be replaced
                # Copy old values that the new index contains, and add new values that the old index does not contain
                new_index_values.append(df.index.get_level_values(i)[contained_in_new].append(new_index_not_in_old))
                new_index_names.append(index.name)
            else:  # For the other index levels
                # Copy old values that the new index contains, and add the first value to the new rows
                new_row_values = pd.Index([df.index.get_level_values(i)[0]] * len(new_index_not_in_old))
                new_index_values.append(df.index.get_level_values(i)[contained_in_new].append(new_row_values))
                new_index_names.append(i)
    else:
        for i in df.index.names:
            if i == level:  # For the index level that should be replaced
                # Replace with new index
                new_index_values.append(index)
                new_index_names.append(index.name)
            else:  # For the other index levels
                # Copy all old values
                new_index_values.append(df.index.get_level_values(i))
                new_index_names.append(i)

    # Construct new MultiIndex
    mux = pd.MultiIndex.from_arrays(new_index_values, names=new_index_names)

    # Apply new MultiIndex
    if intersection is True:
        # Reindex such that new rows get nan column values
        df = df.reindex(mux)
    else:
        # Replace the index
        df.index = mux
    return df.sort_index()


def multiindex_upsampler(df, output_resolution: timedelta, input_resolution: timedelta, fill_method: Optional[str] = "pad"):
    """Upsample the first index level from input_resolution to output_resolution."""
    if df.empty:
        return df
    lvl0 = pd.date_range(start=df.index.get_level_values(0)[0], periods=input_resolution // output_resolution,
                         freq=to_offset(output_resolution).freqstr)
    new_index_values = [lvl0]
    if df.index.nlevels > 0:
        new_index_values.extend([[df.index.get_level_values(i)[0]] for i in range(1, df.index.nlevels)])
    mux = pd.MultiIndex.from_product(new_index_values, names=df.index.names)

    # Todo: allow customisation for de-aggregating event values
    if fill_method is None:
        return df.reindex(mux)
    elif fill_method == "pad":
        return df.reindex(mux).fillna(method="pad")  # pad is the reverse of mean downsampling
    else:
        raise NotImplementedError("Unknown upsample method.")


def respect_sensor_resolution(grouper: DataFrameGroupBy, resolution):
    """Resample to make sure the df slice contains events with the same frequency as the given resolution.
    The input BeliefsDataFrame (see below) should represent beliefs about sequential sub-events formed by a single source
    at a single unique belief time.
    Extra beliefs are added with nan values.
    The BeliefsDataFrame is not passed directly, but part of the DataFrameGroupBy object,
    which can be iterated over to obtain a BeliefsDataFrame of a unique belief time.

    We need to loop over each belief time in this slice, and reindex such that each subslice has rows for each event. Then recombine."""

    # Get a list of n groups, one group for each belief_time with info about how we sliced and the actual slice
    groups = list(grouper.__iter__())

    # Describe the event_start bin for the slices (we take the first, because the slices share the same event_start bin)
    bin_size = grouper.keys[0].freq
    bin_start = groups[0][0][0]
    bin_end = bin_start + bin_size

    # Build up our new BeliefsDataFrame (by copying over and emptying the rows the metadata should be copied over)
    df = groups[0][1].copy().iloc[0:0]
    for group in groups:  # Loop over the groups (we grouped by unique belief time and unique source id)

        # Get the BeliefsDataFrame for a unique belief time and source id
        df_slice = group[1]
        if not df_slice.empty:
            lvl0 = pd.date_range(start=bin_start, end=bin_end,
                                 freq=to_offset(resolution).freqstr, closed="left", name="event_start")
            df = df.append(replace_multi_index_level(df_slice, level="event_start", index=lvl0, intersection=True))

    return df


def propagate_beliefs(slice: "classes.BeliefsDataFrame", unique_belief_times) -> "classes.BeliefsDataFrame":
    """Propagate beliefs such that each event has the same set of unique belief times.
    The most recent belief about an event is valid until a new belief is formed.
    If no previous belief has been formed, a row is still explicitly included with a nan value.
    The input BeliefsDataFrame should represent beliefs about a single event formed by a single source.
    """

    # Get unique source id for this slice
    assert slice.lineage.number_of_sources == 1
    source_id = slice.lineage.sources[0]

    # Get unique event start for this slice
    event_start = slice.index.get_level_values(level="event_start").unique()
    assert len(event_start) == 1
    event_start = event_start[0]

    # Build up input data for new BeliefsDataFrame
    data = []
    previous_slice_with_existing_belief_time = None
    for ubt in unique_belief_times:

        # Check if the unique belief time (ubt) is already in the DataFrame
        slice_with_existing_belief_time = slice.xs(ubt, level="belief_time", drop_level=False)
        if slice_with_existing_belief_time.empty:

            # If not already present, create a new row with the most recent belief (or nan if no previous exists)
            if previous_slice_with_existing_belief_time is not None:
                ps = previous_slice_with_existing_belief_time.reset_index()
                ps["belief_time"] = ubt  # Update belief time to reflect propagation of beliefs over time (you still believe what you believed before).
                data.extend(ps.values.tolist())
            else:
                data.append([event_start, ubt, source_id, np.nan, np.nan])
        else:
            # If already present, copy the row (may be multiple rows in case of a probabilistic belief)
            data.extend(slice_with_existing_belief_time.reset_index().values.tolist())
        previous_slice_with_existing_belief_time = slice_with_existing_belief_time

    # Create new BeliefsDataFrame
    df = slice.copy().reset_index().iloc[0:0]
    sensor = df.sensor
    df = df.append(
        pd.DataFrame(data, columns=["event_start", "belief_time", "source_id", "belief_percentile", "event_value"]))
    df.sensor = sensor
    df = df.set_index(["event_start", "belief_time", "source_id", "belief_percentile"])

    return df


def convolve_events(slice: "classes.BeliefsDataFrame", output_resolution: timedelta, input_resolution: timedelta) -> "classes.BeliefsDataFrame":
    """
    Determine the probabilistic beliefs about the aggregated event.
    The input BeliefsDataFrame should represent beliefs about sequential sub-events formed by a single source
    at a single unique belief time (in case of downsampling), or about a single super-event (in case of upsampling).
    """

    if output_resolution > input_resolution:
        # Todo: implement probabilistic downsampling policy assuming independent random variables
        belief_percentiles = list(slice.index.get_level_values(level="belief_percentile"))
        if not all(p == 0.5 or np.isnan(p) for p in belief_percentiles) :
            raise NotImplementedError

        # Create new BeliefsDataFrame with downsampled event_start
        df = slice.groupby(
            [pd.Grouper(freq=to_offset(output_resolution).freqstr, level="event_start"), "belief_time", "source_id",
             ], group_keys=False).apply(lambda x: nan_mean(x, output_resolution, input_resolution))  # Todo: allow customisation for aggregating event values
    else:
        # Create new BeliefsDataFrame with upsampled event_start
        if input_resolution % output_resolution != timedelta():
            raise NotImplementedError("Cannot upsample from resolution %s to %s." % (input_resolution, output_resolution))
        df = slice.groupby(
            [pd.Grouper(freq=to_offset(output_resolution).freqstr, level="event_start"), "belief_time", "source_id",
             "belief_percentile"], group_keys=False).apply(lambda x: multiindex_upsampler(x, output_resolution, input_resolution))
    return df


def beliefs_resampler(df: "classes.BeliefsDataFrame", output_resolution: timedelta, input_resolution: timedelta) -> "classes.BeliefsDataFrame":
    """For a unique source id."""

    # Determine unique set of belief times
    unique_belief_times = np.sort(df.reset_index()["belief_time"].unique())  # Sorted from past to present

    # Propagate beliefs so that each event has the same set of unique belief times
    df = df.groupby(["event_start"], group_keys=False).apply(lambda x: propagate_beliefs(x, unique_belief_times))

    # Resample to make sure the df slice contains events with the same frequency as the input_resolution
    # (make nan rows if you have to)
    # Todo: this is only necessary when the resampling policy for the event value needs to take into account nan values within the slice, so move it closer to the convolve_events() method
    # df = df.groupby(
    #     [pd.Grouper(freq=to_offset(output_resolution).freqstr, level="event_start"), "belief_time", "source_id"]
    # ).pipe(respect_sensor_resolution, input_resolution)

    # For each unique belief time, determine the probabilistic belief about the aggregated event
    df = df.groupby(["belief_time"], group_keys=False).apply(lambda x: convolve_events(x, output_resolution, input_resolution))

    return df


def multivariate_marginal_to_univariate_joint_cdf(
    marginal_cdfs_p: Union[List[Union[List[float], np.ndarray]], np.ndarray],
    marginal_cdfs_v: Union[List[Union[List[float], np.ndarray]], np.ndarray] = None,
    a: float = 0,
    b: float = 1,
    copula: ot.CopulaImplementation = None,
    agg_function: Callable[[np.ndarray], np.ndarray] = None,
    simplify: bool = True,
    n_draws: int = 100,
) -> Tuple[np.array, np.array]:
    """Calculate univariate joint CDF given a list of multivariate marginal CDFs and a copula,
    returning both the cumulative probabilities and the aggregated outcome of the random variables.

    :param: marginal_cdfs_p: Each marginal CDF is a list (or 2darray) with cumulative probabilities. The cdfs do not
    have to go up to cp=1, as we simply evaluate possible combinations for each marginal cp given. That is, the
    remaining probability is attributed to some higher (but unknown) outcome.
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
    """

    # Set up marginal cdf values
    n_outcomes = len(marginal_cdfs_p[0])
    dim = len(marginal_cdfs_p)
    shared_bins = True
    if marginal_cdfs_v is None:
        values = np.linspace(a, b, n_outcomes)
    elif isinstance(marginal_cdfs_v[0], (list, np.ndarray)):
        shared_bins = False
        values = marginal_cdfs_v
    else:
        values = marginal_cdfs_v

    # Set up marginal distributions
    marginals = []
    for i in range(dim):
        marginal_cdf = marginal_cdfs_p[i]
        if shared_bins is True:
            values_for_cdf = values
        else:
            values_for_cdf = marginal_cdfs_v[i]
        if marginal_cdf[-1] != 1:
            # We can assume some higher outcome exists with cp=1
            values_for_cdf = np.append(values_for_cdf, values_for_cdf[-1]+1)  # Add a higher outcome (+1 suffices)
            marginal_pdf = np.clip(np.concatenate(([marginal_cdf[0]], np.diff(marginal_cdf), [1. - marginal_cdf[-1]])), 0, 1)
            marginals.append(ot.UserDefined([[v] for v in values_for_cdf], marginal_pdf))
        else:
            marginal_pdf = np.clip(np.concatenate(([marginal_cdf[0]], np.diff(marginal_cdf))), 0, 1)
            marginals.append(ot.UserDefined([[v] for v in values_for_cdf], marginal_pdf))

    # If not specified, pick the independent copula as a default (i.e. assume independent random variables)
    if copula is None:
        copula = ot.IndependentCopula(dim)

    # If not specified, pick the sum function as a default for joining values
    if agg_function is None:
        agg_function = np.sum

    # Evaluate exact probabilities only for small bivariate and tri-variate joint distributions
    if dim <= 3 and n_outcomes <= 10:

        # Determine joint distribution (too slow for high dimensions)
        d = ot.ComposedDistribution(marginals, copula)

        # Construct an n-dimensional matrix with all possible points (i.e. combinations of outcomes of our random variables)
        if shared_bins is True:
            matrix = list(product(values, repeat=dim))
            shape = (n_outcomes,) * dim
        else:
            matrix = list(product(*marginal_cdfs_v))
            shape = [len(m) for m in marginal_cdfs_v]

        # Evaluate exact probabilities at each point (too slow for high dimensions)
        joint_multivariate_cdf = np.reshape(d.computeCDF(matrix), shape)
        joint_multivariate_pdf = joint_cdf_to_pdf(joint_multivariate_cdf)

        # Sort the probabilities ascending, keeping track of the corresponding values
        p, v = zip(*sorted(zip(joint_multivariate_pdf.flatten(), agg_function(matrix, 1))))

        # Calculate total probability of each unique value (by adding probability of cases that yield the same value)
        cdf_v = np.unique(v)
        pdf_p = np.array([sum(np.array(p)[np.where(v == i)[0]]) for i in cdf_v])
    else:  # Otherwise, compute the empirical cdf from a sample generated directly from the copula
        uniform_points = np.array(copula.getSample(n_draws))  # Much faster than sampling from the joint cdf
        aggregated_points = np.zeros(n_draws)
        for i, point in enumerate(uniform_points):
            aggregated_points[i] = agg_function(marginal_cdf.computeQuantile(marginal_cdf_p)[0] for marginal_cdf_p, marginal_cdf in zip(point, marginals))
        empirical_cdf = ot.UserDefined([[v] for v in aggregated_points])
        pdf_p = np.array(empirical_cdf.getP())
        cdf_v = np.array(empirical_cdf.getX()).flatten()

    # Simplify resulting pdf
    if simplify is True:
        cdf_v = cdf_v[np.nonzero(pdf_p)]
        pdf_p = pdf_p[np.nonzero(pdf_p)]

    # Return the univariate joint cumulative probability function and transform to the desired range of outcome
    cdf_p = pdf_p.cumsum()
    # cdf_v = agg_function([a] * dim) + (b - a) * cdf_v

    return cdf_p, cdf_v


def joint_cdf_to_pdf(cdf: np.ndarray) -> np.ndarray:
    """Recursive function to determine the joint multivariate pdf from a given joint multivariate cdf."""

    if len(cdf.shape) > 1:
        pdf = cdf.copy()
        for i, cdf_i, in enumerate(cdf):
            if i is not 0:
                pdf[i] = joint_cdf_to_pdf(cdf_i) - joint_cdf_to_pdf(cdf[i-1])
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
    marginal_cdf_p: Union[np.ndarray, List[float]]
):
    """Given outcome bins, and a marginal cdf (outcomes and probabilities), determine the binned marginal cdf."""
    binned_marginal_cdf_p = np.zeros(len(binned_marginal_cdf_v))
    for v, cp in zip(marginal_cdf_v, marginal_cdf_p):
        # Find nearest rather than an exact match
        binned_marginal_cdf_p[np.abs(binned_marginal_cdf_v - v).argmin()] = cp
    binned_marginal_cdf_p = fill_zeros_with_last(binned_marginal_cdf_p)
    return binned_marginal_cdf_p


def equalize_bins(cdf_values: Union[List[List[float]], np.ndarray], cdf_probabilities: List[List[float]], equal_bin_size: bool = False):
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
        dv = 1 / functools.reduce(lambda a, b : a * b // math.gcd(a, b), v)
        values = np.linspace(v_min, v_max, int((v_max-v_min) // dv))
    return values, np.array([bin_it(values, cdf_v, cdf_p) for cdf_v, cdf_p in zip(cdf_values, cdf_probabilities)])
