from datetime import timedelta
from typing import Optional, Union

import numpy as np

import timely_beliefs as tb
from timely_beliefs.beliefs.classes import METADATA


def equal_lists(list_a: Union[list, np.ndarray], list_b: Union[list, np.ndarray]):
    """Equates lists by comparing their elements.

    If lists contain floats, they just have to be close within the default numpy tolerance.
    Also supports comparing missing values:
        None == None  # True
        np.isclose(np.nan, np.nan, equal_nan=True)  # True (np.nan == np.nan is normally considered as False)
    But:
        None == np.nan  # False
    """
    return len(list_a) == len(list_b) and all(
        a == b or np.isclose(a, b, equal_nan=True) for a, b in zip(list_a, list_b)
    )


def assert_metadata_is_retained(
    result_df: Union[tb.BeliefsDataFrame, tb.BeliefsSeries],
    original_df: tb.BeliefsDataFrame,
    is_series: bool = False,
    event_resolution: Optional[timedelta] = None,
):
    """Fail if result_df is not a BeliefsDataFrame with the same metadata as the original BeliefsDataFrame.

    Can also be used to check for a BeliefsSeries (using is_series=True).

    :param result_df: BeliefsDataFrame or BeliefsSeries to be checked for metadata propagation
    :param original_df: BeliefsDataFrame containing the original metadata
    :param is_series: if True, we check that the result is a BeliefsSeries rather than a BeliefsDataFrame
    :param event_resolution: optional timedelta in case we expect a different event_resolution than the original
    """
    metadata = {md: getattr(original_df, md) for md in METADATA}
    assert isinstance(
        result_df, tb.BeliefsDataFrame if not is_series else tb.BeliefsSeries
    )
    for md in metadata:
        if md == "event_resolution" and event_resolution is not None:
            assert result_df.event_resolution == event_resolution
        else:
            assert getattr(result_df, md) == metadata[md]
