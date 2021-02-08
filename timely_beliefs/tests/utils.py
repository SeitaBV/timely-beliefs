from datetime import datetime, timedelta
import time
from typing import Optional, Union

import numpy as np

import timely_beliefs as tb
from timely_beliefs.beliefs.classes import METADATA


def equal_lists(list_a: Union[list, np.ndarray], list_b: Union[list, np.ndarray]):
    return all(np.isclose(a, b) for a, b in zip(list_a, list_b))


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


class Timer(object):
    """Usage example:

    >>> with Timer("Calling max function"):
    >>>     a = max(range(10**6))
    <<<
    [Calling max function] Starting (at Friday, October 18, 2019 16:16:18) ...
    [Calling max function] Elapsed: 35 ms
    """

    def __init__(self, name=None, filename=None):
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()
        print(
            "[%s] Starting (at %s) ..."
            % (
                self.name,
                datetime.fromtimestamp(self.tstart).strftime("%A, %B %d, %Y %H:%M:%S"),
            )
        )

    def __exit__(self, type, value, traceback):
        duration = time.time() - self.tstart
        if duration > 1:
            message = "Elapsed: %.2f seconds" % duration
        else:
            message = "Elapsed: %.0f ms" % (duration * 1000)
        if self.name:
            message = "[%s] " % self.name + message
        print(message)
        if self.filename:
            with open(self.filename, "a") as file:
                print(str(datetime.now()) + ": ", message, file=file)
