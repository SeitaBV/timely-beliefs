from typing import Union
from datetime import datetime
import time

import numpy as np


def equal_lists(list_a: Union[list, np.ndarray], list_b: Union[list, np.ndarray]):
    return all(np.isclose(a, b) for a, b in zip(list_a, list_b))


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
