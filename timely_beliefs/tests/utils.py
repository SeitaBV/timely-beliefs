from typing import Union

import numpy as np


def equal_lists(list_a: Union[list, np.ndarray], list_b: Union[list, np.ndarray]):
    return all(np.isclose(a, b) for a, b in zip(list_a, list_b))
