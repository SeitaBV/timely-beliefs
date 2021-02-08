import pytest

import numpy as np

from timely_beliefs.beliefs import probabilistic_utils


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("discrete"),
        pytest.param("linear"),
        pytest.param("pchip"),
    ],
)
@pytest.mark.parametrize(
    "extrapolate",
    [
        pytest.param(None),
        pytest.param(False),
        pytest.param(True),
        pytest.param("discrete"),
        pytest.param("linear"),
        pytest.param("exponential"),
    ],
)
@pytest.mark.parametrize(
    "cp, v",
    [
        pytest.param(
            [0.05, 0.5, 0.95], [0, 5, 20]
        ),  # cdf with three points, range(x) encompasses range(cdf)
        pytest.param([0, 0.5], [1, 2]),  # cdf with only two points
        pytest.param([1], [1]),  # cdf with only one point
        pytest.param(
            [0.2, 0.4], [30, 100]
        ),  # range(cdf) completely on the right of range(x)
        pytest.param([0.2, 0.4], [-30, 100]),  # range(cdf) encompasses range(x)
        pytest.param(
            [0.2, 0.8], [-30, -8]
        ),  # range(cdf) completely on the left of range(x)
        pytest.param(
            [0.05, 0.5, 0.6, 0.95], [0, 5, 5, 20]
        ),  # cdf with duplicate values v
    ],
)
def test_interpolation(method, extrapolate, cp, v):
    x = [-5, 0, 5, 10, 15, 20, 25]
    y = probabilistic_utils.interpolate_cdf(x, cp, v, method, extrapolate)
    assert len(y) == len(x)
    assert all(np.diff(x) >= 0)  # CDF x values should be sorted
    assert all(np.diff(y) >= 0)  # CDF y values should be sorted
    assert all(0 <= y) and all(y <= 1)  # normalised CDF should lie in the range [0, 1]


def test_empty_interpolation():
    x = np.linspace(0, 10, 5)
    cp = []
    v = []
    with pytest.raises(ValueError, match=r".*empty.*"):
        probabilistic_utils.interpolate_cdf(x, cp, v, "step")
