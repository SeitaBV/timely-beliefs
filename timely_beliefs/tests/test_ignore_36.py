import pytest

from timely_beliefs import utils


@pytest.mark.parametrize(
    "td, ErrorType, match",
    [
        ("1M", ValueError, "not parse"),
        ("1Y", ValueError, "not parse"),
        ("1y", ValueError, "not parse"),
    ],
)
def test_ambiguous_timedelta_parsing(td, ErrorType, match):
    with pytest.raises(ErrorType, match=match):
        utils.parse_timedelta_like(td)
