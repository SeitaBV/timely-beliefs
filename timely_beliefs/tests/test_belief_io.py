import pytest
import os
import pandas as pd
import pytz
from datetime import datetime, timedelta

from timely_beliefs.examples import example_df
from timely_beliefs.beliefs.classes import METADATA
import timely_beliefs as tb


@pytest.fixture(scope="module")
def csv_file(tmpdir_factory):
    """Save BeliefsDataFrame to csv."""

    example_df.to_csv("test.csv")
    yield
    os.remove("test.csv")


def test_load_beliefs(csv_file):
    """Test loading BeliefsDataFrame to csv.
    The saved file does not contain the sensor information, and the sources are saved by their name.
    Therefore, we test the following functionality:
    - The user should specify the sensor upon loading
    - The user should be warned that the loaded sources are not of type BeliefSource.
    - The user should have the possibility to look up the saved source names by passing a list of sources.
    """

    # Load beliefs with tb.read_csv
    df = pd.read_csv("test.csv")
    with pytest.warns(UserWarning, match="created"):
        df = tb.BeliefsDataFrame(df, sensor=tb.Sensor("Sensor Y"))
    assert df.sensor.name == "Sensor Y"

    # No lookup should issue warning
    with pytest.warns(UserWarning, match="looking them up"):
        df = tb.read_csv("test.csv", sensor=tb.Sensor("Sensor Y"))
    for s in df.index.get_level_values("source"):
        assert isinstance(
            s, tb.BeliefSource
        )  # Source names automatically get converted to sources
    assert all(c == tb.BeliefSource for c in df.sources.map(type))

    # This lookup should fail
    with pytest.raises(ValueError, match="not in list"):
        tb.read_csv(
            "test.csv",
            sensor=tb.Sensor("Sensor Y"),
            look_up_sources=[tb.BeliefSource(name="Source X")],
        )

    # This lookup should succeed
    source_a, source_b = tb.BeliefSource("Source A"), tb.BeliefSource("Source B")
    df = tb.read_csv(
        "test.csv", sensor=tb.Sensor("Sensor Y"), look_up_sources=[source_a, source_b]
    )
    assert df.sensor.name == "Sensor Y"
    assert source_a in df.index.get_level_values("source")
    assert source_b in df.index.get_level_values("source")
    assert isinstance(df.index.get_level_values("event_start")[0], datetime)
    assert isinstance(df.index.get_level_values("belief_time")[0], datetime)


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ([], {}),
        ([pd.DataFrame()], {}),
        ([pd.Series()], {}),
        ([], {"sensor": tb.Sensor("test")}),
    ],
)
def test_empty_beliefs(args, kwargs):
    """Test construction of empty BeliefsDataFrame."""

    bdf = tb.BeliefsDataFrame(*args, **kwargs)
    assert bdf.empty
    if bdf.sensor:
        assert bdf.sensor.name == "test"
    else:
        assert bdf.event_resolution is None
    assert "event_value" in bdf
    for name in ["event_start", "belief_time", "source", "cumulative_probability"]:
        assert name in bdf.index.names


@pytest.mark.parametrize(
    "missing_column_name, data, present_column_names",
    [
        (
            "event_start",
            [
                [1, datetime(2000, 1, 1, tzinfo=pytz.utc), 3, 4],
                [5, datetime(2000, 1, 1, tzinfo=pytz.utc), 7, 8],
            ],
            ["source", "belief_time", "cumulative_probability", "event_value"],
        ),
        (
            "belief_time",
            [
                [datetime(2000, 1, 1, tzinfo=pytz.utc), 2, 3, 4],
                [datetime(2000, 1, 1, tzinfo=pytz.utc), 6, 7, 8],
            ],
            ["event_start", "source", "cumulative_probability", "event_value"],
        ),
        (
            "source",
            [
                [
                    datetime(2000, 1, 1, tzinfo=pytz.utc),
                    datetime(2000, 1, 1, tzinfo=pytz.utc),
                    3,
                    4,
                ],
                [
                    datetime(2000, 1, 1, tzinfo=pytz.utc),
                    datetime(2000, 1, 1, tzinfo=pytz.utc),
                    7,
                    8,
                ],
            ],
            ["event_start", "belief_time", "cumulative_probability", "event_value"],
        ),
        (
            "event_value",
            [
                [
                    datetime(2000, 1, 1, tzinfo=pytz.utc),
                    datetime(2000, 1, 1, tzinfo=pytz.utc),
                    3,
                    4,
                ],
                [
                    datetime(2000, 1, 1, tzinfo=pytz.utc),
                    datetime(2000, 1, 1, tzinfo=pytz.utc),
                    7,
                    8,
                ],
            ],
            ["event_start", "belief_time", "cumulative_probability", "source"],
        ),
    ],
)
def test_incomplete_beliefs(missing_column_name, data, present_column_names):
    """Test exceptions are thrown when input data is missing required column headers.
    Only cumulative_probability can be missed, since it is has a default value."""
    df = pd.DataFrame(data, columns=present_column_names)

    with pytest.raises(KeyError, match=missing_column_name):
        with pytest.warns(UserWarning, match="created"):
            tb.BeliefsDataFrame(df)


@pytest.mark.parametrize(
    "invalid_column, data, column_names",
    [
        (
            "event_start",
            [[1, timedelta(), 3, 4]],
            ["event_start", "belief_horizon", "event_value", "source"],
        ),  # event_start is not a datetime
        (
            "event_start",
            [[datetime(2000, 1, 1), timedelta(), 3, 4]],
            ["event_start", "belief_horizon", "event_value", "source"],
        ),  # event_start is missing timezone
        (
            "belief_horizon",
            [[datetime(2000, 1, 1, tzinfo=pytz.utc), 2, 3, 4]],
            ["event_start", "belief_horizon", "event_value", "source"],
        ),  # belief_horizon is not a timedelta
        (
            "belief_time",
            [[datetime(2000, 1, 1, tzinfo=pytz.utc), 2, 3, 4]],
            ["event_start", "belief_time", "event_value", "source"],
        ),  # belief_time is not a datetime
        (
            "source",
            [[datetime(2000, 1, 1, tzinfo=pytz.utc), timedelta(), 3, None]],
            ["event_start", "belief_horizon", "event_value", "source"],
        ),  # source is None
    ],
)
def test_invalid_beliefs(invalid_column, data, column_names):
    """Test exceptions are thrown when input data is of the wrong type."""
    df = pd.DataFrame(data, columns=column_names)
    with pytest.raises(TypeError, match=invalid_column):
        with pytest.warns(UserWarning, match="created"):
            tb.BeliefsDataFrame(df)


@pytest.mark.parametrize(
    "args, kwargs",
    [
        (
            [
                pd.DataFrame(
                    [[datetime(2000, 1, 1, tzinfo=pytz.utc), timedelta(), 3, 4]],
                    columns=["event_start", "belief_horizon", "source", "event_value"],
                )
            ],
            {"sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1))},
        ),
        (
            [
                pd.DataFrame(
                    [
                        [
                            datetime(2000, 1, 1, tzinfo=pytz.utc),
                            datetime(2000, 1, 1, hour=1, tzinfo=pytz.utc),
                            3,
                            4,
                        ]
                    ],
                    columns=["event_start", "belief_time", "source", "event_value"],
                )
            ],
            {"sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1))},
        ),
        (
            [
                pd.DataFrame(
                    [[datetime(2000, 1, 1, tzinfo=pytz.utc), timedelta(), 4]],
                    columns=["event_start", "belief_horizon", "event_value"],
                )
            ],
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
            },
        ),  # move source to keyword argument
        (
            [
                pd.DataFrame(
                    [[datetime(2000, 1, 1, tzinfo=pytz.utc), 4]],
                    columns=["event_start", "event_value"],
                )
            ],
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
                "belief_horizon": timedelta(),
            },
        ),  # move source and belief_horizon to keyword argument
        (
            [pd.DataFrame([[4]], columns=["event_value"])],
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
                "belief_horizon": timedelta(),
                "event_start": datetime(2000, 1, 1, tzinfo=pytz.utc),
            },
        ),  # move source, belief_horizon and event_start to keyword argument
        (
            [pd.Series([4])],
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
                "belief_horizon": timedelta(),
                "event_start": datetime(2000, 1, 1, tzinfo=pytz.utc),
            },
        ),  # move source, belief_horizon and event_start to keyword argument and use Series instead of DataFrame
        (
            [
                pd.Series(
                    [4], index=pd.DatetimeIndex([datetime(2000, 1, 1, tzinfo=pytz.utc)])
                )
            ],
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
                "belief_horizon": timedelta(),
            },
        ),  # move source and belief_horizon keyword argument and use Series instead of DataFrame
    ],
)
def test_belief_setup_with_data_frame(args, kwargs):
    """Test different ways of setting up the same BeliefsDataFrame."""
    with pytest.warns(UserWarning, match="created"):
        bdf = tb.BeliefsDataFrame(*args, **kwargs)
    assert bdf.event_starts[0] == datetime(2000, 1, 1, tzinfo=pytz.utc)
    assert (
        bdf.belief_times[0]
        == datetime(2000, 1, 1, tzinfo=pytz.utc) + bdf.event_resolution
    )
    assert bdf.belief_horizons[0] == timedelta()
    assert bdf.sources[0].name == "3"
    assert bdf.values[0] == 4


@pytest.mark.parametrize(
    "args, kwargs",
    [
        (
            [],
            {
                "beliefs": [
                    tb.TimedBelief(
                        tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                        tb.BeliefSource(3),
                        4,
                        event_start=datetime(2000, 1, 1, tzinfo=pytz.utc),
                        belief_horizon=timedelta(),
                    )
                ]
            },
        ),
        (
            [
                [
                    tb.TimedBelief(
                        tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                        tb.BeliefSource(3),
                        4,
                        event_start=datetime(2000, 1, 1, tzinfo=pytz.utc),
                        belief_horizon=timedelta(),
                    )
                ]
            ],
            {},
        ),
    ],
)
def test_belief_setup_with_timed_beliefs(args, kwargs):
    """Test different ways of setting up the same BeliefsDataFrame."""
    bdf = tb.BeliefsDataFrame(*args, **kwargs)
    assert bdf.event_starts[0] == datetime(2000, 1, 1, tzinfo=pytz.utc)
    assert (
        bdf.belief_times[0]
        == datetime(2000, 1, 1, tzinfo=pytz.utc) + bdf.event_resolution
    )
    assert bdf.belief_horizons[0] == timedelta()
    assert bdf.sources[0].name == "3"
    assert bdf.values[0] == 4
    tb.BeliefsDataFrame()


def test_converting_between_data_frame_and_series_retains_metadata():
    """
    Test whether slicing of a BeliefsDataFrame into a BeliefsSeries retains the metadata.
    Test whether expanding dimensions of a BeliefsSeries into a BeliefsDataFrame retains the metadata.
    """
    df = example_df
    metadata = {md: getattr(example_df, md) for md in METADATA}
    series = df["event_value"]
    for md in metadata:
        assert getattr(series, md) == metadata[md]
    df = series.to_frame()
    for md in metadata:
        assert getattr(df, md) == metadata[md]


def test_dropping_index_levels_retains_metadata():
    df = example_df.copy()
    metadata = {md: getattr(example_df, md) for md in METADATA}
    df.index = df.index.get_level_values("event_start")  # drop all other index levels
    for md in metadata:
        assert getattr(df, md) == metadata[md]
