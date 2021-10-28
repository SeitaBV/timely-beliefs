import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import pytz

import timely_beliefs as tb
from timely_beliefs.examples import get_example_df
from timely_beliefs.tests.utils import assert_metadata_is_retained


@pytest.fixture(scope="module")
def csv_file(tmpdir_factory):
    """Save BeliefsDataFrame to csv."""

    get_example_df().to_csv("test.csv")
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
    df_copy = df.copy()
    with pytest.warns(UserWarning, match="created"):
        bdf = tb.BeliefsDataFrame(df, sensor=tb.Sensor("Sensor Y"))
    assert bdf.sensor.name == "Sensor Y"

    # Check that input frame was not altered
    # GH 34
    pd.testing.assert_frame_equal(df, df_copy)

    # No lookup should issue warning
    with pytest.warns(UserWarning, match="looking them up"):
        bdf = tb.read_csv("test.csv", sensor=tb.Sensor("Sensor Y"))
    for s in bdf.index.get_level_values("source"):
        assert isinstance(
            s, tb.BeliefSource
        )  # Source names automatically get converted to sources
    assert all(c == tb.BeliefSource for c in bdf.sources.map(type))

    # This lookup should fail
    with pytest.raises(ValueError, match="not in list"):
        tb.read_csv(
            "test.csv",
            sensor=tb.Sensor("Sensor Y"),
            look_up_sources=[tb.BeliefSource(name="Source X")],
        )

    # This lookup should succeed
    source_a, source_b = tb.BeliefSource("Source A"), tb.BeliefSource("Source B")
    bdf = tb.read_csv(
        "test.csv", sensor=tb.Sensor("Sensor Y"), look_up_sources=[source_a, source_b]
    )
    assert bdf.sensor.name == "Sensor Y"
    assert source_a in bdf.index.get_level_values("source")
    assert source_b in bdf.index.get_level_values("source")
    assert isinstance(bdf.index.get_level_values("event_start")[0], datetime)
    assert isinstance(bdf.index.get_level_values("belief_time")[0], datetime)


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

    # Check that initializing with self returns a copy of self
    # GH 34
    bdf_copy = bdf.copy()
    bdf = tb.BeliefsDataFrame(bdf)
    pd.testing.assert_frame_equal(bdf_copy, bdf)


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
    "df_or_s, kwargs",
    [
        (
            pd.DataFrame(
                [[datetime(2000, 1, 1, tzinfo=pytz.utc), timedelta(), 3, 4]],
                columns=["event_start", "belief_horizon", "source", "event_value"],
            ),
            {"sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1))},
        ),
        (
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
            ),
            {"sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1))},
        ),
        (
            pd.DataFrame(
                [[datetime(2000, 1, 1, tzinfo=pytz.utc), timedelta(), 4]],
                columns=["event_start", "belief_horizon", "event_value"],
            ),
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
            },
        ),  # move source to keyword argument
        (
            pd.DataFrame(
                [[datetime(2000, 1, 1, tzinfo=pytz.utc), 4]],
                columns=["event_start", "event_value"],
            ),
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
                "belief_horizon": timedelta(),
            },
        ),  # move source and belief_horizon to keyword argument
        (
            pd.DataFrame([[4]], columns=["event_value"]),
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
                "belief_horizon": timedelta(),
                "event_start": datetime(2000, 1, 1, tzinfo=pytz.utc),
            },
        ),  # move source, belief_horizon and event_start to keyword argument
        (
            pd.Series([4]),
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
                "belief_horizon": timedelta(),
                "event_start": datetime(2000, 1, 1, tzinfo=pytz.utc),
            },
        ),  # move source, belief_horizon and event_start to keyword argument and use Series instead of DataFrame
        (
            pd.Series(
                [4], index=pd.DatetimeIndex([datetime(2000, 1, 1, tzinfo=pytz.utc)])
            ),
            {
                "sensor": tb.Sensor(name="temp", event_resolution=timedelta(hours=1)),
                "source": 3,
                "belief_horizon": timedelta(),
            },
        ),  # move source and belief_horizon keyword argument and use Series instead of DataFrame
    ],
)
def test_belief_setup_with_data_frame(df_or_s, kwargs):
    """Test different ways of setting up the same BeliefsDataFrame."""
    df_or_s_copy = df_or_s.copy()

    with pytest.warns(UserWarning, match="created"):
        bdf = tb.BeliefsDataFrame(df_or_s, **kwargs)
    assert bdf.event_starts[0] == datetime(2000, 1, 1, tzinfo=pytz.utc)
    assert (
        bdf.belief_times[0]
        == datetime(2000, 1, 1, tzinfo=pytz.utc) + bdf.event_resolution
    )
    assert bdf.belief_horizons[0] == timedelta()
    assert bdf.sources[0].name == "3"
    assert bdf.values[0] == 4

    # Check that input data frame or series was not altered
    # GH 34
    if isinstance(df_or_s, pd.DataFrame):
        pd.testing.assert_frame_equal(df_or_s, df_or_s_copy)
    elif isinstance(df_or_s, pd.Series):
        pd.testing.assert_series_equal(df_or_s, df_or_s_copy)

    # Check that initializing with self returns a copy of self
    # GH 34
    bdf_copy = bdf.copy()
    bdf = tb.BeliefsDataFrame(bdf)
    pd.testing.assert_frame_equal(bdf_copy, bdf)


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
    example_df = get_example_df()
    df = example_df
    series = df["event_value"]
    assert_metadata_is_retained(series, original_df=example_df, is_series=True)
    df = series.to_frame()
    assert_metadata_is_retained(df, original_df=example_df)


def test_dropping_index_levels_retains_metadata():
    example_df = get_example_df()
    df = example_df.copy()
    df.index = df.index.get_level_values("event_start")  # drop all other index levels
    assert_metadata_is_retained(df, original_df=example_df)


@pytest.mark.parametrize("drop_level", [True, False])
def test_slicing_retains_metadata(drop_level):
    """
    Test whether slicing the index of a BeliefsDataFrame retains the metadata.
    """
    example_df = get_example_df()
    df = example_df
    df = df.xs("2000-01-03 10:00:00+00:00", level="event_start", drop_level=drop_level)
    print(df)
    assert_metadata_is_retained(df, original_df=example_df)


@pytest.mark.parametrize("resolution", [timedelta(minutes=30), timedelta(hours=2)])
def test_mean_resampling_retains_metadata(resolution):
    """
    Test whether mean resampling retains the metadata.

    Fails with pandas==1.0.0
    Succeeds with pandas==1.1.0
    """
    example_df = get_example_df()
    df = example_df
    df = df.resample(resolution, level="event_start").mean()
    print(df)
    assert_metadata_is_retained(
        df,
        original_df=example_df,
        event_resolution=example_df.event_resolution,
    )  # todo: the event_resolution metadata is only updated when resampling using df.resample_events(). A reason to override the original resample method, or otherwise something to document.


@pytest.mark.parametrize("resolution", [timedelta(minutes=30), timedelta(hours=2)])
def _test_agg_resampling_retains_metadata(resolution):
    """
    Test whether aggregate resampling retains the metadata.

    Fails with pandas==1.1.5
    """
    example_df = get_example_df()
    df = example_df
    df = df.reset_index(level=["belief_time", "source", "cumulative_probability"])
    df = df.resample(resolution).agg(
        {
            "event_value": np.nanmean,
            "source": "first",  # keep the only source
            "belief_time": "max",  # keep the latest belief
            "cumulative_probability": "prod",  # assume independent variables
        }
    )
    df = df.set_index(["belief_time", "source", "cumulative_probability"], append=True)
    print(df)
    assert_metadata_is_retained(
        df,
        original_df=example_df,
        event_resolution=example_df.event_resolution,
    )  # todo: the event_resolution metadata is only updated when resampling using df.resample_events(). A reason to override the original resample method, or otherwise something to document.


def test_groupby_retains_metadata():
    """Test whether grouping by index level retains the metadata.

    Succeeds with pandas==1.0.0
    Fails with pandas==1.1.0
    Fixed with pandas==1.1.5
    Fails with pandas==1.3.0
    """
    example_df = get_example_df()
    df = example_df

    def assert_function(x):
        print(x)
        assert_metadata_is_retained(x, original_df=example_df)
        return x

    df = df.groupby(level="event_start").apply(lambda x: assert_function(x))
    assert_metadata_is_retained(df, original_df=example_df)


def test_copy_series_retains_name_and_metadata():
    # GH 41
    df = get_example_df()
    sensor = df.sensor
    s = df["event_value"]
    assert s.sensor == sensor
    name = s.name
    s_copy = s.copy()
    assert s_copy.name == name
    assert s_copy.sensor == sensor


def test_init_from_beliefs_data_frame():
    """Check that input BeliefsDataFrame was not altered."""
    # GH 34
    df = get_example_df().rename(columns={"event_value": "reference_value"})
    df_copy = df.copy()
    tb.BeliefsDataFrame(df)
    pd.testing.assert_frame_equal(df, df_copy)


def test_init_from_beliefs_series():
    """Check that input BeliefsSeries was not altered."""
    # GH 34
    df = get_example_df().rename(columns={"event_value": "reference_value"})
    s = df["reference_value"]
    df_copy = df.copy()
    s_copy = s.copy()

    # check method using to_frame
    bdf = s.to_frame()
    pd.testing.assert_frame_equal(df, df_copy)  # original bdf was not altered
    pd.testing.assert_frame_equal(
        bdf, df_copy
    )  # new bdf retains altered column of original bdf
    pd.testing.assert_series_equal(s, s_copy)  # input BeliefsSeries was not altered

    # check method using class init
    bdf = tb.BeliefsDataFrame(s)
    pd.testing.assert_frame_equal(df, df_copy)  # original bdf was not altered
    pd.testing.assert_frame_equal(
        bdf, df_copy
    )  # new bdf retains altered column of original bdf
    pd.testing.assert_series_equal(s, s_copy)  # input BeliefsSeries was not altered


def test_groupby_does_not_retain_temporary_attribute():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "y"])
    df.a = "b"
    assert df.a == "b"
    df2 = df.groupby("x").apply(lambda x: x)
    assert not hasattr(df2, "a")
    df3 = df.groupby("x").sum()
    assert not hasattr(df3, "a")


@pytest.mark.parametrize(
    "att, args",
    [
        # ("all", []),
        # ("any", []),
        # ("count", []),
        ("first", []),
        ("last", []),
        ("max", []),
        # ("mean", []),
        # ("median", []),
        ("min", []),
        ("prod", []),
        # ("sem", []),
        # ("size", []),
        # ("std", []),
        ("sum", []),
        # ("var", []),
        # ("apply", [lambda x: x]),
        # ("apply", [np.max])
        # ("apply", [np.min])
        # ("apply", [np.nanmean]),
        ("agg", ["first"]),
        ("agg", ["max"]),
        # ("agg", ["mean"]),
        ("agg", ["min"]),
        ("agg", ["sum"]),
        # ("agg", [{"y": "min"}]),
        # ("agg", [{"x": "min", "y": "max"}]),
    ],
)
def test_groupby_retains_subclass_attribute(att, args):
    """Checks on metadata propagation for subclassed DataFrames under groupby operations.

    Commented-out parameter combinations fail with pandas==1.1.5
    The relevant issue has to do with calling finalize after operations:
    see https://github.com/pandas-dev/pandas/issues/28283
    """

    METADATA = ["a"]

    class SubclassedSeries(pd.Series):

        _metadata = METADATA

        @property
        def _constructor(self):
            return SubclassedSeries

        @property
        def _constructor_expanddim(self):
            return SubclassedDataFrame

    class SubclassedDataFrame(pd.DataFrame):

        _metadata = METADATA

        @property
        def _constructor(self):
            return SubclassedDataFrame

        @property
        def _constructor_sliced(self):
            return SubclassedSeries

    df = SubclassedDataFrame([[1, 2], [3, 4]], columns=["x", "y"])
    df.a = "b"
    assert df.a == "b"
    df2 = getattr(df.groupby("x"), att)(*args)
    print(df2)
    assert df2.a == "b"


@pytest.mark.parametrize("constant", [1, -1, 3.14, timedelta(hours=1), ["TiledString"]])
def test_multiplication_with_constant_retains_metadata(constant):
    """Check whether the metadata is still there after multiplication."""
    # GH 35
    example_df = get_example_df()
    df = example_df * constant
    assert_metadata_is_retained(df, original_df=example_df)

    # Also check suggested workarounds from GH 35
    if constant == -1:
        df = -example_df
        assert_metadata_is_retained(df, original_df=example_df)

        df = example_df.abs()
        assert_metadata_is_retained(df, original_df=example_df)
