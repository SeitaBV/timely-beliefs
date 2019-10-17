import pytest
import os
import pandas as pd
from datetime import datetime

from timely_beliefs.examples import example_df
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
    with pytest.warns(UserWarning, match="type other than BeliefSource"):
        df = tb.BeliefsDataFrame(df, sensor=tb.Sensor("Sensor Y"))
    assert df.sensor.name == "Sensor Y"

    # No lookup should issue warning
    with pytest.warns(UserWarning, match="type other than BeliefSource"):
        df = tb.read_csv("test.csv", sensor=tb.Sensor("Sensor Y"))
    assert all(
        c != tb.BeliefSource for c in df.index.get_level_values("source").map(type)
    )

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
