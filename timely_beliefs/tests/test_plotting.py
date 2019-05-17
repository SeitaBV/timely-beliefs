import pytest

from timely_beliefs.examples import example_df


def test_chart_creation():
    """Create a chart JSON object with vega-lite specifications."""
    with pytest.raises(ValueError) as v:
        example_df.plot()
    assert "Must set reference source." in str(v.value)

    chart = example_df.plot(
        show_accuracy=False, reference_source=example_df.lineage.sources[0]
    )
    chart.to_dict()  # this should not fail
    chart = example_df.plot(
        show_accuracy=True, reference_source=example_df.lineage.sources[0]
    )
    chart.to_dict()
