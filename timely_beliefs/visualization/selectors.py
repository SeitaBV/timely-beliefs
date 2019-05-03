import altair as alt


time_selection_brush = alt.selection_interval(encodings=["x"])
horizon_selection_brush = alt.selection_multi(
    nearest=False, encodings=["x"]
)  # , empty="none", init=unique_belief_horizons[-1])  # Todo: set initial value once altair supports the init property: https://vega.github.io/vega-lite/docs/selection.html
horizon_hover_brush = alt.selection_single(
    on="mouseover", nearest=False, encodings=["x"]
)
source_selection_brush = alt.selection_multi(fields=["source"])


def time_window_selector(base) -> alt.LayerChart:
    tws = (
        base.mark_bar()
        .encode(
            x=alt.X("event_start", bin=alt.Bin(maxbins=1000), title=""),
            x2=alt.X2("event_end:T"),
            y=alt.Y("expected_value", title="", axis=alt.Axis(values=[])),
            color=alt.ColorValue("lightgray"),
            # tooltip={},
        )
        .properties(height=30, title="Select time window")
    )
    tws = tws.add_selection(time_selection_brush) + tws.transform_filter(
        time_selection_brush
    ).encode(
        color=alt.condition(
            time_selection_brush, alt.ColorValue("#c21431"), alt.ColorValue("lightgray")
        )
    )
    return tws


def horizon_selector(
    base, belief_horizon_unit: str, unique_belief_horizons
) -> alt.LayerChart:
    bar_chart = (
        base.mark_bar(orient="vertical")
        .transform_filter(
            time_selection_brush
        )  # Apply brush before calculating accuracy metrics for the selected events on the fly
        .transform_calculate(constant=1 + alt.datum.event_start - alt.datum.event_start)
        .encode(
            opacity=alt.condition(
                time_selection_brush,
                alt.Opacity(
                    "event_start:T", scale=alt.Scale(domain=(0.9999, 1)), legend=None
                ),
                alt.value(0),
            ),
            # Trick to be able to apply the selection filter for event_start (event_start must be a field in one of the encoding channels)
            x=alt.X(
                "belief_horizon:Q",
                scale=alt.Scale(
                    zero=False,
                    domain=(unique_belief_horizons[0], unique_belief_horizons[-1]),
                ),
                title="",
            ),
            y=alt.Y("constant:Q", title=" ", axis=alt.Axis(values=[])),
            color=alt.condition(
                horizon_selection_brush or horizon_hover_brush,
                alt.ColorValue("#c21431"),
                alt.ColorValue("lightgray"),
            ),
            size=alt.condition(
                horizon_selection_brush or horizon_hover_brush,
                alt.value(1),
                alt.value(1),
            ),
            tooltip=[
                alt.Tooltip(
                    "belief_horizon:Q",
                    title="Belief horizon (%s)" % belief_horizon_unit,
                )
            ],
        )
        .properties(height=30, title="Select horizon")
        .transform_filter(time_selection_brush)
    )
    circle_chart = (
        bar_chart.mark_circle()
        .transform_calculate(half_constant=alt.datum.constant / 2)
        .encode(
            y=alt.Y("half_constant:Q", title="", axis=alt.Axis(values=[])),
            size=alt.value(100),
        )
    )
    return (
        bar_chart.add_selection(horizon_selection_brush, horizon_hover_brush)
        + circle_chart
    )


def source_color_or(alternative_color: str = "lightgray"):
    return alt.condition(
        source_selection_brush,
        alt.Color("source:N", legend=None),
        alt.value(alternative_color),
    )


def source_selector(source) -> alt.Chart:
    return (
        alt.Chart(source)
        .mark_square(size=50, opacity=0.3)
        .encode(
            y=alt.Y("source:N", axis=alt.Axis(orient="right"), title=None),
            color=source_color_or("lightgray"),
        )
        .add_selection(source_selection_brush)
        .properties(title=alt.TitleParams("Select source", anchor="start"))
    )
