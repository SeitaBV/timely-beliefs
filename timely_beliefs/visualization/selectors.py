from typing import Optional

import altair as alt


idle_color = "lightgray"

time_selection_brush = alt.selection_interval(encodings=["x"], name="time_select")
horizon_hover_brush = alt.selection_single(
    on="mouseover", nearest=True, encodings=["x"], empty="all"
)
source_selection_brush = alt.selection_multi(fields=["source"], name="source_select")

# Create selection brushes that choose the nearest point & selects based on x-value
nearest_x_hover_brush = alt.selection_single(
    nearest=True, on="mouseover", encodings=["x"], empty="none", name="nearest_x_hover"
)
nearest_x_select_brush = alt.selection_single(
    nearest=True, encodings=["x"], empty="all", name="nearest_x_select"
)


def horizon_selection_brush(init_belief_horizon=None) -> alt.MultiSelection:
    """Create a brush for selecting one or multiple horizons.

    :param init_belief_horizon: Optional initialisation value
    """
    if init_belief_horizon is None:
        return alt.selection_multi(
            nearest=False, encodings=["x"], empty="all", name="horizon_select"
        )
    else:
        return alt.selection_multi(
            nearest=False,
            encodings=["x"],
            name="horizon_select",
            empty="all",
            init={"belief_horizon": init_belief_horizon},
        )


def fixed_viewpoint_selector(base, idle: bool = False) -> alt.LayerChart:
    """Transparent selectors across the chart. This is what tells us the x-value of the cursor."""
    selector = base.mark_rule().encode(
        x=alt.X("belief_time:T", scale={"domain": time_selection_brush.ref()}),
        color=alt.ColorValue(idle_color) if idle is True else alt.ColorValue("#c21431"),
        opacity=alt.condition(nearest_x_hover_brush, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip(
                "belief_time:T",
                timeUnit="yearmonthdatehoursminutes",
                title="Click to select belief time",
            )
        ]
        if idle is False
        else None,
    )
    return selector.add_selection(nearest_x_select_brush).add_selection(
        nearest_x_hover_brush
    )


def time_window_selector(base) -> alt.LayerChart:
    tws = (
        base.mark_bar()
        .encode(
            x=alt.X("event_start", title=""),
            x2=alt.X2("event_end:T"),
            y=alt.Y(
                "reference_value",
                title="",
                axis=alt.Axis(values=[], domain=False, ticks=False),
            ),
            color=alt.ColorValue(idle_color),
            tooltip=alt.TooltipValue("Click and drag to select time window"),
        )
        .properties(height=30, title="Select time window")
    )
    tws = tws.add_selection(time_selection_brush) + tws.transform_filter(
        time_selection_brush
    ).encode(
        color=alt.condition(
            time_selection_brush, alt.ColorValue("#c21431"), alt.ColorValue(idle_color)
        )
    )
    return tws


def horizon_selector(
    base,
    horizon_selection_brush: alt.MultiSelection,
    belief_horizon_unit: str,
    intuitive_forecast_horizon: bool,
    unique_belief_horizons,
) -> alt.LayerChart:
    bar_chart = (
        base.mark_rule(orient="vertical")
        # # Remove the most recent belief horizon from the selector
        # .transform_joinaggregate(most_recent_belief_horizon="min(belief_horizon)")
        # .transform_filter(
        #     alt.datum.belief_horizon > alt.datum.most_recent_belief_horizon
        # )
        .transform_filter(
            time_selection_brush
        )  # Apply brush before calculating accuracy metrics for the selected events on the fly
        .transform_calculate(constant=1 + alt.datum.event_start - alt.datum.event_start)
        .transform_calculate(
            belief_horizon_str='datum.belief_horizon + " %s"' % belief_horizon_unit
        )
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
            y=alt.Y(
                "constant:Q",
                title=" ",
                axis=alt.Axis(values=[], domain=False, ticks=False),
            ),
            color=alt.condition(
                horizon_selection_brush | horizon_hover_brush,
                alt.ColorValue("#c21431"),
                alt.ColorValue(idle_color),
            ),
            size=alt.value(1),
            tooltip=[
                alt.Tooltip(
                    "belief_horizon_str:N",
                    title="Click to select %s"
                    % ("horizon" if intuitive_forecast_horizon else "belief horizon"),
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


def source_color_or(alternative_color: Optional[str] = idle_color, brush=None):
    if alternative_color is None:
        alternative_color = ""
    if brush is None:
        brush = source_selection_brush
    return alt.condition(
        brush, alt.Color("source:N", legend=None), alt.value(alternative_color)
    )


def source_selector(source) -> alt.Chart:
    return (
        alt.Chart(source)
        .mark_square(size=50, opacity=0.3)
        .encode(
            y=alt.Y(
                "source:N",
                axis=alt.Axis(orient="right", domain=False, ticks=False),
                title=None,
            ),
            color=source_color_or(idle_color),
        )
        .add_selection(source_selection_brush)
        .properties(title=alt.TitleParams("Select source", anchor="start"))
    )
