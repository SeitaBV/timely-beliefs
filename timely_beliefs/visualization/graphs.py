import altair as alt

from timely_beliefs.visualization import selectors


def base_chart(source, belief_horizon_unit: str) -> alt.Chart:
    return (
        alt.Chart(source)
        .encode(
            x=alt.X(
                "event_start",
                scale={"domain": selectors.time_selection_brush.ref()},
                title="Event start",
            ),
            y=alt.Y("expected_value", title="Event value"),
            color=selectors.source_color_or(selectors.idle_color),
            tooltip=[
                alt.Tooltip(
                    "event_start:T",
                    timeUnit="yearmonthdatehoursminutes",
                    title="Event start",
                ),
                alt.Tooltip(
                    "event_end:T",
                    timeUnit="yearmonthdatehoursminutes",
                    title="Event end",
                ),
                alt.Tooltip("expected_value:Q", title="Expected value", format=".2f"),
                alt.Tooltip(
                    "belief_horizon:Q",
                    title="Belief horizon (%s)" % belief_horizon_unit,
                ),
                alt.Tooltip("source", title="Source"),
            ],
        )
        .properties(width=600, height=200)
    )


def time_series_chart(
    base, show_accuracy, belief_horizon_unit, ci: float
) -> alt.LayerChart:

    # Configure the stepwise line
    ts_line_chart = (
        base.mark_rule(interpolate="step-before")
        .transform_filter(selectors.source_selection_brush)
        .encode(x2=alt.X2("event_end:T"))
    )
    if show_accuracy is True:
        ts_line_chart = ts_line_chart.transform_filter(
            selectors.horizon_hover_brush | selectors.horizon_selection_brush
        )

    # Configure the confidence intervals
    confidence_interval = (
        base.mark_bar(opacity=0.3)
        .transform_filter(selectors.source_selection_brush)
        .encode(
            x=alt.X(
                "event_start", scale={"domain": selectors.time_selection_brush.ref()}
            ),
            x2=alt.X2("event_end:T"),
            y="lower_value",
            y2="upper_value",
            color=selectors.source_color_or(
                None, selectors.horizon_selection_brush | selectors.horizon_hover_brush
            ),
            tooltip=[
                alt.Tooltip(
                    "event_start:T",
                    timeUnit="yearmonthdatehoursminutes",
                    title="Event start",
                ),
                alt.Tooltip(
                    "event_end:T",
                    timeUnit="yearmonthdatehoursminutes",
                    title="Event end",
                ),
                alt.Tooltip("expected_value:Q", title="Expected value", format=".2f"),
                alt.Tooltip(
                    "belief_horizon:Q",
                    title="Belief horizon (%s)" % belief_horizon_unit,
                ),
                alt.Tooltip("source", title="Source"),
                alt.Tooltip(
                    "upper_value:Q",
                    format=".2f",
                    title="Upper value of {0:.0f}% confidence interval".format(
                        100 * ci
                    ),
                ),
                alt.Tooltip(
                    "lower_value:Q",
                    format=".2f",
                    title="Lower value of {0:.0f}% confidence interval".format(
                        100 * ci
                    ),
                ),
            ],
        )
    )

    return (ts_line_chart + confidence_interval).properties(title="Beliefs over time")


def horizon_accuracy_chart(
    base, belief_horizon_unit, unique_belief_horizons
) -> alt.LayerChart:
    ha_chart = (
        base.mark_circle()
        .transform_filter(
            selectors.time_selection_brush
        )  # Apply brush before calculating accuracy metrics for the selected events on the fly
        .transform_filter(selectors.source_selection_brush)
        .transform_window(
            on_the_fly_mae="mean(mae)",
            on_the_fly_reference="mean(reference_value)",
            frame=[None, None],
            groupby=["belief_horizon", "source"],
        )
        .transform_calculate(
            on_the_fly_wape=alt.datum.on_the_fly_mae / alt.datum.on_the_fly_reference,
            accuracy=1 - alt.datum.on_the_fly_wape,
        )
        .encode(
            opacity=alt.condition(
                selectors.time_selection_brush,
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
                title="Belief horizon (%s)" % belief_horizon_unit,
            ),
            y=alt.Y(
                "accuracy:Q",
                title="Accuracy (1-WAPE)",
                axis=alt.Axis(format="%", minExtent=30),
                scale={"domain": (0, 1)},
            ),
            color=selectors.source_color_or(
                selectors.idle_color,
                brush=selectors.horizon_selection_brush | selectors.horizon_hover_brush,
            ),
            # size=alt.Size(
            #     "belief_horizon:Q",
            #     scale=alt.Scale(zero=False),
            #     title="Belief horizon (%s)" % belief_horizon_unit,
            #     sort="ascending",
            #     legend=alt.Legend(format=".0f"),  # , orient="right"),
            # ),
            size=alt.value(100),
            tooltip=[
                alt.Tooltip("accuracy:Q", title="Accuracy (1-WAPE)", format=".1%"),
                alt.Tooltip(
                    "belief_horizon:Q",
                    title="Belief horizon (%s)" % belief_horizon_unit,
                ),
                alt.Tooltip("source:O", title="Source"),
            ],
        )
        .properties(title="Accuracy in the %s before" % belief_horizon_unit)
        # .configure_title(offset=-270, orient="bottom")
    )
    ha_interpolation_chart = ha_chart.mark_line(interpolate="monotone").encode(
        size=alt.value(1)
    )
    ha_annotation_chart = ha_interpolation_chart.mark_bar().transform_filter(
        selectors.horizon_hover_brush | selectors.horizon_selection_brush
    )
    return ha_annotation_chart + ha_interpolation_chart + ha_chart


def hour_date_chart(base) -> alt.FacetChart:
    return (
        base.mark_rect()
        .transform_filter(
            selectors.time_selection_brush
        )  # Apply brushes before calculating accuracy metrics for the selected events on the fly
        .transform_filter(selectors.source_selection_brush)
        .transform_filter(
            selectors.horizon_hover_brush | selectors.horizon_selection_brush
        )
        .transform_window(
            on_the_fly_mae="mean(mae)",
            on_the_fly_reference="mean(reference_value)",
            frame=[None, None],
            groupby=["event_start", "source"],
        )
        .transform_calculate(
            on_the_fly_wape=alt.datum.on_the_fly_mae / alt.datum.on_the_fly_reference,
            accuracy=1 - alt.datum.on_the_fly_wape,
        )
        .encode(
            x=alt.X(
                "event_start:O",
                timeUnit="monthdate",
                title="UTC date",
                axis=alt.Axis(domain=False, ticks=False),
            ),
            y=alt.Y(
                "event_start:O",
                timeUnit="hours",
                axis=alt.Axis(domain=False, ticks=False),
                scale=alt.Scale(domain=list(range(24))),
                title="UTC hour of day",
            ),
            color=alt.condition(
                selectors.time_selection_brush,
                alt.Color(
                    "accuracy:Q",
                    scale=alt.Scale(zero=True, domain=(0, 1), scheme="redyellowgreen"),
                    title="Accuracy (1-WAPE)",
                    legend=alt.Legend(format=".0%"),
                ),
                alt.value(selectors.idle_color),
            ),
            tooltip=[
                alt.Tooltip("event_start:T", timeUnit="hours", title="Hour of day"),
                alt.Tooltip("accuracy:Q", title="Accuracy (1-WAPE)", format=".1%"),
            ],
        )
        # .properties(height=300, width=300)
        .facet(column=alt.Column("source:O", title=None))
        .properties(
            title=alt.TitleParams("Accuracy given a time of day", anchor="middle")
        )
    )
