from typing import Tuple, Union

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
                    "belief_time:T",
                    timeUnit="yearmonthdatehoursminutes",
                    title="Belief time",
                ),
                alt.Tooltip(
                    "belief_horizon:Q",
                    title="Belief horizon (%s)" % belief_horizon_unit,
                ),
                alt.Tooltip("source", title="Source"),
            ],
        )
        .properties(width=550, height=200)
    )


def time_series_chart(
    base,
    active_fixed_viewpoint_selector: bool,
    sensor_name: str,
    sensor_unit: str,
    belief_horizon_unit: str,
    intuitive_forecast_horizon: bool,
    ci: float,
    event_value_range: Tuple[float, float],
) -> alt.LayerChart:

    # Configure the stepwise line for the reference
    ts_line_reference_chart = base.mark_rule().encode(
        x2=alt.X2("event_end:T"),
        y=alt.Y(
            "reference_value",
            scale=alt.Scale(domain=(event_value_range[0], event_value_range[-1])),
        ),
        color=alt.ColorValue("black"),
        tooltip=[
            alt.Tooltip(
                "event_start:T",
                timeUnit="yearmonthdatehoursminutes",
                title="Event start",
            ),
            alt.Tooltip(
                "event_end:T", timeUnit="yearmonthdatehoursminutes", title="Event end"
            ),
            alt.Tooltip("reference_value:Q", title="Real value", format=".2f"),
        ],
    )

    # Configure the stepwise line for the beliefs
    ts_line_chart = base.mark_rule().encode(
        x2=alt.X2("event_end:T"),
        y=alt.Y("expected_value", title="%s (%s)" % (sensor_name, sensor_unit)),
    )

    if active_fixed_viewpoint_selector is True:
        ts_line_chart = (
            ts_line_chart.transform_filter(
                "datum.belief_time <= nearest_x_select.belief_time"
            )
            .transform_joinaggregate(
                most_recent_belief_time="max(belief_time)",
                groupby=["event_start", "source"],
            )
            .transform_filter("datum.belief_time == datum.most_recent_belief_time")
        )

    # Configure the confidence intervals
    confidence_interval = ts_line_chart.mark_bar(opacity=0.3).encode(
        y="lower_value",
        y2="upper_value",
        tooltip=[
            alt.Tooltip(
                "event_start:T",
                timeUnit="yearmonthdatehoursminutes",
                title="Event start",
            ),
            alt.Tooltip(
                "event_end:T", timeUnit="yearmonthdatehoursminutes", title="Event end"
            ),
            alt.Tooltip("expected_value:Q", title="Expected value", format=".2f"),
            alt.Tooltip(
                "belief_time:T",
                timeUnit="yearmonthdatehoursminutes",
                title="Belief time",
            ),
            alt.Tooltip(
                "belief_horizon:Q",
                title="%s (%s)"
                % (
                    "Forecast horizon"
                    if intuitive_forecast_horizon
                    else "Belief horizon",
                    belief_horizon_unit,
                ),
            ),
            alt.Tooltip("source", title="Source"),
            alt.Tooltip(
                "upper_value:Q",
                format=".2f",
                title="Upper value of {0:.0f}% confidence interval".format(100 * ci),
            ),
            alt.Tooltip(
                "lower_value:Q",
                format=".2f",
                title="Lower value of {0:.0f}% confidence interval".format(100 * ci),
            ),
        ],
    )

    return (ts_line_reference_chart + ts_line_chart + confidence_interval).properties(
        title="Beliefs over time"
    )


def horizon_accuracy_chart(
    base,
    horizon_selection_brush,
    belief_horizon_unit: str,
    intuitive_forecast_horizon: bool,
    unique_belief_horizons,
) -> alt.LayerChart:
    ha_chart = (
        base.mark_circle()
        .transform_joinaggregate(
            on_the_fly_mae="mean(mae)",
            on_the_fly_reference="mean(reference_value)",
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
                title="%s (%s)"
                % (
                    "Forecast horizon"
                    if intuitive_forecast_horizon
                    else "Belief horizon",
                    belief_horizon_unit,
                ),
            ),
            y=alt.Y(
                "accuracy:Q",
                title="Accuracy (1-WAPE)",
                axis=alt.Axis(format="%", minExtent=30),
                scale={"domain": (0, 1)},
            ),
            color=selectors.source_color_or(
                selectors.idle_color,
                selectors.horizon_hover_brush | horizon_selection_brush,
            ),
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
        size=alt.value(1), color=alt.Color("source:N", legend=None)
    )
    return ha_interpolation_chart + ha_chart


def hour_date_chart(base, faceted: bool = False) -> Union[alt.Chart, alt.FacetChart]:
    hd_chart = (
        base.mark_rect()
        .transform_joinaggregate(
            on_the_fly_mae="mean(mae)",
            on_the_fly_reference="mean(reference_value)",
            groupby=["event_start", "source"],
        )
        .transform_calculate(
            on_the_fly_wape=alt.datum.on_the_fly_mae / alt.datum.on_the_fly_reference,
            accuracy=alt.expr.if_(
                alt.datum.on_the_fly_wape > 1, "inf", 1 - alt.datum.on_the_fly_wape
            ),  # draw white if off the chart
        )
        .encode(
            x=alt.X(
                "event_start:O",
                timeUnit="hours",
                axis=alt.Axis(domain=False, ticks=False, labelAngle=0),
                scale=alt.Scale(domain=list(range(24))),
                title="Hour of day",  # "UTC hour of day"
            ),
            color=alt.condition(
                selectors.time_selection_brush,
                alt.Color(
                    "accuracy:Q",
                    scale=alt.Scale(zero=True, domain=(0, 1), scheme="rainbow"),
                    title="Accuracy",  # "Accuracy (1-WAPE)",
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
    )
    if faceted:
        hd_chart = hd_chart.facet(
            row=alt.Row("source:O", title=None, header=alt.Header(labelAngle=0))
        )
    else:
        hd_chart = hd_chart.encode(
            y=alt.Y(
                "source:O",
                axis=alt.Axis(domain=False, ticks=False, labelAngle=0),
                title=None,
            )
        )
    return hd_chart.properties(
        title=alt.TitleParams("Accuracy given a time of day", anchor="middle")
    )
