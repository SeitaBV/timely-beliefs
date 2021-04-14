from typing import Tuple, Union

import altair as alt
import pandas as pd

from timely_beliefs.visualization import selectors


def time_series_base_chart(
    source: Union[pd.DataFrame, str],
    belief_horizon_unit: str,
    intuitive_forecast_horizon: bool,
) -> alt.Chart:
    return (
        alt.Chart(source)
        .encode(
            x=alt.X(
                "event_start",
                axis=alt.Axis(labelFlush=False),
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
                    title="%s (%s)"
                    % (
                        "Forecast horizon"
                        if intuitive_forecast_horizon
                        else "Belief horizon",
                        belief_horizon_unit,
                    ),
                ),
                alt.Tooltip("source", title="Source"),
            ],
        )
        .properties(width=550, height=200)
    )


def value_vs_time_chart(
    base: alt.Chart,
    active_fixed_viewpoint_selector: bool,
    sensor_name: str,
    sensor_unit: str,
    belief_horizon_unit: str,
    intuitive_forecast_horizon: bool,
    interpolate: bool,
    ci: float,
    event_value_range: Tuple[float, float],
) -> alt.LayerChart:

    # Configure the stepwise line for the reference
    if interpolate is True:
        ts_line_reference_chart = base.mark_line(interpolate="monotone")
    else:
        ts_line_reference_chart = base.mark_rule().encode(x2=alt.X2("event_end:T"))
    ts_line_reference_chart = ts_line_reference_chart.encode(
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
    if interpolate is True:
        ts_line_chart = base.mark_line(interpolate="monotone")
    else:
        ts_line_chart = base.mark_rule().encode(x2=alt.X2("event_end:T"))
    ts_line_chart = ts_line_chart.encode(
        y=alt.Y("expected_value", title="%s (%s)" % (sensor_name, sensor_unit))
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
    if interpolate is True:
        confidence_interval = ts_line_chart.mark_area(
            interpolate="monotone", opacity=0.3
        )
    else:
        confidence_interval = ts_line_chart.mark_bar(opacity=0.3)
    confidence_interval = confidence_interval.encode(
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
        title="Model results"
    )


def accuracy_vs_horizon_chart(
    base: alt.Chart,
    horizon_selection_brush: alt.Selection,
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
                axis=alt.Axis(labelFlush=False),
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
                    title="%s (%s)"
                    % (
                        "Forecast horizon"
                        if intuitive_forecast_horizon
                        else "Belief horizon",
                        belief_horizon_unit,
                    ),
                ),
                alt.Tooltip("source:O", title="Source"),
            ],
        )
        .properties(title="Model accuracy in the %s before" % belief_horizon_unit)
        # .configure_title(offset=-270, orient="bottom")
    )
    ha_interpolation_chart = ha_chart.mark_line(interpolate="monotone").encode(
        size=alt.value(1), color=alt.Color("source:N", legend=None)
    )
    return ha_interpolation_chart + ha_chart


def source_vs_hour_chart(
    base: alt.Chart, sensor_unit: str, max_absolute_error: float, faceted: bool = False
) -> Union[alt.Chart, alt.FacetChart]:
    hd_chart = (
        base.mark_rect()
        .transform_joinaggregate(
            on_the_fly_mae="mean(mae)",
            on_the_fly_reference="mean(reference_value)",
            groupby=["event_start", "source"],
        )
        .transform_calculate(accuracy=alt.datum.on_the_fly_mae)
        .encode(
            x=alt.X(
                "event_start:O",
                timeUnit="hours",
                axis=alt.Axis(domain=False, ticks=False, labelAngle=0),
                # scale=alt.Scale(domain=list(range(24))),
                title="Hour of day",  # "UTC hour of day"
            ),
            color=alt.condition(
                selectors.time_selection_brush,
                alt.Color(
                    "accuracy:Q",
                    scale=alt.Scale(
                        domain=(max_absolute_error, 0), scheme="redyellowgreen"
                    ),
                    title="Error",
                ),
                alt.value(selectors.idle_color),
            ),
            tooltip=[
                alt.Tooltip("event_start:T", timeUnit="hours", title="Hour of day"),
                alt.Tooltip(
                    "accuracy:Q",
                    title="Mean absolute error (%s)" % sensor_unit,
                    format=".2f",
                ),
            ],
        )
    )
    if faceted:
        hd_chart = hd_chart.facet(
            row=alt.Row("source:O", title=None, header=alt.Header(labelAngle=0))
        )
    else:
        hd_chart = hd_chart.encode(
            y=alt.Y(
                "source:O",
                axis=alt.Axis(domain=False, ticks=False, labelAngle=0, labelPadding=5),
                title=None,
            )
        )
    return hd_chart.properties(
        title=alt.TitleParams("Model performance given a time of day", anchor="middle")
    )


def deterministic_chart(probability_scale_range: Tuple[float, float]):
    return (
        alt.Chart()
        .mark_circle(color="red")
        .transform_calculate(zero=alt.datum.probability * 0)
        .encode(
            x=alt.X(
                "event_value:Q",
                aggregate={"argmax": "probability"},
                bin="binned",
                scale=alt.Scale(padding=0),
            ),
            y=alt.Y(
                "zero:Q", scale=alt.Scale(range=probability_scale_range), axis=None
            ),
        )
    )


def probabilistic_chart(
    probability_scale_range: Tuple[float, float],
    belief_horizon_unit: str,
    sensor_name: str,
    sensor_unit: str,
):
    base_chart = alt.Chart().encode(
        x=alt.X(
            "event_value:Q",
            bin="binned",
            scale=alt.Scale(padding=0),
            title=sensor_name + " (" + sensor_unit + ")",
        ),
        y=alt.Y(
            "probability:Q", scale=alt.Scale(range=probability_scale_range), axis=None
        ),
    )
    line_chart = base_chart.mark_line(interpolate="monotone").encode(
        stroke=alt.condition(
            selectors.ridgeline_hover_brush, alt.value("black"), alt.value("lightgray")
        ),
        strokeWidth=alt.StrokeWidthValue(
            0.5,
            condition=alt.StrokeWidthValue(
                2.5, selection=selectors.ridgeline_hover_brush.name
            ),
        ),
    )
    area_chart = base_chart.mark_area(interpolate="monotone", fillOpacity=0.6).encode(
        fill=alt.Fill(
            "belief_horizon:N",
            sort="ascending",
            legend=None,
            scale=alt.Scale(scheme="viridis"),
        ),
        tooltip=[
            alt.Tooltip("event_value:Q", title="Value", format=".2f"),
            alt.Tooltip("probability:Q", title="Probability", format=".2f"),
            alt.Tooltip(
                "belief_horizon:Q",
                title="%s (%s)" % ("Belief horizon", belief_horizon_unit),
            ),
        ],
    )
    return alt.layer(area_chart, line_chart)
