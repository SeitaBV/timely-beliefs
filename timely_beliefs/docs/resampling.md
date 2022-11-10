# Resampling

BeliefsDataFrames come with a custom resample method `.resample_events()` to infer new beliefs about underlying events over time (upsampling) or aggregated events over time (downsampling).

Resampling a BeliefsDataFrame can be an expensive operation, especially when the frame contains beliefs from multiple sources and/or probabilistic beliefs.

## Table of contents

1. [Upsampling](#upsampling)
1. [Downsampling](#downsampling)

## Upsampling

Upsample to events with a resolution of 5 minutes:

    >>> from datetime import timedelta
    >>> import timely_beliefs as tb
    >>> df = tb.examples.get_example_df()
    >>> df5m = df.resample_events(timedelta(minutes=5))
    >>> df5m.sort_index(level=["belief_time", "source"]).head(9)
                                                                                         event_value
    event_start               belief_time               source   cumulative_probability
    2000-01-03 09:00:00+00:00 2000-01-01 00:00:00+00:00 Source A 0.1587                         90.0
                                                                 0.5000                        100.0
                                                                 0.8413                        110.0
    2000-01-03 09:05:00+00:00 2000-01-01 00:00:00+00:00 Source A 0.1587                         90.0
                                                                 0.5000                        100.0
                                                                 0.8413                        110.0
    2000-01-03 09:10:00+00:00 2000-01-01 00:00:00+00:00 Source A 0.1587                         90.0
                                                                 0.5000                        100.0
                                                                 0.8413                        110.0
    sensor: <Sensor: weight>, event_resolution: 0:05:00

When resampling, the event resolution of the underlying sensor remains the same (it's still a fixed property of the sensor):

    >>> df.sensor.event_resolution == df5m.sensor.event_resolution
    True

However, the event resolution of the BeliefsDataFrame is updated, as well as knowledge horizons and knowledge times: 

    >>> df5m.event_resolution
    datetime.timedelta(seconds=300)
    >>> -df.knowledge_horizons[0]  # note negative horizons denote "after the fact", and the original resolution was 15 minutes
    Timedelta('0 days 00:15:00')
    >>> -df5m.knowledge_horizons[0]
    Timedelta('0 days 00:05:00')

## Downsampling

Downsample to events with a resolution of 2 hours:

    >>> df2h = df.resample_events(timedelta(hours=2))
    >>> df2h.sort_index(level=["belief_time", "source"]).head(15)
                                                                                         event_value
    event_start               belief_time               source   cumulative_probability
    2000-01-03 09:00:00+00:00 2000-01-01 00:00:00+00:00 Source A 0.158700                       90.0
                                                                 0.500000                      100.0
                                                                 1.000000                      110.0
    2000-01-03 10:00:00+00:00 2000-01-01 00:00:00+00:00 Source A 0.025186                      225.0
                                                                 0.079350                      235.0
                                                                 0.133514                      240.0
                                                                 0.212864                      245.0
                                                                 0.329350                      250.0
                                                                 0.408700                      255.0
                                                                 0.579350                      260.0
                                                                 0.750000                      265.0
                                                                 1.000000                      275.0
    2000-01-03 12:00:00+00:00 2000-01-01 00:00:00+00:00 Source A 0.158700                      360.0
                                                                 0.500000                      400.0
                                                                 1.000000                      440.0
    sensor: <Sensor: weight>, event_resolution: 2:00:00
    >>> -df2h.knowledge_horizons[0]
    Timedelta('0 days 02:00:00')

Notice the time-aggregation of probabilistic beliefs about the two events between 10 AM and noon.
Three possible outcomes for both events led to nine possible worlds, because downsampling assumes by default that the values indicate discrete possible outcomes.
