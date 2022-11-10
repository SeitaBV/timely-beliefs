# Convenient slicing methods

## Table of contents

1. [Slicing as usual](#slicing-as-usual)
1. [Rolling viewpoint](#rolling-viewpoint)
1. [Fixed viewpoint](#fixed-viewpoint)
1. [Belief history](#belief-history)

## Slicing as usual

Being an extension of the pandas DataFrame, all of pandas excellent slicing methods are available on the BeliefsDataFrame.
For example, to select all beliefs about events from 11 AM onwards:

    >>> from datetime import datetime, timedelta
    >>> import pytz
    >>> import timely_beliefs as tb
    >>> df = tb.examples.get_example_df()
    >>> df[df.index.get_level_values("event_start") >= datetime(2000, 1, 3, 11, tzinfo=pytz.utc)]

Besides these, `timely-beliefs` provides custom methods to conveniently slice through time in different ways.

## Rolling viewpoint

Select the latest forecasts from a rolling viewpoint (beliefs formed at least 2 days and 10 hours before the event could be known):

    >>> df.rolling_viewpoint(timedelta(days=2, hours=10))
                                                                               event_value
    event_start               belief_horizon  source   cumulative_probability
    2000-01-03 10:00:00+00:00 2 days 10:15:00 Source A 0.1587                          180
                                                       0.5000                          200
                                                       0.8413                          220
                                              Source B 0.5000                            0
                                                       1.0000                          200
    2000-01-03 11:00:00+00:00 2 days 10:15:00 Source A 0.1587                          297
                                                       0.5000                          300
                                                       0.8413                          303
                                              Source B 0.5000                            0
                                                       1.0000                          300
    2000-01-03 12:00:00+00:00 2 days 11:15:00 Source A 0.1587                          396
                                                       0.5000                          400
    sensor: <Sensor: weight>, event_resolution: 0:15:00

## Fixed viewpoint

Select the latest forecasts from a fixed viewpoint (beliefs formed at least before 2 AM January 1st 2000:

    >>> df.fixed_viewpoint(datetime(2000, 1, 1, 2, tzinfo=pytz.utc)).head(8)
                                                                                         event_value
    event_start               belief_time               source   cumulative_probability
    2000-01-03 09:00:00+00:00 2000-01-01 01:00:00+00:00 Source A 0.1587                           99
                                                                 0.5000                          100
                                                                 0.8413                          101
                                                        Source B 0.5000                            0
                                                                 1.0000                          100
    2000-01-03 10:00:00+00:00 2000-01-01 01:00:00+00:00 Source A 0.1587                          198
                                                                 0.5000                          200
                                                                 0.8413                          202
    sensor: <Sensor: weight>, event_resolution: 0:15:00

## Belief history

Select a history of beliefs about a single event:

    >>> df.belief_history(datetime(2000, 1, 3, 11, tzinfo=pytz.utc))
                                                               event_value
    belief_time               source   cumulative_probability
    2000-01-01 00:00:00+00:00 Source A 0.1587                          270
                                       0.5000                          300
                                       0.8413                          330
                              Source B 0.5000                            0
                                       1.0000                          300
    2000-01-01 01:00:00+00:00 Source A 0.1587                          297
                                       0.5000                          300
                                       0.8413                          303
                              Source B 0.5000                            0
                                       1.0000                          300
    sensor: <Sensor: weight>, event_resolution: 0:15:00
