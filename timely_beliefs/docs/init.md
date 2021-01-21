# Creating a BeliefsDataFrame

## Table of contents

1. [Sensors and BeliefSources](#sensors-and-beliefsources)
1. [From a Pandas Series](#from-a-pandas-series)
1. [From a Pandas DataFrame](#from-a-pandas-dataframe)
1. [From a CSV file](#from-a-csv-file)
1. [From a list of TimedBeliefs](#from-a-list-of-timedbeliefs)

## Sensors and BeliefSources

Each belief requires a `Sensor` and a `BeliefSource`. For example:

    >>> import timely_beliefs as tb
    >>> sensor = tb.Sensor("EPEX SPOT day-ahead price", event_resolution=timedelta(hours=1), unit="EUR/MWh")
    >>> source = tb.BeliefSource("EPEX")

When creating a `BeliefsDataFrame`, a sensor always needs to be passed.
A `BeliefsDataFrame` can only refer to a single sensor.
A source can be passed once for the entire frame, or per belief.

## From a Pandas Series

Create a BeliefsDataFrame from a conventional time series (a Pandas Series with a DatetimeIndex) together with keyword arguments.

Required arguments:

- either:
  - `belief_time` (one moment at which all beliefs were formed), or
  - `belief_horizon` (each belief was formed the same duration before or after)
- `source` (one source for all beliefs)
- `sensor` (one sensor to which all beliefs refer)


    >>> import pandas as pd
    >>> s = pd.Series([63, 60], index=pd.date_range(datetime(2000, 1, 3, 9), periods=2, tz=pytz.utc))
    >>> bdf = tb.BeliefsDataFrame(s, belief_horizon=timedelta(hours=0), source=source, sensor=sensor)
    >>> print(bdf)
                                                                            event_value
    event_start               belief_horizon source cumulative_probability             
    2000-01-03 09:00:00+00:00 0 days         EPEX   0.5                              63
    2000-01-04 09:00:00+00:00 0 days         EPEX   0.5                              60

## From a Pandas DataFrame

Pass a Pandas DataFrame with columns ["event_start", "belief_time", "source", "cumulative_probability", "event_value"]. The "cumulative_probability" column is optional (a default of 0.5 will be used if the column is missing).

    >>> df = pd.DataFrame([[63, datetime(2000, 1, 3, 9, tzinfo=pytz.utc), timedelta(hours=0), source], [60, datetime(2000, 1, 3, 10, tzinfo=pytz.utc), timedelta(hours=0), source]], columns=["event_value", "event_start", "belief_horizon", "source"])
    >>> bdf = tb.BeliefsDataFrame(df, sensor=sensor)
    >>> print(bdf)
                                                                            event_value
    event_start               belief_horizon source cumulative_probability             
    2000-01-03 09:00:00+00:00 0 days         EPEX   0.5                              63
    2000-01-03 10:00:00+00:00 0 days         EPEX   0.5                              60

Alternatively, a keyword argument can be used to replace a column that contains the same value for each belief.

    >>> df = pd.DataFrame([[63, datetime(2000, 1, 3, 9, tzinfo=pytz.utc)], [60, datetime(2000, 1, 3, 10, tzinfo=pytz.utc)]], columns=["event_value", "event_start"])
    >>> bdf = tb.BeliefsDataFrame(df, belief_horizon=timedelta(hours=0), source=source, sensor=sensor)
    >>> print(bdf)
                                                                            event_value
    event_start               belief_horizon source cumulative_probability             
    2000-01-03 09:00:00+00:00 0 days         EPEX   0.5                              63
    2000-01-03 10:00:00+00:00 0 days         EPEX   0.5                              60

## From a CSV file

The utility function `tb.read_csv` lets you load a BeliefsDataFrame from a csv file.
You still need to set the source and sensor for the BeliefsDataFrame; the csv file only contains their names.

An example, adapted from `timely_beliefs/example/__init__.py`:

    temperature_df = tb.read_csv(
        "temperature.csv",
        sensor=tb.Sensor("Thermometer A", unit="°C", event_resolution=timedelta(hours=1)),
        source=tb.BeliefSource("Source X"),
    )

In case the csv file contains multiple source names, you can pass a list of sources.
Each source name will then be replaced by the actual `Source` object from the list you provided.
To write a BeliefsDataFrame to a csv file, just use the pandas way:

    >>> bdf.to_csv("data.csv")
    >>> tb.read_csv("data.csv", source=source, sensor=sensor)

## From a list of TimedBeliefs

Create a list of `TimedBelief` or `DBTimedBelief` objects and use it to initialize a BeliefsDataFrame.

    >>> from datetime import datetime, timedelta
    >>> import pytz
    >>> belief_1 = tb.TimedBelief(event_value=63, event_start=datetime(2000, 1, 3, 9, tzinfo=pytz.utc), belief_horizon=timedelta(hours=0), sensor=sensor, source=source)
    >>> belief_2 = tb.TimedBelief(event_value=60, event_start=datetime(2000, 1, 3, 10, tzinfo=pytz.utc), belief_horizon=timedelta(hours=0), sensor=sensor, source=source)
    >>> beliefs = [belief_1, belief_2]
    >>> bdf = tb.BeliefsDataFrame(beliefs)
    >>> print(bdf)
                                                                                       event_value
    event_start               belief_time               source cumulative_probability             
    2000-01-03 09:00:00+00:00 2000-01-03 10:00:00+00:00 EPEX   0.5                              63
    2000-01-03 10:00:00+00:00 2000-01-03 11:00:00+00:00 EPEX   0.5                              60
