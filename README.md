# Timely beliefs

[![Build Status](https://travis-ci.com/SeitaBV/timely-beliefs.svg?branch=master)](https://travis-ci.com/SeitaBV/timely-beliefs)
[![Python Version](https://img.shields.io/pypi/pyversions/timely-beliefs.svg)](https://pypi.python.org/pypi/timely-beliefs)
[![Pypi Version](https://img.shields.io/pypi/v/timely-beliefs.svg)](https://pypi.python.org/pypi/timely-beliefs)

*Model data as beliefs (at a certain time) about events (at a certain time).*

The `timely-beliefs` package provides a convenient data model for numerical time series,
that is both simple enough for humans to understand and sufficiently rich for forecasting and machine learning.
The data model is an extended [pandas](https://pandas.pydata.org/) DataFrame that contains the origins of data to answer such question as:

- Who (or what) created the data?
- When was the data created?
- How certain were they?

Some use cases of the package:

- Clearly distinguish forecasts from rolling forecasts.
- Analyse your predictive power by showing forecast accuracy as you approach an event.
- Learn **when** someone is a bad predictor.
- Evaluate the risk of being wrong about an event.

## Table of contents
1. [The data model](#the-data-model)
    1. [Keeping track of time](#keeping-track-of-time)
        1. [Events and sensors](#events-and-sensors)
        1. [Beliefs in physics](#beliefs-in-physics)
        1. [Beliefs in economics](#beliefs-in-economics)
        1. [Special cases](#special-cases)
    1. [Convenient slicing methods](#convenient-slicing-methods)
    1. [Resampling](#resampling)
    1. [Lineage](#lineage)
1. [More examples](#more-examples)

## The data model

The example BeliefsDataFrame in our tests module demonstrates the basic timely-beliefs data model:

    >>> import timely_beliefs
    >>> df = timely_beliefs.tests.example_df
    >>> df.head(8)
                                                                                     event_value
    event_start               belief_time               source_id belief_percentile             
    2000-01-03 09:00:00+00:00 2000-01-01 00:00:00+00:00 1         0.1587                      90
                                                                  0.5000                     100
                                                                  0.8413                     110
                                                        2         0.5000                       0
                                                                  1.0000                     100
                              2000-01-01 01:00:00+00:00 1         0.1587                      99
                                                                  0.5000                     100
                                                                  0.8413                     101
The first 8 entries of this BeliefsDataFrame show beliefs about a single event.
Beliefs were formed by two distinct sources (1 and 2), with the first updating its beliefs at a later time.
Source 1 first thought the value of this event would be 100 ± 10 (the percentiles suggest a normal distribution),
and then increased its accuracy by lowering the standard deviation to 1.
Source 2 thought the value would be equally likely to be 0 or 100.

- Read more about how the DataFrame is [keeping track of time](#keeping-track-of-time).
- Discover [convenient slicing methods](#convenient-slicing-methods), e.g. to show a rolling horizon forecast.
- Serve your data fast by [resampling](#resampling), while taking into account auto-correlation.
- Track where your data comes from, by following its [lineage](#lineage). 

### Keeping track of time

#### Events and sensors

Numerical data quantifies something, it is a value assigned to an event;
a physical event, such as a temperature reading, power flow or water flow,
or an economical event, such as daily goods sold or a price agreement about a future delivery.
An event typically takes some time, so it has an `event_start` and an `event_end` time.

To observe an event, we need a sensor. We say that a sensor has a certain `event_resolution`:

    event_resolution = event_end - event_start

We define the resolution to be a fixed property of the sensor. For example:

- An anemometer (wind speed meter) determines the number of revolutions within some period of time.
- A futures contract determines the price of a future delivery within some period of time.

#### Beliefs in physics

Assigning a value to an event implies a belief.
We say that a belief can be formed some time before or after an event, and call this time `belief_time`.
A weather forecast is a good example, where:

    belief_time < event_end

For physical events, the time at which we can say the event could be known is at the `event_end`.

    knowledge_time = event_end

The forecast horizon, or `belief_horizon`, says how long before (the event could be known) the belief was formed: 

    belief_horizon = knowledge_time - belief_time

For example, a forecast of solar irradiation on June 10th 2017 with a horizon of 27 hours means a belief_time of 9 PM on June 9th 2017.
That is:

    event_start = datetime(2017, 6, 10, hour=0)
    event_end = datetime(2017, 6, 11, hour=0)
    belief_horizon = timedelta(hours=27)
    belief_time = datetime(2017, 6, 9, hour=21)

#### Beliefs in economics

For economical events, the time at which we say an event could be known is typically not at the `end`.
Most contracts deal with future events, such that:

    knowledge_time < event_start

The `knowledge horizon` says how long before (the event starts) the event could be known:

    knowledge_horizon > 0  # for most economical events
    knowledge_horizon = -resolution  # for physical events

We define the knowledge horizon to be a fixed property of the sensor.
For example, hourly prices on the day-ahead electricity market are determined at noon one day before delivery starts, such that:

    knowledge_time = event_start.replace(hour=12) - timedelta(days=1)

Then for an hourly price between 3 and 4 PM on June 10th 2017:
    
    event_start = datetime(2017, 6, 10, hour=15)
    event_end = datetime(2017, 6, 10, hour=16)
    knowledge_time = datetime(2017, 6, 9, hour=12)
    knowledge_horizon = timedelta(hours=27)

Continuing this example, a price forecast with a forecast horizon of 1 hour constitutes a belief formed at 11 AM:

    belief_horizon = timedelta(hours=1)
    belief_time = datetime(2017, 6, 9, hour=11)

In general, we have the following relationships:

    belief_time + belief_horizon = knowledge_time
    belief_time + belief_horizon + knowledge_horizon = event_start 
    belief_time + belief_horizon + knowledge_horizon + event_resolution = event_end

#### Special cases

##### Instantaneous events
Instantaneous events can be modelled by defining a sensor with:

    event_resolution = 0

##### Past events

Beliefs about past events can be modelled using a negative horizon:

    belief_horizon < 0
    knowledge_time < belief_time

That is, your beliefs can still change after you (think you) know about an event.
NB in the following case a price has been determined (you could know about it) for a future event:

    knowledge_time < belief_time < event_start 

##### Ex-post knowledge

Our concept of `knowledge_time` supports to define sensors for agreements about ongoing or past events, such as ex-post contracts.

    event_start < knowledge_time
    -resolution < knowledge_horizon < 0  # for ongoing events
    knowledge_horizon < -resolution  # for past events

### Convenient slicing methods

Select the latest forecasts for a rolling horizon (beliefs formed at least 2 days and 10 hours before the event could be known): 

    >>> from datetime import timedelta
    >>> df = timely_beliefs.tests.example_df
    >>> df.rolling_horizon(timedelta(days=2, hours=10))
                                                                           event_value
    event_start               belief_horizon  source_id belief_percentile             
    2000-01-03 11:00:00+00:00 2 days 10:15:00 1         0.1587                     297
                                                        0.5000                     300
                                                        0.8413                     303
                                              2         1.0000                     300
    2000-01-03 12:00:00+00:00 2 days 11:15:00 1         0.1587                     396
                                                        0.8413                     404
                                              2         0.5000                       0
                                                        1.0000                     400

Select a history of beliefs about a single event:

    >>> from datetime import datetime
    >>> df.belief_history(datetime(2000, 1, 3, 11))
                                                           event_value
    belief_time               source_id belief_percentile             
    2000-01-01 00:00:00+00:00 1         0.1587                     270
                                        0.5000                     300
                                        0.8413                     330
                              2         0.5000                       0
                                        1.0000                     300
    2000-01-01 01:00:00+00:00 1         0.1587                     297
                                        0.5000                     300
                                        0.8413                     303
                              2         0.5000                       0
                                        1.0000                     300

### Resampling

Upsample to events with a resolution of 5 minutes:

    >>> from datetime import timedelta
    >>> df = timely_beliefs.tests.example_df
    >>> df = df.resample_events(timedelta(minutes=5)
    >>> df.head(9)
                                                                                     event_value
    event_start               belief_time               source_id belief_percentile             
    2000-01-03 09:00:00+00:00 2000-01-01 00:00:00+00:00 1         0.1587                    90.0
    2000-01-03 09:05:00+00:00 2000-01-01 00:00:00+00:00 1         0.1587                    90.0
    2000-01-03 09:10:00+00:00 2000-01-01 00:00:00+00:00 1         0.1587                    90.0
    2000-01-03 09:00:00+00:00 2000-01-01 00:00:00+00:00 1         0.5000                   100.0
    2000-01-03 09:05:00+00:00 2000-01-01 00:00:00+00:00 1         0.5000                   100.0
    2000-01-03 09:10:00+00:00 2000-01-01 00:00:00+00:00 1         0.5000                   100.0
    2000-01-03 09:00:00+00:00 2000-01-01 00:00:00+00:00 1         0.8413                   110.0
    2000-01-03 09:05:00+00:00 2000-01-01 00:00:00+00:00 1         0.8413                   110.0
    2000-01-03 09:10:00+00:00 2000-01-01 00:00:00+00:00 1         0.8413                   110.0

Downsample to events with a resolution of 2 hours:

    >>> from datetime import timedelta
    >>> df = timely_beliefs.tests.example_df
    >>> df.resample_events(timedelta(hours=2)


### Lineage

Get the (number of) sources contributing to the BeliefsDataFrame:

    >>> df.lineage.sources
    array([1, 2], dtype=int64)
    >>> df.lineage.number_of_sources
    2

## More examples

...
