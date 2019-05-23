# Timely beliefs

[![Build Status](https://travis-ci.com/SeitaBV/timely-beliefs.svg?branch=master)](https://travis-ci.com/SeitaBV/timely-beliefs)
[![Python Version](https://img.shields.io/pypi/pyversions/timely-beliefs.svg)](https://pypi.python.org/pypi/timely-beliefs)
[![Pypi Version](https://img.shields.io/pypi/v/timely-beliefs.svg)](https://pypi.python.org/pypi/timely-beliefs)

_Model data as beliefs (at a certain time) about events (at a certain time)._

The `timely-beliefs` package provides a convenient data model for numerical time series,
that is both simple enough for humans to understand and sufficiently rich for forecasting and machine learning.
The data model is an extended [pandas](https://pandas.pydata.org/) DataFrame that assigns properties and index levels to describe:

- [What the data is about](#events-and-sensors)
- Who (or what) created the data
- [When the data was created](#beliefs-in-physics)
- How certain they were

The package contains the following functionality:

- [A model for time series data](#the-data-model), suitable for a notebook or a database-backed program (using [sqlalchemy](https://sqlalche.me))
- [Selecting/querying beliefs](#convenient-slicing-methods), e.g. those held at a certain moment in time
- [Computing accuracy](#accuracy), e.g. against after-the-fact knowledge, also works with probabilistic forecasts
- [Resampling time series with uncertainty](#resampling) (experimental)
- [Visualising time series and accuracy metrics](#visualisation) (experimental)

Some use cases of the package:

- Clearly distinguish forecasts from rolling forecasts.
- Analyse your predictive power by showing forecast accuracy as you approach an event.
- Learn **when** someone is a bad predictor.
- Evaluate the risk of being wrong about an event.

Check out [our interactive demonstration](http://forecasting-accuracy.seita.nl) comparing forecasting models for renewable energy production.
These visuals are created simply by calling the plot method on our BeliefsDataFrame, using the visualisation library [Altair](https://altair-viz.github.io/).

![Comparing wind speed forecasting models](timely_beliefs/docs/comparing_wind_speed_forecasting_models.png)

## Table of contents

1. [The data model](#the-data-model)
   1. [Keeping track of time](#keeping-track-of-time)
      1. [Events and sensors](#events-and-sensors)
      1. [Beliefs in physics](#beliefs-in-physics)
      1. [Beliefs in economics](#beliefs-in-economics)
      1. [A common misconception](#a-common-misconception)
      1. [Special cases](#special-cases)
   1. [Convenient slicing methods](#convenient-slicing-methods)
   1. [Resampling](#resampling)
   1. [Lineage](#lineage)
   1. [Database storage](#database-storage)
      1. [Table creation and session](#table-creation-and-session)
      1. [Subclassing](#subclassing)
1. [Accuracy](#accuracy)
1. [Visualisation](#visualisation)
1. [More examples](#more-examples)

## The data model

The example BeliefsDataFrame in our examples module demonstrates the basic timely-beliefs data model:

    >>> import timely_beliefs
    >>> df = timely_beliefs.examples.example_df
    >>> df.head(8)
                                                                                         event_value
    event_start               belief_time               source   cumulative_probability
    2000-01-03 09:00:00+00:00 2000-01-01 00:00:00+00:00 Source A 0.1587                           90
                                                                 0.5000                          100
                                                                 0.8413                          110
                                                        Source B 0.5000                            0
                                                                 1.0000                          100
                              2000-01-01 01:00:00+00:00 Source A 0.1587                           99
                                                                 0.5000                          100
                                                                 0.8413                          101

The first 8 entries of this BeliefsDataFrame show beliefs about a single event.
Beliefs were formed by two distinct sources (1 and 2), with the first updating its beliefs at a later time.
Source A first thought the value of this event would be 100 ± 10 (the probabilities suggest a normal distribution),
and then increased its accuracy by lowering the standard deviation to 1.
Source B thought the value would be equally likely to be 0 or 100.

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

For physical events, the time at which we can say the event could be known (which we call `knowledge_time`) is at the `event_end`.

    knowledge_time = event_end

The forecast horizon, or `belief_horizon`, says how long before (the event could be known) the belief was formed:

    belief_horizon = knowledge_time - belief_time

For example, a forecast of solar irradiation on June 10th 2017 with a horizon of 27 hours means a belief time of 9 PM on June 9th 2017.
That is:

    event_start = datetime(2017, 6, 10, hour=0)
    event_end = datetime(2017, 6, 11, hour=0)
    belief_horizon = timedelta(hours=27)
    belief_time = datetime(2017, 6, 9, hour=21)

#### Beliefs in economics

For economical events, the time at which we say an event could be known is typically not at the `event_end`.
Most contracts deal with future events, such that:

    knowledge_time < event_start

The `knowledge_horizon` says how long before (the event starts) the event could be known:

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

#### A common misconception

In many applications, people tend to interpret a forecast horizon as the duration between forming the belief and the start of the event.
When this happens, we have:

    forecast_horizon = event_start - belief_time

and:

    forecast_horizon = belief_horizon + knowledge_horizon

For example, consider a forecast formed at 9 AM about the average wind speed between 10 and 11 AM.
It may feel intuitive to talk about a forecast horizon of 1 hour, because people tend to index events by their start time and then talk about the timing of beliefs with respect to that index.

While this is a perfectly acceptable definition, we set out to be precise in handling the timing of beliefs.
Therefore, we use the term `belief_horizon` rather than `forecast_horizon` throughout the `timely-beliefs` package.

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

#### Rolling viewpoint

Select the latest forecasts from a rolling viewpoint (beliefs formed at least 2 days and 10 hours before the event could be known):

    >>> from datetime import timedelta
    >>> df = timely_beliefs.examples.example_df
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

#### Fixed viewpoint

Select the latest forecasts from a fixed viewpoint (beliefs formed at least before 2 AM January 1st 2000:

    >>> from datetime import datetime
    >>> from pytz import utc
    >>> df.fixed_viewpoint(datetime(2000, 1, 1, 2, tzinfo=utc)).head(8)
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

Select a history of beliefs about a single event:

    >>> df.belief_history(datetime(2000, 1, 3, 11, tzinfo=utc))
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

### Resampling

Upsample to events with a resolution of 5 minutes:

    >>> from datetime import timedelta
    >>> df = timely_beliefs.examples.example_df
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

Notice the time-aggregation of probabilistic beliefs about the two events between 10 AM and noon.
Three possible outcomes for both events led to nine possible worlds, because downsampling assumes by default that the values indicate discrete possible outcomes.

### Lineage

Get the (number of) sources contributing to the BeliefsDataFrame:

    >>> df.lineage.sources
    array([1, 2], dtype=int64)
    >>> df.lineage.number_of_sources
    2

## Database storage

All of the above can be done with `TimedBelief` objects in a `BeliefsDataFrame`. However, if you are dealing with a lot of data and need performance, you'll want to persist your belief data in a database. The timely-beliefs library supports this, as all relevant classes have a subclass which also derives from [sqlalchemy's declarative base](https://docs.sqlalchemy.org/en/13/orm/extensions/declarative/index.html?highlight=declarative).

The timely-beliefs library comes with database-backed classes for the three main components of the data model - `DBTimedBelief`, `DBSensor` and `DBBeliefSource`. Objects from these classes can be used just like their super classes, so for instance `DBTimedBelief` objects can be used for creating a `BeliefDataFrame`.

### Table creation and storage

You can let sqlalchemy create the tables in your database session and start using the DB classes (or subclasses, see below) and program code without much work by yourself. The database session is under your control - where or how you get it, deoends on the context you're working in. Here is an example how to set up a session and also have sqlachemy create the tables:

    from timely_beliefs.db_base import Base as TBBase
    from sqlalchemy.orm import sessionmaker

    SessionClass = sessionmaker()
    session = None

    def create_db_and_session():
        engine = create_engine("your-db-connection-string")
        SessionClass.configure(bind=engine)

        TBBase.metadata.create_all(engine)

        if session is None:
            session = SessionClass()

        # maybe add some inital sensors and sources to your session here ...

        return session

Note how we're using timely-belief's sqlalchemy base (we're calling it `TBBase`) to create them. This does not create other tables you might have in your data model.

### Subclassing

`DBTimedBelief`, `DBSensor` and `DBBeliefSource` also can be subclassed, if more attributes are needed. This should be most interesting for sensors and maybe belief sources.

Below is an example, for the case of a db-backed case, where we wanted a sensor to have a location. We added three attributes, `latitude`, `longitude` and `location_name`:

    from timely_beliefs import DBSensor
    from sqlalchemy import Column, Float, String

    class DBLocatedSensor(DBSensor):
        """A sensor with a location lat/long and location name"""

        latitude = Column(Float(), nullable=False)
        longitude = Column(Float(), nullable=False)
        location_name = Column(String(80), nullable=False)

        def __init__(
            self,
            latitude: float = None,
            longitude: float = None,
            location_name: str = "",
            **kwargs,
        ):
            self.latitude = latitude
            self.longitude = longitude
            self.location_name = location_name
            DBSensor.__init__(self, **kwargs)

## Accuracy

The accuracy of a belief is defined with respect to some reference.
The default reference is the most recent belief held by the same source,
but it is possible to set beliefs held by a specific source at a specific time to serve as the reference instead.

### Accuracy and error metrics

To our knowledge, there is no standard metric for accuracy.
However, there are some standard metrics for what can be considered to be its opposite: error.
By default, we give back the Mean Absolute Error (MAE),
the Mean Absolute Percentage Error (MAPE)
and the Weighted Absolute Percentage Error (WAPE).
Each of these metrics is a representation of how wrong a belief is (believed to be),
with its convenience depending on use case.
For example, for intermittent demand time series (i.e. sparse data with lots of zero values) MAPE is not a useful metric.
For an intuitive representation of accuracy that works in many cases, we suggest to use `df["accuracy"] = 1 - df["wape"]`.
With this definition:

- 100% accuracy denotes that all values are correct
- 50% accuracy denotes that, on average, the values are wrong by half of the reference value
- 0% accuracy denotes that, on average, the values are wrong by exactly the reference value (i.e. zeros or twice the reference value)
- negative accuracy denotes that, on average, the values are off-the-chart wrong (by more than the reference value itself)

### Probabilistic forecasts

The previous metrics (MAE, MAPE and WAPE) are technically not defined for probabilistic beliefs.
However, there is a straightforward generalisation of MAE called the Continuous Ranked Probability Score (CRPS), which is used instead.
The other metrics follow by dividing over the deterministic reference value.
For simplicity in usage of the `timely-beliefs` package,
the metrics names in the BeliefsDataFrame are the same regardless of whether the beliefs are deterministic or probabilistic.

### Probabilistic reference

It is possible that the reference itself is a probabilistic belief rather than a deterministic belief.
Our implementation of CRPS handles this case, too, by calculating the distance between the cumulative distribution functions of each forecast and reference [(Hans Hersbach, 2000)](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0434%282000%29015%3C0559%3ADOTCRP%3E2.0.CO%3B2).
As the denominator for calculating MAPE and WAPE, we use the expected value of the probabilistic reference.

### Viewpoints

There are two common use cases for wanting to know the accuracy of beliefs,
each with a different viewpoint.
With a rolling viewpoint, you get the accuracy of beliefs at a certain `belief_horizon` before (or after) `knowledge_time`,
for example, some days before each event ends.

    >>> df.rolling_viewpoint_accuracy(timedelta(days=2, hours=9), reference_source=df.lineage.sources[0])
                     mae      mape      wape
    source
    Source A    1.482075  0.014821  0.005928
    Source B  125.853250  0.503413  0.503413

With a fixed viewpoint, you get the accuracy of beliefs held at a certain `belief_time`.

    >>> df.fixed_viewpoint_accuracy(datetime(2000, 1, 2, tzinfo=utc), reference_source=df.lineage.sources[0])
                    mae      mape      wape
    source
    Source A    0.00000  0.000000  0.000000
    Source B  125.85325  0.503413  0.503413

## Visualisation

Create interactive charts using Altair and view them in your browser.

    >>> chart = df.plot(reference_source=df.lineage.sources[0], show_accuracy=True)
    >>> chart.serve()

This will create the screenhsot at the top of this Readme.

...

## More examples

...
