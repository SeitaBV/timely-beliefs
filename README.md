# Timely beliefs

*Model data as beliefs (at a certain time) about events (at a certain time).*

The `timely-beliefs` package provides a convenient data model for numerical time series,
that is both simple enough for humans to understand and sufficiently rich for forecasting and machine learning.
The data model includes the origins of data to answer such question as:

- Who (or what) created the data?
- When was the data created?
- How certain were they?

Some use cases of the package:

- Clearly distinguish forecasts from rolling forecasts.
- Analyse your predictive power by showing forecast accuracy as you approach an event.
- Learn **when** someone is a bad predictor.
- Evaluate the risk of being wrong about an event.

## Events and sensors

Numerical data quantifies something. We say that it assigns a value to an event;
a physical event, such as a temperature reading, power flow or water flow,
or an economical event, such as daily goods sold or a price agreement about a future delivery.
An event typically takes some time, so it has an `event_start` and an `event_end` time.

To observe an event, we need a sensor. We say that a sensor has a certain `event_resolution`:

    event_resolution = event_end - event_start

We define the resolution to be a fixed property of the sensor. For example:

- An anemometer (wind speed meter) determines the number of revolutions within some period of time.
- A futures contract determines the price of a future delivery within some period of time.

## Beliefs in physics

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

## Beliefs in economics

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

## Special cases

### Instantaneous events
Instantaneous events can be modelled by defining a sensor with:

    event_resolution = 0

### Past events

Beliefs about past events can be modelled using a negative horizon:

    belief_horizon < 0
    knowledge_time < belief_time

That is, your beliefs can still change after you (think you) know about an event.
NB in the following case a price has been determined (you could know about it) for a future event:

    knowledge_time < belief_time < event_start 

### Ex-post knowledge

Our concept of `knowledge_time` supports to define sensors for agreements about ongoing or past events, such as ex-post contracts.

    event_start < knowledge_time
    -resolution < knowledge_horizon < 0  # for ongoing events
    knowledge_horizon < -resolution  # for past events
