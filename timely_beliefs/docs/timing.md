# Keeping track of time

## Table of contents

1. [Events and sensors](#events-and-sensors)
1. [Beliefs in physics](#beliefs-in-physics)
1. [Beliefs in economics](#beliefs-in-economics)
1. [A common misconception](#a-common-misconception)
1. [Special cases](#special-cases)

## Events and sensors

Numerical data quantifies something, it is a value assigned to an event;
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

## Beliefs in economics

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

## A common misconception

In many applications, people tend to interpret a forecast horizon as the duration between forming the belief and the start of the event.
When this happens, we have:

    forecast_horizon = event_start - belief_time

and:

    forecast_horizon = belief_horizon + knowledge_horizon

For example, consider a forecast formed at 9 AM about the average wind speed between 10 and 11 AM.
It may feel intuitive to talk about a forecast horizon of 1 hour, because people tend to index events by their start time and then talk about the timing of beliefs with respect to that index.

While this is a perfectly acceptable definition, we set out to be precise in handling the timing of beliefs.
Therefore, we use the term `belief_horizon` rather than `forecast_horizon` throughout the `timely-beliefs` package.

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
