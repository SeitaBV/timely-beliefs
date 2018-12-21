from datetime import datetime, timedelta

from timely_beliefs import TimedBelief, Sensor
from timely_beliefs.utils import timedelta_x_days_ago_at_y_oclock


# Define sensor for instantaneous events
Sensor()

# Define sensor for time slot events
Sensor(event_resolution=timedelta(minutes=15))

# Define sensor for time slot events known in advance
Sensor(
    event_resolution=timedelta(minutes=15),
    knowledge_horizon=(timedelta_x_days_ago_at_y_oclock, dict(x=1, y=12)),
)

# Define day-ahead belief about a time slot
TimedBelief(
    belief_time=datetime(2018, 1, 1, 15),
    event_start=datetime(2018, 1, 2, 0),
    event_end=datetime(2018, 1, 2, 1)
)

# Define day-ahead belief about an event
TimedBelief(
    belief_time=datetime(2018, 1, 1, 15),
    event_time=datetime(2018, 1, 2, 0),
)

