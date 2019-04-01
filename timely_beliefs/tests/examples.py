from datetime import datetime, timedelta
from pytz import utc

from timely_beliefs import BeliefsDataFrame, BeliefSource, Sensor, TimedBelief


def df_example() -> BeliefsDataFrame:
    """Nice BeliefsDataFrame to show.
    For a single sensor, it contains 4 events, for each of which 2 beliefs by 2 sources each, described by 2 or 3
    probabilistic values, depending on the source.
    Note that the event resolution of the sensor is 15 minutes.
    """

    n_events = 4
    n_beliefs = 2
    n_sources = 2
    true_value = 100

    example_sensor = Sensor(event_resolution=timedelta(minutes=15), name="Sensor 1")
    example_source_a = BeliefSource(name="Source A")
    example_source_b = BeliefSource(name="Source B")

    sources = [example_source_a, example_source_b]
    cps = [0.1587, 0.5, 0.8413, 0.5, 1]

    # Build up a BeliefsDataFrame with various events, beliefs, sources and probabilistic accuracy (for a single sensor)
    beliefs = [
        TimedBelief(
            source=sources[s],
            sensor=example_sensor,
            value=int(
                1
                * (e + 1)
                * (
                    true_value + (10 ** (n_beliefs - b - 1)) * (cps[p] - 0.5) / 0.3413
                    if s % 2 == 0
                    else true_value * (p - 3)
                )
            ),
            belief_time=datetime(2000, 1, 1, tzinfo=utc) + timedelta(hours=b),
            event_start=datetime(2000, 1, 3, 9, tzinfo=utc) + timedelta(hours=e),
            cumulative_probability=cps[p],
        )
        for e in range(n_events)  # 4 events
        for b in range(n_beliefs)  # 2 beliefs
        for s in range(n_sources)  # 2 sources
        for p in range(
            3 * (s % 2), 2 * (s % 2) + 3
        )  # alternating 3 and 2 cumulative probabilities
    ]
    return BeliefsDataFrame(sensor=example_sensor, beliefs=beliefs)
