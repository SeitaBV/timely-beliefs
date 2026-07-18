"""Benchmark the search_session hot path against the test database.

Usage:

    python dev/bench_search_session.py [--events 500] [--horizons 100] [--sources 2] [--reps 5] [--profile]

Creates all tables in the test database (assumed clean, like the test suite does),
bulk-inserts synthetic beliefs, times three search scenarios, and drops all tables again.

Scenarios:

1. plain:         search_session with only an event time range
2. most_recent:   most_recent_beliefs_only=True (SQL subquery path)
3. fallback:      most_recent_beliefs_only=True + beliefs_before on a sensor with a
                  custom knowledge horizon function (pandas post-processing fallback)
"""

import argparse
import cProfile
import pstats
import time
from datetime import datetime, timedelta
from statistics import median

from pytz import utc
from sqlalchemy import insert

from timely_beliefs import DBBeliefSource, DBSensor, DBTimedBelief
from timely_beliefs.db_base import Base
from timely_beliefs.sensors.func_store.knowledge_horizons import x_days_ago_at_y_oclock
from timely_beliefs.tests import engine, session

EVENT_RESOLUTION = timedelta(minutes=15)
START = datetime(2025, 1, 1, tzinfo=utc)


def setup(n_events: int, n_horizons: int, n_sources: int) -> DBSensor:
    Base.metadata.create_all(engine)
    sensor = DBSensor(
        name="BenchSensor",
        event_resolution=EVENT_RESOLUTION,
        knowledge_horizon=(
            x_days_ago_at_y_oclock,
            dict(x=1, y=12, z="Europe/Amsterdam"),
        ),
    )
    session.add(sensor)
    sources = [DBBeliefSource(name=f"Source {i}") for i in range(n_sources)]
    session.add_all(sources)
    session.flush()
    rows = [
        dict(
            sensor_id=sensor.id,
            source_id=source.id,
            event_start=START + e * EVENT_RESOLUTION,
            belief_horizon=timedelta(hours=h),
            cumulative_probability=0.5,
            event_value=float(e + h),
        )
        for e in range(n_events)
        for h in range(n_horizons)
        for source in sources
    ]
    session.execute(insert(DBTimedBelief), rows)
    session.flush()
    print(f"Inserted {len(rows)} beliefs.")
    return sensor


def bench(label: str, fn, reps: int, profile: bool = False):
    times = []
    df = None
    for _ in range(reps):
        t0 = time.perf_counter()
        df = fn()
        times.append(time.perf_counter() - t0)
    print(
        "{:>12}: min {:8.1f} ms | median {:8.1f} ms | shape {}".format(
            label, min(times) * 1000, median(times) * 1000, df.shape
        )
    )
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        fn()
        profiler.disable()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--horizons", type=int, default=100)
    parser.add_argument("--sources", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    sensor = setup(args.events, args.horizons, args.sources)
    event_end = START + args.events * EVENT_RESOLUTION
    beliefs_before = START - timedelta(hours=args.horizons // 2)
    try:
        scenarios = {
            "plain": dict(),
            "most_recent": dict(most_recent_beliefs_only=True),
            "fallback": dict(
                most_recent_beliefs_only=True,
                beliefs_before=beliefs_before,
            ),
        }
        results = {}
        for label, kwargs in scenarios.items():
            results[label] = bench(
                label,
                lambda kwargs=kwargs: DBTimedBelief.search_session(
                    session,
                    sensor,
                    event_starts_after=START,
                    event_ends_before=event_end,
                    **kwargs,
                ),
                args.reps,
                profile=args.profile,
            )
        for label, df in results.items():
            print("\n{} head:\n{}".format(label, df.head(3)))
            print(
                f"{label} index dtypes: {[df.index.get_level_values(n).dtype for n in df.index.names]}"
            )
    finally:
        session.rollback()
        session.close()
        Base.metadata.drop_all(engine)
        print("\nDropped all tables.")


if __name__ == "__main__":
    main()
