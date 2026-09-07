"""The COPY path must write exactly what the multi-row INSERT wrote.

`add_to_session` streams large batches of beliefs with PostgreSQL's COPY instead of
binding one parameter per value. These tests drive batches over COPY_THRESHOLD, so they
take that path, and compare against the same beliefs written the old way.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz
from sqlalchemy.exc import IntegrityError

from timely_beliefs import BeliefsDataFrame, DBTimedBelief
from timely_beliefs.beliefs import classes
from timely_beliefs.tests import session

N = classes.COPY_THRESHOLD + 50  # comfortably over the threshold


def frame_of(sensor, source, n=N, offset=0.0, belief_horizon=timedelta(0)):
    """A frame of `n` beliefs about consecutive events, one belief each."""
    index = pd.date_range(
        datetime(2000, 1, 3, 9, tzinfo=pytz.utc),
        periods=n,
        freq=sensor.event_resolution,
        name="event_start",
    )
    return BeliefsDataFrame(
        pd.Series(
            [float(i) + offset for i in range(n)], index=index, name="event_value"
        ),
        belief_horizon=belief_horizon,
        sensor=sensor,
        source=source,
    )


def read_back(sensor, source):
    bdf = DBTimedBelief.search_session(session=session, sensor=sensor, source=source)
    return bdf.reset_index().sort_values("event_start").reset_index(drop=True)


def test_copy_path_is_taken(time_slot_sensor, test_source_a):
    """A batch over the threshold goes through COPY; a small one does not."""
    assert classes._should_copy(session, N) is True
    assert classes._should_copy(session, classes.COPY_THRESHOLD - 1) is False


@pytest.mark.parametrize("allow_overwrite", [False, True])
def test_copy_matches_insert(
    time_slot_sensor, test_source_a, test_source_b, allow_overwrite
):
    """The same beliefs written via COPY and via the multi-row INSERT come back equal."""
    via_copy = frame_of(time_slot_sensor, test_source_a)
    DBTimedBelief.add_to_session(
        session, via_copy, allow_overwrite=allow_overwrite, commit_transaction=True
    )

    # Force the INSERT path for the second source, by raising the threshold.
    original = classes.COPY_THRESHOLD
    classes.COPY_THRESHOLD = N + 1
    try:
        via_insert = frame_of(time_slot_sensor, test_source_b)
        DBTimedBelief.add_to_session(
            session,
            via_insert,
            allow_overwrite=allow_overwrite,
            commit_transaction=True,
        )
    finally:
        classes.COPY_THRESHOLD = original

    copied = read_back(time_slot_sensor, test_source_a)
    inserted = read_back(time_slot_sensor, test_source_b)

    assert len(copied) == N
    assert len(inserted) == N
    pd.testing.assert_series_equal(
        copied["event_value"], inserted["event_value"], check_names=False
    )
    pd.testing.assert_series_equal(
        copied["event_start"], inserted["event_start"], check_names=False
    )
    pd.testing.assert_series_equal(
        copied["belief_time"], inserted["belief_time"], check_names=False
    )


def test_copy_upsert_overwrites(time_slot_sensor, test_source_a):
    """With allow_overwrite, a second COPY of the same events updates their values."""
    DBTimedBelief.add_to_session(
        session, frame_of(time_slot_sensor, test_source_a), commit_transaction=True
    )
    DBTimedBelief.add_to_session(
        session,
        frame_of(time_slot_sensor, test_source_a, offset=1000.0),
        allow_overwrite=True,
        commit_transaction=True,
    )
    got = read_back(time_slot_sensor, test_source_a)
    assert len(got) == N, "upsert should update rows, not add them"
    assert got["event_value"].iloc[0] == 1000.0
    assert got["event_value"].iloc[-1] == float(N - 1) + 1000.0


def test_copy_without_overwrite_still_conflicts(time_slot_sensor, test_source_a):
    """Without allow_overwrite, a duplicate raises, exactly as the INSERT path did."""
    DBTimedBelief.add_to_session(
        session, frame_of(time_slot_sensor, test_source_a), commit_transaction=True
    )
    with pytest.raises(IntegrityError):
        DBTimedBelief.add_to_session(
            session,
            frame_of(time_slot_sensor, test_source_a, offset=1.0),
            commit_transaction=True,
        )
    session.rollback()


def test_copy_handles_negative_and_fractional_horizons(time_slot_sensor, test_source_a):
    """Belief horizons are written as intervals, including negative ones."""
    horizon = timedelta(seconds=-1.5)
    DBTimedBelief.add_to_session(
        session,
        frame_of(time_slot_sensor, test_source_a, belief_horizon=horizon),
        commit_transaction=True,
    )
    got = read_back(time_slot_sensor, test_source_a)
    assert len(got) == N
    # belief_time = event knowledge time - horizon, so a negative horizon lands after it
    assert (got["belief_time"] - got["event_start"]).nunique() == 1
