"""The COPY path must write exactly what the multi-row INSERT wrote.

`add_to_session` streams large batches of beliefs with PostgreSQL's COPY instead of
binding one parameter per value. These tests drive batches over COPY_THRESHOLD, so they
take that path, and compare against the same beliefs written the old way.
"""

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz
from sqlalchemy.exc import IntegrityError

from timely_beliefs import BeliefsDataFrame, DBTimedBelief
from timely_beliefs.beliefs import classes
from timely_beliefs.tests import session

N = classes.COPY_THRESHOLD + 50  # comfortably over the threshold


def frame_of(sensor, source, n=N, offset=0.0, belief_horizon=timedelta(0), values=None):
    """A frame of `n` beliefs about consecutive events, one belief each."""
    index = pd.date_range(
        datetime(2000, 1, 3, 9, tzinfo=pytz.utc),
        periods=n,
        freq=sensor.event_resolution,
        name="event_start",
    )
    if values is None:
        values = [float(i) + offset for i in range(n)]
    return BeliefsDataFrame(
        pd.Series(values, index=index, name="event_value"),
        belief_horizon=belief_horizon,
        sensor=sensor,
        source=source,
    )


def read_back(sensor, source):
    bdf = DBTimedBelief.search_session(session=session, sensor=sensor, source=source)
    return bdf.reset_index().sort_values("event_start").reset_index(drop=True)


def test_copy_path_is_taken(time_slot_sensor, test_source_a, monkeypatch):
    """A batch over the threshold goes through COPY; a small one does not."""
    taken = []
    original = DBTimedBelief._copy_to_session.__func__
    monkeypatch.setattr(
        DBTimedBelief,
        "_copy_to_session",
        classmethod(lambda cls, *args: (taken.append(True), original(cls, *args))[1]),
    )

    DBTimedBelief.add_to_session(
        session, frame_of(time_slot_sensor, test_source_a), commit_transaction=True
    )
    assert taken == [True]

    small = classes.COPY_THRESHOLD - 1
    DBTimedBelief.add_to_session(
        session,
        frame_of(time_slot_sensor, test_source_a, n=small, offset=1000.0),
        allow_overwrite=True,
        commit_transaction=True,
    )
    assert taken == [True], "a batch under the threshold should keep the INSERT"


def test_copy_leaves_python_side_defaults_to_the_insert(time_slot_sensor):
    """COPY only sees the table's own DEFAULTs, so a Python-side one has to opt out.

    cumulative_probability defaults to 0.5 in Python, not in the table, so a frame
    without that column must not take the COPY path.
    """
    table = DBTimedBelief.__table__
    frame = pd.DataFrame(
        columns=["event_start", "belief_horizon", "event_value", "source_id"],
        index=range(N),
    )
    assert classes._should_copy(session, table, frame) is False

    frame["cumulative_probability"] = 0.5
    assert classes._should_copy(session, table, frame) is True
    assert classes._should_copy(session, table, frame.head(1)) is False


def test_copy_upserts_twice_in_one_transaction(time_slot_sensor, test_source_a):
    """The staging table outlives a batch, so it must not leak rows into the next one."""
    DBTimedBelief.add_to_session(
        session,
        frame_of(time_slot_sensor, test_source_a),
        allow_overwrite=True,
        commit_transaction=False,
    )
    DBTimedBelief.add_to_session(
        session,
        frame_of(time_slot_sensor, test_source_a, offset=7.0),
        allow_overwrite=True,
        commit_transaction=True,
    )
    got = read_back(time_slot_sensor, test_source_a)
    assert len(got) == N
    assert got["event_value"].iloc[0] == 7.0


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


def test_copy_handles_sub_microsecond_scale_horizons(time_slot_sensor, test_source_a):
    """A horizon under 1e-4 seconds must not be written in exponent notation.

    str(timedelta.total_seconds()) renders 15 microseconds as "1.5e-05", which
    PostgreSQL's interval parser rejects.
    """
    horizon = timedelta(microseconds=15)
    DBTimedBelief.add_to_session(
        session,
        frame_of(time_slot_sensor, test_source_a, belief_horizon=horizon),
        commit_transaction=True,
    )
    got = read_back(time_slot_sensor, test_source_a)
    assert len(got) == N
    assert (got["belief_time"] - got["event_start"]).nunique() == 1


def test_copy_writes_nan_event_values_as_nan(time_slot_sensor, test_source_a):
    """A NaN event value stays NaN, as it was under the multi-row INSERT.

    An empty CSV field would be NULL, and event_value is NOT NULL.
    """
    values = [float(i) for i in range(N)]
    values[3] = float("nan")
    DBTimedBelief.add_to_session(
        session,
        frame_of(time_slot_sensor, test_source_a, values=values),
        commit_transaction=True,
    )
    got = read_back(time_slot_sensor, test_source_a)
    assert len(got) == N
    assert math.isnan(got["event_value"].iloc[3])
    assert got["event_value"].iloc[4] == 4.0
