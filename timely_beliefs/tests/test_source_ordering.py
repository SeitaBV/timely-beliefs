"""Tests for BeliefSource strict total ordering (issues #238 and #256).

In-memory BeliefSource equality and hashing use object identity (Python default),
and __lt__ tiebreaks on id(self) when names are equal,
giving a strict total order consistent with identity-based equality.

A database-backed source (BeliefSourceDBMixin) is equal to any object loaded for the same row,
and __lt__ tiebreaks on the row's primary key instead,
so its order is the same in every process.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

import timely_beliefs as tb
from timely_beliefs import BeliefSource, DBBeliefSource
from timely_beliefs.tests import Session, session
from timely_beliefs.tests.test_db_subclassing import RatedSourceInCustomTable


def test_belief_source_lt_different_names():
    """Alphabetic ordering when names differ."""
    s_a = BeliefSource("alpha")
    s_b = BeliefSource("beta")

    assert s_a < s_b
    assert not (s_b < s_a)


def test_belief_source_lt_same_name_tiebreaks_on_identity():
    """Two distinct objects sharing a name must be strictly comparable.

    Before the fix both ``a < b`` and ``b < a`` were False for same-name
    objects, violating the strict total order and causing pandas to produce
    NaN sources.  With id() tiebreaking exactly one of the two must hold.
    """
    s1 = BeliefSource("same")
    s2 = BeliefSource("same")

    # Exactly one must be strictly less than the other.
    assert (s1 < s2) != (
        s2 < s1
    ), "__lt__ must tiebreak on id() so distinct same-name objects are comparable"

    # They are not equal to each other (identity semantics).
    assert s1 != s2


def test_belief_source_total_ordering_consistency():
    """@total_ordering derived comparisons must be consistent with __lt__."""
    s1 = BeliefSource("alpha")
    s2 = BeliefSource("beta")
    s3 = BeliefSource("gamma")

    assert s1 < s2 < s3
    assert s3 > s2 > s1
    assert s1 <= s2
    assert s3 >= s2

    # An object is not strictly less than itself.
    assert not (s1 < s1)
    assert s1 <= s1
    assert s1 >= s1


def test_belief_source_identity_equality_and_hash():
    """__eq__ and __hash__ use object identity (Python default behaviour)."""
    s1 = BeliefSource("Source A")
    s2 = BeliefSource("Source A")  # same name, different object

    assert s1 == s1  # same object is equal to itself
    assert s1 != s2  # different objects, even with the same name

    # Each source is its own unique dict key / set member.
    source_set = {s1, s2}
    assert len(source_set) == 2

    d = {s1: "first", s2: "second"}
    assert d[s1] == "first"
    assert d[s2] == "second"


def test_belief_source_incompatible_type():
    """Comparisons against non-BeliefSource must raise TypeError."""
    s = BeliefSource("x")

    with pytest.raises(TypeError):
        _ = s < "x"

    with pytest.raises(TypeError):
        _ = s < 42


def test_pandas_nan_regression():
    """Regression: pd.concat with same-name distinct sources must not produce NaN.

    Deterministic reproduction from issue #238.  Before the fix this yielded
    4 NaN sources; after the fix it must yield 0.
    """
    sensor = tb.Sensor("x", event_resolution=timedelta(hours=1))
    # 6 objects, only 2 distinct names ("s1" and "s2")
    sources = [tb.BeliefSource("s" + str(i % 2 + 1)) for i in range(6)]

    E = pd.date_range("2025-01-01", periods=5, freq="1h", tz="UTC")
    B = pd.date_range("2024-12-31", periods=3, freq="1h", tz="UTC")
    spec = [
        (4, 1, 0, [0.3, 0.7]),
        (2, 3, 2, [0.5]),
        (1, 3, 2, [0.5]),
        (3, 0, 1, [0.3, 0.7]),
        (1, 3, 1, [0.5]),
        (5, 2, 0, [0.5]),
        (4, 2, 2, [0.3, 0.7]),
    ]
    frames = [
        tb.BeliefsDataFrame(
            [
                tb.TimedBelief(
                    sensor=sensor,
                    source=sources[s],
                    event_start=E[e],
                    belief_time=B[b],
                    cumulative_probability=cp,
                    event_value=1.0,
                )
                for cp in cps
            ]
        )
        for s, e, b, cps in spec
    ]
    bdf = pd.concat(frames)
    nan_sources = sum(1 for t in bdf.index if not isinstance(t[2], tb.BeliefSource))
    assert nan_sources == 0, f"Expected 0 NaN sources, got {nan_sources}"


def flushed_sources(*names: str) -> list[DBBeliefSource]:
    """Database-backed sources with the given names, flushed in the given order, so their primary keys ascend in that order.

    The objects are created in reverse order,
    so that their memory addresses cannot be mistaken for their primary keys.
    """
    sources = [DBBeliefSource(name) for name in reversed(names)][::-1]
    for source in sources:
        session.add(source)
        session.flush()
    return sources


def test_db_belief_source_lt_different_names(db):
    """Alphabetic ordering when names differ, whatever the primary keys."""
    s_b, s_a = flushed_sources("beta", "alpha")

    assert s_a < s_b
    assert not (s_b < s_a)


def test_db_belief_source_lt_same_name_orders_by_id(db):
    """Database-backed sources sharing a name order by primary key (issue #256)."""
    sources = flushed_sources(*["same"] * 10)

    assert sorted(reversed(sources)) == sources
    assert [s.id for s in sorted(reversed(sources))] == sorted(s.id for s in sources)


def test_db_belief_source_equality_and_hash_per_row(db):
    """Two objects loaded for one row, in different sessions, are equal, hash equal and neither orders first."""
    (s1,) = flushed_sources("same")
    session.commit()
    other_session = Session()
    try:
        s1_again = other_session.get(DBBeliefSource, s1.id)
        assert s1_again is not s1
        s2 = DBBeliefSource("same")
        other_session.add(s2)
        other_session.flush()

        assert s1 == s1_again
        assert s1 != s2
        assert not (s1 < s1_again)
        assert not (s1_again < s1)
        assert s1 <= s1_again
        assert s1 >= s1_again

        assert hash(s1) == hash(s1_again)
        assert len({s1, s1_again}) == 1
        assert len({s1, s2}) == 2
    finally:
        other_session.rollback()
        other_session.close()


def test_db_belief_source_equality_and_hash_do_not_load(db):
    """Expired, detached sources can still be compared for equality and hashed, so neither loads from the database."""
    s1, s2 = flushed_sources("same", "same")
    session.commit()  # expires both
    session.close()  # detaches both

    assert s1 != s2
    assert s1 == s1
    assert len({s1, s2}) == 2


def test_db_belief_source_different_tables_are_different_sources(db):
    """Sources from different tables are different sources, even with the same name and primary key."""
    (s,) = flushed_sources("same")
    custom = RatedSourceInCustomTable(name="same")
    custom.id = s.id
    session.add(custom)
    session.flush()
    assert s.id == custom.id

    assert s != custom
    assert (s < custom) != (custom < s)


def test_db_belief_source_unflushed_ordering_and_equality(db):
    """Sources that were not flushed yet only equal themselves, and sort after flushed ones, tiebreaking on identity."""
    (s_flushed,) = flushed_sources("same")
    s_unflushed1 = DBBeliefSource("same")
    s_unflushed2 = DBBeliefSource("same")

    assert s_flushed < s_unflushed1
    assert not (s_unflushed1 < s_flushed)
    assert s_flushed != s_unflushed1

    assert s_unflushed1 == s_unflushed1
    assert s_unflushed1 != s_unflushed2
    assert (s_unflushed1 < s_unflushed2) != (s_unflushed2 < s_unflushed1)


def test_db_belief_source_hash_stability_on_flush(db):
    """A source used as a dict key before it is flushed is still found after the flush and a commit."""
    s = DBBeliefSource("Source A")
    d = {s: "created"}

    session.add(s)
    session.flush()
    session.commit()

    assert s in d
    assert d[s] == "created"


def test_db_belief_source_mixed_with_in_memory_source(db):
    """A database-backed source and an in-memory source with the same name are different, and order consistently."""
    (db_s,) = flushed_sources("same")
    mem_s = BeliefSource("same")

    assert db_s != mem_s
    assert mem_s != db_s
    assert db_s < mem_s
    assert not (mem_s < db_s)
