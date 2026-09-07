"""Tests for BeliefSource strict total ordering (issue #238).

BeliefSource equality and hashing use object identity (Python default).
__lt__ tiebreaks on id(self) when names are equal, giving a strict total order
consistent with identity-based equality.
"""

from datetime import timedelta

import pandas as pd
import pytest

import timely_beliefs as tb
from timely_beliefs import BeliefSource


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
