from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pytz import utc
from sqlalchemy import and_, select, text

import timely_beliefs.beliefs.queries as query_utils
import timely_beliefs.beliefs.utils as belief_utils
from timely_beliefs import (
    BeliefsDataFrame,
    DBBeliefSource,
    DBSensor,
    DBTimedBelief,
    TimedBelief,
)
from timely_beliefs.beliefs.classes import _custom_criteria_are_group_constant
from timely_beliefs.tests import session


@pytest.fixture(scope="function")
def beliefs_recorded_at_unique_knowledge_time(
    unique_knowledge_time_sensor: DBSensor, test_source_a: DBBeliefSource
) -> list[DBTimedBelief]:
    """Define beliefs about a future event at its unique knowledge time (e.g. a publication date)."""
    beliefs = [
        DBTimedBelief(
            source=test_source_a,
            sensor=unique_knowledge_time_sensor,
            event_value=10 + i,
            belief_time=datetime(1990, 5, 10, 0, tzinfo=utc),
            event_start=datetime(1990, 6, 1 + i, 0, tzinfo=utc),
        )
        for i in range(2)
    ]
    session.add_all(beliefs)
    return beliefs


@pytest.mark.parametrize("use_mview", [False, True])
def test_query_belief_for_sensor_with_unique_knowledge_time(
    unique_knowledge_time_sensor: DBSensor,
    beliefs_recorded_at_unique_knowledge_time: list[DBTimedBelief],
    use_mview: bool,
):
    """Test query of sensor with a unique knowledge time, in combination with a belief time window."""
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=unique_knowledge_time_sensor,
        beliefs_after=pd.Timestamp("1990-04-01 00:00Z"),
        beliefs_before=pd.Timestamp("1990-06-01 00:00Z"),
        use_materialized_view=use_mview,
    ).convert_index_from_belief_time_to_horizon()
    assert belief_df.belief_horizons[0] == timedelta(0)
    assert belief_df.belief_horizons[1] == timedelta(0)
    assert belief_df.knowledge_horizons[0] == timedelta(days=22)
    assert belief_df.knowledge_horizons[1] == timedelta(days=23)


@pytest.fixture(scope="function")
def day_ahead_belief_about_ex_ante_economical_event(
    ex_ante_economics_sensor: DBSensor, test_source_a: DBBeliefSource
):
    """Define day-ahead belief about an ex-ante economical event."""
    belief = DBTimedBelief(
        source=test_source_a,
        sensor=ex_ante_economics_sensor,
        event_value=10,
        belief_time=datetime(2018, 1, 1, 10, tzinfo=utc),
        event_start=datetime(2018, 1, 2, 22, 45, tzinfo=utc),
    )
    session.add(belief)
    return belief


@pytest.fixture(scope="function")
def multiple_day_ahead_beliefs_about_ex_ante_economical_event(
    ex_ante_economics_sensor: DBSensor, test_source_a: DBBeliefSource
):
    """Define multiple day-ahead beliefs about an ex-ante economical event."""
    n = 10
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    beliefs = []
    for i in range(n):
        belief = DBTimedBelief(
            source=test_source_a,
            sensor=ex_ante_economics_sensor,
            event_value=10 + i,
            belief_time=ex_ante_economics_sensor.knowledge_time(event_start)
            - timedelta(hours=i + 1),
            event_start=event_start,
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


@pytest.fixture(scope="function")
def multiple_probabilistic_day_ahead_beliefs_about_ex_ante_economical_event(
    ex_ante_economics_sensor: DBSensor,
    ex_ante_economics_sensor_b: DBSensor,
    test_source_a: DBBeliefSource,
):
    """Define multiple probabilistic day-ahead beliefs about an ex-ante economical event on two sensors."""
    n = 10  # number of belief times
    np = 2  # number of probabilities per belief
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    beliefs = []
    for sensor in [ex_ante_economics_sensor, ex_ante_economics_sensor_b]:
        for i in range(n):
            for j in range(np):
                if sensor == ex_ante_economics_sensor and i == 0:
                    # Skip to ensure that sensor B has more recent beliefs
                    # This is to test the most_recent_beliefs_only parameter for a query on the other sensor
                    continue
                belief = DBTimedBelief(
                    source=test_source_a,
                    sensor=sensor,
                    event_value=10 + i - j / 100,
                    belief_time=ex_ante_economics_sensor.knowledge_time(event_start)
                    - timedelta(hours=i + 1),
                    event_start=event_start,
                    cumulative_probability=0.5 * (1 - j / np),
                )
                session.add(belief)
                beliefs.append(belief)
    return beliefs


@pytest.fixture(scope="function")
def multiple_day_after_beliefs_about_ex_ante_economical_event(
    ex_ante_economics_sensor: DBSensor, test_source_a: DBBeliefSource
):
    """Define multiple day-after beliefs about an ex-ante economical event."""
    n = 10
    event_start = datetime(2025, 1, 2, 23, 00, tzinfo=utc)
    beliefs = []
    for i in range(n):
        belief = DBTimedBelief(
            source=test_source_a,
            sensor=ex_ante_economics_sensor,
            event_value=10 + i,
            belief_time=ex_ante_economics_sensor.knowledge_time(event_start)
            + timedelta(hours=i + 1),
            event_start=event_start,
        )
        session.add(belief)
        beliefs.append(belief)
    return beliefs


@pytest.mark.parametrize("use_mview", [False, True])
def test_query_belief_with_empty_source_list(
    ex_ante_economics_sensor: DBSensor,
    day_ahead_belief_about_ex_ante_economical_event: DBTimedBelief,
    use_mview: bool,
):
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        source=[],
        use_materialized_view=use_mview,
    )
    assert belief_df.empty


@pytest.mark.parametrize(
    "beliefs_after, beliefs_before, expected_length",
    [
        (None, None, 1),  # Just one belief set up for this sensor
        (None, datetime(2017, 1, 1, 10, tzinfo=utc), 0),  # No beliefs a year earlier
        (datetime(2018, 1, 3, 10, tzinfo=utc), None, 0),  # No beliefs 2 months later
        (datetime(2018, 1, 1, 10, tzinfo=utc), None, 1),  # One belief after 10am UTC
        (None, datetime(2018, 1, 1, 9, tzinfo=utc), 0),  # No beliefs an hour earlier
        (datetime(2018, 1, 1, 13, tzinfo=utc), None, 0),  # No beliefs after 1pm UTC
    ],
)
@pytest.mark.parametrize("use_mview", [False, True])
def test_query_belief_by_belief_time(
    ex_ante_economics_sensor: DBSensor,
    day_ahead_belief_about_ex_ante_economical_event: DBTimedBelief,
    beliefs_after,
    beliefs_before,
    expected_length,
    use_mview: bool,
):
    bdf = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        beliefs_after=beliefs_after,
        beliefs_before=beliefs_before,
        use_materialized_view=use_mview,
    )
    assert len(bdf) == expected_length

    if expected_length == 1:
        # By calling a pandas Series for its values we lose the timezone (a pandas bug still present in version 0.23.4)
        # This next test warns us when it has been fixed (if it fails, just replace != with ==).
        assert bdf.knowledge_times.values[0] != datetime(2018, 1, 1, 11, 0, tzinfo=utc)
        # And this test is just a workaround to test what we wanted to test.
        assert pd.Timestamp(bdf.knowledge_times.values[0]) == pd.Timestamp(
            datetime(2018, 1, 1, 11, 0)
        )


@pytest.mark.parametrize("use_mview", [False, True])
def test_query_belief_history(
    ex_ante_economics_sensor: DBSensor,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
    use_mview: bool,
):
    df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        use_materialized_view=use_mview,
    )
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    df2 = df.belief_history(event_start).sort_index(
        level="belief_time", ascending=False
    )
    assert len(df2) == 10
    assert (df2["event_value"].values == np.arange(10, 20)).all()
    df3 = df.belief_history(
        event_start,
        belief_time_window=(
            datetime(2025, 1, 1, 7, tzinfo=utc),
            datetime(2025, 1, 1, 9, tzinfo=utc),
        ),
    )
    assert len(df3) == 3
    df4 = df.belief_history(
        event_start, belief_horizon_window=(timedelta(weeks=-10), timedelta(hours=2.5))
    )  # Only 2 beliefs were formed up to 2.5 hours before knowledge_time, and none after
    assert len(df4) == 2


@pytest.mark.parametrize("use_mview", [False, True])
def test_query_rolling_horizon(
    time_slot_sensor: DBSensor,
    rolling_day_ahead_beliefs_about_time_slot_events,
    use_mview: bool,
):
    """Make sure that a rolling viewpoint includes the most recent beliefs."""
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(2050, 1, 1, 14, tzinfo=utc),
        use_materialized_view=use_mview,
    )  # select beliefs up until 14 o'clock (4 events have 2 beliefs, and 1 event has 1 belief)
    rolling_df = belief_df.rolling_viewpoint(
        belief_horizon=timedelta(hours=49)
    )  # select only the five older beliefs
    assert len(rolling_df) == 5  # 5 older (made at 10,11,12,13,14 o'clock)
    assert (rolling_df["event_value"].values == np.arange(101, 106)).all()
    rolling_df = belief_df.rolling_viewpoint(
        belief_horizon=timedelta(hours=48)
    )  # select the most recent beliefs
    assert (
        len(rolling_df) == 5
    )  # 4 more recent (made at 11,12,13,14 o'clock), 1 older (at 14 o'clock, because the 15 o'clock is missing)
    assert (rolling_df["event_value"].values == [11, 12, 13, 14, 105]).all()


@pytest.mark.parametrize("use_mview", [False, True])
def test_query_fixed_horizon(
    time_slot_sensor: DBSensor,
    rolling_day_ahead_beliefs_about_time_slot_events,
    test_source_a,
    test_source_b,
    use_mview: bool,
):
    belief_time = datetime(2050, 1, 1, 11, tzinfo=utc)
    df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(2050, 1, 1, 15, tzinfo=utc),
        source=[test_source_a, test_source_b],
        use_materialized_view=use_mview,
    )
    df2 = df.fixed_viewpoint(belief_time=belief_time)
    assert len(df2) == 2
    assert df2[df2.index.get_level_values("belief_time") > belief_time].empty
    assert (df2["event_value"].values == np.array([11, 102])).all()
    df3 = df.fixed_viewpoint(
        belief_time_window=(belief_time - timedelta(minutes=1), belief_time)
    )
    assert len(df3) == 2  # The belief formed at 10 AM is now considered too old
    assert (df3["event_value"].values == np.array([11, 102])).all()


@pytest.mark.parametrize("use_mview", [False, True])
def test_downsample(
    time_slot_sensor, rolling_day_ahead_beliefs_about_time_slot_events, use_mview: bool
):
    """Downsample from 15 minutes to 2 hours."""
    new_resolution = timedelta(hours=2)
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(2100, 1, 1, 13, tzinfo=utc),
        use_materialized_view=use_mview,
    )
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution


@pytest.mark.parametrize("use_mview", [False, True])
def test_upsample(
    time_slot_sensor, rolling_day_ahead_beliefs_about_time_slot_events, use_mview: bool
):
    """Upsample from 15 minutes to 5 minutes."""
    new_resolution = timedelta(minutes=5)
    belief_df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(2100, 1, 1, 13, tzinfo=utc),
        use_materialized_view=use_mview,
    )
    belief_df = belief_df.resample_events(new_resolution)
    assert belief_df.sensor.event_resolution == timedelta(minutes=15)
    assert belief_df.event_resolution == new_resolution


def _test_empty_frame(time_slot_sensor, use_mview: bool = False):
    """pandas GH30517"""
    bdf = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        beliefs_before=datetime(1900, 1, 1, 13, tzinfo=utc),
        use_materialized_view=use_mview,
    )
    assert bdf.empty  # no data expected
    assert pd.api.types.is_datetime64_dtype(bdf.index.get_level_values("belief_time"))
    bdf = bdf.convert_index_from_belief_time_to_horizon()
    assert pd.api.types.is_timedelta64_dtype(
        bdf.index.get_level_values("belief_horizon")
    )  # dtype of belief_horizon is timedelta64[ns], so the minimum horizon on an empty BeliefsDataFrame is NaT instead of NaN


@pytest.mark.parametrize("use_mview", [False, True])
def test_search_by_sensor_id(
    ex_ante_economics_sensor: DBSensor,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
    use_mview: bool,
):
    """Check db query by sensor id, against query by sensor instance, for a non-empty dataset."""

    # Query all beliefs for this sensor, using sensor instance (our reference)
    df_by_instance = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=False,
        use_materialized_view=use_mview,
    )

    # Query all beliefs for this sensor, using sensor id (our test)
    df_by_id = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor.id,
        most_recent_beliefs_only=False,
        use_materialized_view=use_mview,
    )
    assert not df_by_id.empty
    pd.testing.assert_frame_equal(df_by_id, df_by_instance)


@pytest.mark.parametrize("use_mview", [False, True])
def test_select_most_recent_deterministic_beliefs(
    ex_ante_economics_sensor: DBSensor,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
    multiple_day_after_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
    use_mview: bool,
    refresh_mview,
):
    """Check db query filters for most recent beliefs, most recent events, and both at once."""
    refresh_mview()

    # Query all beliefs for this sensor
    df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=False,
        use_materialized_view=use_mview,
    )

    # Most recent beliefs selected after query (our reference)
    df_recent_beliefs_after_query = belief_utils.select_most_recent_belief(df)

    # Most recent beliefs selected within query (our test)
    df_recent_beliefs_within_query = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        use_materialized_view=use_mview,
    )
    pd.testing.assert_frame_equal(
        df_recent_beliefs_within_query, df_recent_beliefs_after_query
    )

    # Most recent events selected after query (our reference)
    df_recent_events_after_query = df[
        df.index.get_level_values("event_start") == df.event_starts.max()
    ]

    # Most recent events selected within query (our test)
    df_recent_events_within_query = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_events_only=True,
        use_materialized_view=use_mview,
    )
    pd.testing.assert_frame_equal(
        df_recent_events_within_query, df_recent_events_after_query
    )

    # Most recent beliefs and most recent events selected after query (our reference)
    df_recent_both_after_query = df_recent_beliefs_after_query[
        df_recent_beliefs_after_query.index.get_level_values("event_start")
        == df_recent_beliefs_after_query.event_starts.max()
    ]

    # Most recent beliefs and most recent events selected within query (our test)
    df_recent_both_within_query = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        most_recent_events_only=True,
        use_materialized_view=use_mview,
    )
    pd.testing.assert_frame_equal(
        df_recent_both_within_query, df_recent_both_after_query
    )


@pytest.mark.parametrize("use_mview", [False, True])
def test_select_most_recent_beliefs_with_event_window(
    time_slot_sensor: DBSensor,
    rolling_day_ahead_beliefs_about_time_slot_events: list[DBTimedBelief],
    use_mview: bool,
    refresh_mview,
):
    """Check that event window filters (event_starts_after/event_ends_before) are
    respected when selecting most recent beliefs through the materialized view.

    The mview subquery only carries a sensor_id filter (plus, for the live-tail
    variant, an event_start < cutoff filter). Without pushing the caller's event
    window down into that subquery too, the join would still return the right rows
    (the outer query's own event_start filter would exclude the rest), but at the
    cost of scanning the entire view. This test protects correctness of the
    pushed-down filter, i.e. that results are unaffected by the optimization.
    """
    refresh_mview()

    event_starts_after = datetime(2050, 1, 3, 15, tzinfo=utc)
    event_ends_before = datetime(2050, 1, 3, 19, tzinfo=utc)

    # Reference: compute over the full result set, without database-side filtering
    full_df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        most_recent_beliefs_only=False,
        use_materialized_view=use_mview,
    )
    reference_df = belief_utils.select_most_recent_belief(full_df)
    reference_df = reference_df[
        (reference_df.event_starts >= event_starts_after)
        & (reference_df.event_ends <= event_ends_before)
    ]
    assert not reference_df.empty

    # Test: apply the event window filters within the query itself
    df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        most_recent_beliefs_only=True,
        event_starts_after=event_starts_after,
        event_ends_before=event_ends_before,
        use_materialized_view=use_mview,
    )
    pd.testing.assert_frame_equal(df, reference_df)


@pytest.mark.parametrize("use_mview", [False, True])
def test_select_most_recent_probabilistic_beliefs(
    ex_ante_economics_sensor: DBSensor,
    multiple_probabilistic_day_ahead_beliefs_about_ex_ante_economical_event: list[
        DBTimedBelief
    ],
    use_mview: bool,
    refresh_mview,
):
    refresh_mview()
    df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=False,
        use_materialized_view=use_mview,
    )
    most_recent_df = belief_utils.select_most_recent_belief(df)
    df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        use_materialized_view=use_mview,
    )
    pd.testing.assert_frame_equal(df, most_recent_df)


@pytest.mark.parametrize(
    "event_values, expected_unchanged_event_values",
    [
        ([10, 10, 10, 9, 9, 9], [None, 10, 10, None, 9, 9]),
        ([10, 11, 10, 9, 9.5, 9], [None, None, None, None, None, None]),
        ([10, 9, 10, 10, 9.5], [None, None, None, 10, None]),
        ([10, 10, 9, 10, 10], [None, 10, None, None, 10]),
    ],
)
@pytest.mark.parametrize("use_mview", [False, True])
def test_query_unchanged_beliefs(
    event_values, expected_unchanged_event_values, use_mview: bool
):
    sensor = session.execute(select(DBSensor).limit(1)).scalar()
    source = session.execute(select(DBBeliefSource).limit(1)).scalar()
    beliefs = [
        DBTimedBelief(
            sensor=sensor,
            source=source,
            event_value=v,
            event_start=pd.Timestamp("2022-01-26 13:50+01:00").to_pydatetime(),
            belief_time=pd.Timestamp("2022-01-05 13:50+01:00").to_pydatetime()
            + i * timedelta(hours=1),
        )
        for i, v in enumerate(event_values)
    ]
    expected_unchanged_beliefs = BeliefsDataFrame(
        [
            TimedBelief(
                sensor=sensor,
                source=source,
                event_value=v,
                event_start=pd.Timestamp("2022-01-26 13:50+01:00").to_pydatetime(),
                belief_time=pd.Timestamp("2022-01-05 13:50+01:00").to_pydatetime()
                + i * timedelta(hours=1),
            )
            for i, v in enumerate(expected_unchanged_event_values)
            if v is not None
        ]
    )
    session.add_all(beliefs)
    all_beliefs_query = select(DBTimedBelief).filter(
        DBTimedBelief.sensor == sensor, DBTimedBelief.source == source
    )
    q = query_utils.query_unchanged_beliefs(
        session=session,
        query=all_beliefs_query,
    )
    unchanged_beliefs = BeliefsDataFrame(session.scalars(q).all())
    pd.testing.assert_frame_equal(unchanged_beliefs, expected_unchanged_beliefs)


@pytest.mark.parametrize("use_mview", [False, True])
def test_most_recent_beliefs_with_horizon_filters_bypass_mview(
    ex_ante_economics_sensor: DBSensor,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
    use_mview: bool,
):
    """Belief timing filters cannot be applied to the materialized view.

    The view caches the global minimum belief horizon, so the search should bypass the view
    (here: deliberately left unrefreshed, i.e. empty) and use the beliefs table instead.
    """
    reference_df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        horizons_at_least=timedelta(hours=5),
        use_materialized_view=False,
    )
    assert not reference_df.empty
    df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        horizons_at_least=timedelta(hours=5),
        use_materialized_view=use_mview,
    )
    pd.testing.assert_frame_equal(df, reference_df)


@pytest.mark.parametrize("use_mview", [True])
def test_mview_live_tail_includes_events_recorded_after_refresh(
    ex_ante_economics_sensor: DBSensor,
    test_source_a: DBBeliefSource,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
    use_mview: bool,
    refresh_mview,
):
    """Events recorded after the last view refresh should still show up when passing a cutoff."""
    refresh_mview()

    # Record a belief about a new event, without refreshing the view
    mview_cutoff = datetime(2025, 1, 3, 0, 0, tzinfo=utc)
    new_event_start = datetime(2025, 1, 3, 22, 45, tzinfo=utc)
    session.add(
        DBTimedBelief(
            source=test_source_a,
            sensor=ex_ante_economics_sensor,
            event_value=100,
            belief_time=ex_ante_economics_sensor.knowledge_time(new_event_start)
            - timedelta(hours=1),
            event_start=new_event_start,
        )
    )

    # Without a cutoff, the view is trusted for all events, so the new event is missed
    df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        use_materialized_view=True,
    )
    assert new_event_start not in df.index.get_level_values("event_start")

    # With a cutoff, the new event is looked up in the beliefs table, and results are complete
    df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        use_materialized_view=True,
        mview_cutoff=mview_cutoff,
    )
    reference_df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        use_materialized_view=False,
    )
    assert new_event_start in df.index.get_level_values("event_start")
    pd.testing.assert_frame_equal(df, reference_df)


@pytest.mark.parametrize("use_mview", [True])
def test_mview_returns_stale_most_recent_beliefs_until_refreshed(
    ex_ante_economics_sensor: DBSensor,
    test_source_a: DBBeliefSource,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
    use_mview: bool,
    refresh_mview,
):
    """Belief revisions recorded after the last view refresh only show up after the next refresh.

    This documents the staleness semantics of using the materialized view:
    for events starting before the cutoff, the view determines which belief is the most recent one.
    """
    refresh_mview()

    # Record a more recent belief about the existing event, without refreshing the view
    event_start = datetime(2025, 1, 2, 22, 45, tzinfo=utc)
    session.add(
        DBTimedBelief(
            source=test_source_a,
            sensor=ex_ante_economics_sensor,
            event_value=999,
            belief_time=ex_ante_economics_sensor.knowledge_time(event_start)
            - timedelta(minutes=30),
            event_start=event_start,
        )
    )

    def search(use_materialized_view: bool) -> BeliefsDataFrame:
        return DBTimedBelief.search_session(
            session=session,
            sensor=ex_ante_economics_sensor,
            most_recent_beliefs_only=True,
            use_materialized_view=use_materialized_view,
            mview_cutoff=datetime(2025, 1, 3, 0, 0, tzinfo=utc),
        )

    # The view still reports the previously most recent belief (stale, but present)
    assert search(use_materialized_view=True)["event_value"].tolist() == [10]

    # The beliefs table knows better
    assert search(use_materialized_view=False)["event_value"].tolist() == [999]

    # After a refresh, the view catches up
    refresh_mview()
    assert search(use_materialized_view=True)["event_value"].tolist() == [999]


@pytest.mark.parametrize("use_mview", [False, True])
def test_most_recent_beliefs_with_belief_time_filter_bypasses_mview(
    time_slot_sensor: DBSensor,
    rolling_day_ahead_beliefs_about_time_slot_events: list[DBTimedBelief],
    use_mview: bool,
    refresh_mview,
):
    """A beliefs_before/beliefs_after filter cannot be applied to the materialized view.

    The view caches the global minimum belief horizon (per event, per source), computed
    without regard to any belief-time window. If we naively joined the (unfiltered) view
    and then filtered by belief time afterwards, we could silently drop events whose
    globally-most-recent belief falls outside the window, even though an earlier belief
    (within the window) exists for that same event.

    time_slot_sensor's knowledge horizon function is ex_post (the default), so it is
    eligible for the most_recent_beliefs_only mview branch even with beliefs_before set;
    this test therefore protects the fix at the mview-eligibility level (not just via the
    post-processing fallback for knowledge functions incompatible with the mview branch
    altogether).
    """
    refresh_mview()

    beliefs_before = datetime(2050, 1, 1, 14, tzinfo=utc)

    # Reference: compute over the full result set, without database-side filtering
    full_df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        most_recent_beliefs_only=False,
        use_materialized_view=use_mview,
    )
    full_df = full_df[
        full_df.index.get_level_values("belief_time") <= beliefs_before
    ]
    reference_df = belief_utils.select_most_recent_belief(full_df)
    assert not reference_df.empty

    # Test: apply the belief_before filter within the query itself
    df = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        most_recent_beliefs_only=True,
        beliefs_before=beliefs_before,
        use_materialized_view=use_mview,
    )
    pd.testing.assert_frame_equal(df, reference_df)


@pytest.mark.parametrize("use_mview", [False, True])
def test_most_recent_beliefs_with_row_level_custom_criterion_bypasses_mview(
    ex_ante_economics_sensor: DBSensor,
    multiple_day_ahead_beliefs_about_ex_ante_economical_event: list[DBTimedBelief],
    use_mview: bool,
    refresh_mview,
):
    """A custom filter criterion whose truth value can vary within a single
    (event_start, source_id) group (e.g. a criterion on event_value) cannot be applied to
    the materialized view, for the same reason belief-time filters cannot: the mview only
    caches the *globally* most recent belief horizon, so applying such a criterion after
    the join could silently drop an event whose most recent belief fails the criterion,
    even though an earlier belief for that event would have passed it.

    The fixture creates 10 beliefs about a single event, with event_value 10 (most
    recent belief) up to 19 (oldest belief). Filtering out event_value == 10 forces the
    correct answer to be the second most recent belief (event_value 11); a mview that
    (incorrectly) caches the unfiltered global minimum belief horizon and applies the
    criterion only after the join would instead drop the event entirely.
    """
    refresh_mview()

    # Reference: compute over the full result set, without database-side filtering
    full_df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=False,
        use_materialized_view=use_mview,
    )
    full_df = full_df[full_df["event_value"] > 10]
    reference_df = belief_utils.select_most_recent_belief(full_df)
    assert not reference_df.empty
    assert reference_df["event_value"].tolist() == [11]

    df = DBTimedBelief.search_session(
        session=session,
        sensor=ex_ante_economics_sensor,
        most_recent_beliefs_only=True,
        custom_filter_criteria=[DBTimedBelief.event_value > 10],
        use_materialized_view=use_mview,
    )
    pd.testing.assert_frame_equal(df, reference_df)


def test_custom_criteria_are_group_constant_helper():
    """Unit tests for the conservative introspection helper that decides whether custom
    filter criteria are safe to apply after (rather than before) the materialized view's
    MIN(belief_horizon) aggregation.
    """
    beliefs_table = DBTimedBelief.__table__
    source_table = DBBeliefSource.__table__

    # A criterion on a column of another table (e.g. the source class, joined via
    # source_id) is group-constant and safe.
    assert _custom_criteria_are_group_constant(
        [source_table.c.name == "Source A"], beliefs_table
    )

    # Criteria on the beliefs table's own group-constant columns are safe.
    assert _custom_criteria_are_group_constant(
        [DBTimedBelief.event_start == datetime(2020, 1, 1, tzinfo=utc)],
        beliefs_table,
    )
    assert _custom_criteria_are_group_constant(
        [DBTimedBelief.source_id == 1], beliefs_table
    )

    # Criteria on beliefs-table columns that can vary within a group are unsafe.
    assert not _custom_criteria_are_group_constant(
        [DBTimedBelief.event_value < 1000], beliefs_table
    )
    assert not _custom_criteria_are_group_constant(
        [DBTimedBelief.belief_horizon > timedelta(0)], beliefs_table
    )

    # A text() clause cannot be introspected, so it is conservatively unsafe.
    assert not _custom_criteria_are_group_constant([text("1=1")], beliefs_table)

    # A mixed and_() is unsafe if any part of it is unsafe.
    assert not _custom_criteria_are_group_constant(
        [
            and_(
                DBTimedBelief.source_id == 1,
                DBTimedBelief.event_value < 1000,
            )
        ],
        beliefs_table,
    )

    # ... but safe if every part of it is safe.
    assert _custom_criteria_are_group_constant(
        [
            and_(
                DBTimedBelief.source_id == 1,
                DBTimedBelief.event_start == datetime(2020, 1, 1, tzinfo=utc),
            )
        ],
        beliefs_table,
    )
