import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from timely_beliefs import DBSensor, DBTimedBelief
from timely_beliefs.tests import session


@pytest.mark.parametrize("replace_source", [False, True])
@pytest.mark.parametrize("bulk_save_objects", [False, True])
def test_adding_to_session(
    replace_source: bool,
    bulk_save_objects: bool,
    time_slot_sensor: DBSensor,
    rolling_day_ahead_beliefs_about_time_slot_events,
    test_source_a,
    test_source_b,
    test_source_without_initial_data,
):

    # Retrieve some data from the database
    bdf = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        source=test_source_b,
        most_recent_beliefs_only=True,
        use_materialized_view=True,
        most_recent_beliefs_mview="bla",
    )

    # Replace the source
    if replace_source:
        bdf = bdf._replace_multi_index_level("source", test_source_without_initial_data)

    # Overwriting the data should succeed, at least if we expunge everything from the session
    DBTimedBelief.add_to_session(
        session,
        bdf,
        expunge_session=True,
        allow_overwrite=True,
        bulk_save_objects=bulk_save_objects,
        commit_transaction=True,
    )
    new_bdf = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        source=test_source_without_initial_data if replace_source else test_source_b,
        most_recent_beliefs_only=True,
        use_materialized_view=True,
        most_recent_beliefs_mview="bla",
    )
    assert len(bdf) == len(new_bdf)

    # A single more recent belief
    if not bdf.empty:
        more_recent_belief = bdf.head(1)._replace_multi_index_level("belief_time", bdf.head(1).belief_times + pd.Timedelta(minutes=1))
        more_recent_belief["event_value"] = 1000
        DBTimedBelief.add_to_session(
            session,
            more_recent_belief,
            expunge_session=True,
            allow_overwrite=True,
            bulk_save_objects=bulk_save_objects,
            commit_transaction=True,
        )
        newer_bdf = DBTimedBelief.search_session(
            session=session,
            sensor=time_slot_sensor,
            source=test_source_without_initial_data if replace_source else test_source_b,
            most_recent_beliefs_only=True,
            use_materialized_view=True,
            most_recent_beliefs_mview="bla",
        )
        assert newer_bdf.event_value[0] == 1000


@pytest.mark.parametrize("bulk_save_objects", [False, True])
def test_fail_adding_to_session(
    bulk_save_objects: bool,
    time_slot_sensor: DBSensor,
    rolling_day_ahead_beliefs_about_time_slot_events,
):

    # Retrieve some data from the database
    bdf = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
    )

    # Attempting to save the same data should fail, even if we expunge everything from the session
    with pytest.raises(IntegrityError):
        DBTimedBelief.add_to_session(
            session,
            bdf,
            expunge_session=True,
            bulk_save_objects=bulk_save_objects,
            commit_transaction=True,
        )
