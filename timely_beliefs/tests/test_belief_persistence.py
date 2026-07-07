import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from timely_beliefs import DBBeliefSource, DBSensor, DBTimedBelief
from timely_beliefs.tests import Session, session


@pytest.mark.parametrize("replace_source", [False, True])
@pytest.mark.parametrize("bulk_save_objects", [False, True])
@pytest.mark.parametrize("use_mview", [False, True])
def test_adding_to_session(
    replace_source: bool,
    bulk_save_objects: bool,
    use_mview: bool,
    time_slot_sensor: DBSensor,
    rolling_day_ahead_beliefs_about_time_slot_events,
    test_source_a,
    test_source_b,
    test_source_without_initial_data,
    refresh_mview,
):
    # Refresh, so a materialized view search actually returns the beliefs set up above
    # (otherwise the use_mview=True case would trivially pass, having found nothing to check)
    refresh_mview()

    # Retrieve some data from the database
    bdf = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        source=test_source_b,
        most_recent_beliefs_only=True,
        use_materialized_view=use_mview,
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
    # Refresh again, so a materialized view search sees the overwritten beliefs
    # (relevant when the source was replaced, because the view knows nothing about the new source yet)
    refresh_mview()
    new_bdf = DBTimedBelief.search_session(
        session=session,
        sensor=time_slot_sensor,
        source=test_source_without_initial_data if replace_source else test_source_b,
        most_recent_beliefs_only=True,
        use_materialized_view=use_mview,
    )
    assert len(bdf) == len(new_bdf)

    # Add a single more recent belief about the first event in bdf,
    # with an event value (1000) distinct from any value set up so far
    if not bdf.empty:
        more_recent_belief = bdf.head(1)._replace_multi_index_level(
            "belief_time", bdf.head(1).belief_times + pd.Timedelta(minutes=1)
        )
        more_recent_belief["event_value"] = 1000
        revised_event_start = more_recent_belief.event_starts[0]
        DBTimedBelief.add_to_session(
            session,
            more_recent_belief,
            expunge_session=True,
            allow_overwrite=True,
            bulk_save_objects=bulk_save_objects,
            commit_transaction=True,
        )

        # Refresh again, so a materialized view search sees the newly added belief,
        # and commit, so the refresh releases its lock on the view
        # before we search from another session
        refresh_mview()
        session.commit()

        time_slot_sensor_id = time_slot_sensor.id
        test_source_without_initial_data_id = test_source_without_initial_data.id
        test_source_b_id = test_source_b.id
        session.close()
        new_session = Session()
        try:
            time_slot_sensor = new_session.get(DBSensor, time_slot_sensor_id)
            test_source_without_initial_data = new_session.get(
                DBBeliefSource, test_source_without_initial_data_id
            )
            test_source_b = new_session.get(DBBeliefSource, test_source_b_id)

            newer_bdf = DBTimedBelief.search_session(
                session=new_session,
                sensor=time_slot_sensor,
                source=(
                    test_source_without_initial_data
                    if replace_source
                    else test_source_b
                ),
                most_recent_beliefs_only=True,
                use_materialized_view=use_mview,
            )
            revised_beliefs = newer_bdf[
                newer_bdf.index.get_level_values("event_start") == revised_event_start
            ]
            assert (
                len(revised_beliefs) == 1
            ), "expected exactly one most recent belief about the revised event"
            assert (
                revised_beliefs.event_value.iloc[0] == 1000
            ), "expected the newly added belief (proving that a fresh session sees the belief revision, also when searching via the materialized view)"
        finally:
            # Close even upon failure, so an open transaction can't block
            # the materialized view from being dropped/created by the next test
            new_session.close()


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
