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
    )

    # Replace the source
    if replace_source:
        bdf = bdf.reset_index()
        bdf["source"] = test_source_without_initial_data
        bdf = bdf.set_index(
            ["event_start", "belief_time", "source", "cumulative_probability"]
        )

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
    )
    assert len(bdf) == len(new_bdf)


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
