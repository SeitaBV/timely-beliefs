"""Tests for the IntTimedelta type decorator and timedelta/seconds conversion utilities."""

from datetime import timedelta

import pandas as pd
import pytest

from timely_beliefs import DBSensor, DBTimedBelief, utils
from timely_beliefs.beliefs.classes import IntTimedelta
from timely_beliefs.tests import session


class TestTimedeltaSecondsConversion:
    """Test timedelta_to_seconds and seconds_to_timedelta utility functions."""

    def test_positive_hours(self):
        td = timedelta(hours=9)
        assert utils.timedelta_to_seconds(td) == 9 * 60 * 60

    def test_positive_days(self):
        td = timedelta(days=1)
        assert utils.timedelta_to_seconds(td) == 24 * 60 * 60

    def test_zero(self):
        td = timedelta(0)
        assert utils.timedelta_to_seconds(td) == 0

    def test_negative_hours(self):
        td = timedelta(hours=-4)
        assert utils.timedelta_to_seconds(td) == -4 * 60 * 60

    def test_seconds_only(self):
        td = timedelta(seconds=15)
        assert utils.timedelta_to_seconds(td) == 15

    def test_minutes_only(self):
        td = timedelta(minutes=15)
        assert utils.timedelta_to_seconds(td) == 15 * 60

    def test_complex_timedelta(self):
        td = timedelta(days=2, hours=3, minutes=30, seconds=45)
        assert utils.timedelta_to_seconds(td) == (
            2 * 24 * 60 * 60 + 3 * 60 * 60 + 30 * 60 + 45
        )

    def test_sub_second_truncated(self):
        """Sub-second precision is truncated (floored) when converting to seconds."""
        td = timedelta(seconds=5, microseconds=500000)
        assert utils.timedelta_to_seconds(td) == 5

    def test_negative_sub_second_floored(self):
        """Negative sub-second precision is floored toward negative infinity."""
        td = timedelta(seconds=-5, microseconds=-500000)
        assert utils.timedelta_to_seconds(td) == -6

    def test_roundtrip(self):
        """Converting timedelta to seconds and back should preserve whole-second values."""
        td = timedelta(hours=48)
        assert utils.seconds_to_timedelta(utils.timedelta_to_seconds(td)) == td

    def test_roundtrip_negative(self):
        td = timedelta(hours=-4)
        assert utils.seconds_to_timedelta(utils.timedelta_to_seconds(td)) == td

    def test_seconds_to_timedelta(self):
        assert utils.seconds_to_timedelta(3600) == timedelta(hours=1)

    def test_seconds_to_timedelta_zero(self):
        assert utils.seconds_to_timedelta(0) == timedelta(0)

    def test_seconds_to_timedelta_negative(self):
        assert utils.seconds_to_timedelta(-4 * 60 * 60) == timedelta(hours=-4)

    def test_timedelta_to_seconds_overflow_raises(self):
        with pytest.raises(OverflowError, match="integer-second range"):
            utils.timedelta_to_seconds(timedelta(seconds=utils.INTEGER_SECONDS_MAX + 1))


class TestIntTimedeltaTypeDecorator:
    """Test the IntTimedelta SQLAlchemy type decorator."""

    def test_process_bind_param_timedelta(self):
        t = IntTimedelta()
        result = t.process_bind_param(timedelta(hours=9), None)
        assert result == 9 * 60 * 60
        assert isinstance(result, int)

    def test_process_bind_param_pd_timedelta(self):
        t = IntTimedelta()
        result = t.process_bind_param(pd.Timedelta(hours=9), None)
        assert result == 9 * 60 * 60
        assert isinstance(result, int)

    def test_process_bind_param_none(self):
        t = IntTimedelta()
        result = t.process_bind_param(None, None)
        assert result is None

    def test_process_bind_param_int(self):
        t = IntTimedelta()
        result = t.process_bind_param(60, None)
        assert result == 60
        assert isinstance(result, int)

    def test_process_bind_param_negative(self):
        t = IntTimedelta()
        result = t.process_bind_param(timedelta(hours=-4), None)
        assert result == -4 * 60 * 60

    def test_process_bind_param_int_overflow_raises(self):
        t = IntTimedelta()
        with pytest.raises(OverflowError, match="integer-second range"):
            t.process_bind_param(utils.INTEGER_SECONDS_MAX + 1, None)

    def test_process_bind_param_unsupported_type_raises(self):
        t = IntTimedelta()
        with pytest.raises(TypeError, match="IntTimedelta only supports"):
            t.process_bind_param(3.5, None)

    def test_process_bind_param_string_raises(self):
        t = IntTimedelta()
        with pytest.raises(TypeError, match="IntTimedelta only supports"):
            t.process_bind_param("60", None)

    def test_process_result_value_positive(self):
        t = IntTimedelta()
        result = t.process_result_value(9 * 60 * 60, None)
        assert result == timedelta(hours=9)
        assert isinstance(result, timedelta)

    def test_process_result_value_none(self):
        t = IntTimedelta()
        result = t.process_result_value(None, None)
        assert result is None

    def test_process_result_value_zero(self):
        t = IntTimedelta()
        result = t.process_result_value(0, None)
        assert result == timedelta(0)

    def test_process_result_value_negative(self):
        t = IntTimedelta()
        result = t.process_result_value(-4 * 60 * 60, None)
        assert result == timedelta(hours=-4)


class TestIntTimedeltaDBRoundtrip:
    """Test that beliefs stored in the DB with IntTimedelta roundtrip correctly."""

    def test_db_belief_horizon_roundtrip(
        self,
        time_slot_sensor: DBSensor,
        rolling_day_ahead_beliefs_about_time_slot_events,
    ):
        """Test that belief_horizon values roundtrip correctly through the DB."""
        bdf = DBTimedBelief.search_session(
            session=session,
            sensor=time_slot_sensor,
        )
        # belief_horizon values should be timedelta objects after retrieval
        belief_horizons = bdf.belief_horizons
        for h in belief_horizons:
            assert isinstance(h, timedelta)

    def test_db_belief_horizon_search_with_horizon_filter(
        self,
        time_slot_sensor: DBSensor,
        rolling_day_ahead_beliefs_about_time_slot_events,
        test_source_a,
    ):
        """Test querying with horizons_at_least and horizons_at_most filters."""
        bdf_all = DBTimedBelief.search_session(
            session=session,
            sensor=time_slot_sensor,
            source=test_source_a,
        )
        # Filter to only get beliefs with horizon >= 0
        bdf_filtered = DBTimedBelief.search_session(
            session=session,
            sensor=time_slot_sensor,
            source=test_source_a,
            horizons_at_least=timedelta(0),
        )
        # All beliefs should be included (they are day-ahead beliefs)
        assert len(bdf_filtered) == len(bdf_all)

    def test_db_stores_integer_column(
        self,
        time_slot_sensor: DBSensor,
        rolling_day_ahead_beliefs_about_time_slot_events,
    ):
        """Verify that the DB column uses IntTimedelta by checking the column type."""
        col = DBTimedBelief.__table__.c.belief_horizon
        assert isinstance(col.type, IntTimedelta)
