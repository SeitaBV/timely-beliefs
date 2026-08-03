"""Tests for the indexes declared on the beliefs table.

These build a throwaway declarative class and inspect the resulting table
metadata, so they need no database.
"""

from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import declarative_base, declared_attr

from timely_beliefs.beliefs.classes import TimedBeliefDBMixin


def build_belief_class(tablename: str):
    Base = declarative_base()

    def source_id(cls):
        return Column(Integer, ForeignKey("belief_source.id"), primary_key=True)

    return type(
        tablename,
        (Base, TimedBeliefDBMixin),
        {"__tablename__": tablename, "source_id": declared_attr(source_id)},
    )


def index_columns(cls) -> list[list[str]]:
    return [[c.name for c in index.columns] for index in cls.__table__.indexes]


def test_no_redundant_single_column_indexes():
    """Single-column indexes must not duplicate a composite index's leading column.

    A single-column index on event_start or sensor_id is fully covered by the
    composite indexes declared on this mixin, so it only costs storage and slows
    every write down.
    """
    single_column = [
        cols for cols in index_columns(build_belief_class("t1")) if len(cols) == 1
    ]
    assert single_column == [], f"redundant single-column indexes: {single_column}"


def test_composite_indexes_cover_the_columns_we_stopped_indexing():
    """Whatever we dropped a single-column index for must still lead a composite one.

    event_start is needed for time-range scans, and sensor_id additionally backs the
    ON DELETE CASCADE from the sensor table -- PostgreSQL can use a composite index
    for that as long as the column leads it.
    """
    leading_columns = {cols[0] for cols in index_columns(build_belief_class("t2"))}
    assert "event_start" in leading_columns
    assert "sensor_id" in leading_columns
