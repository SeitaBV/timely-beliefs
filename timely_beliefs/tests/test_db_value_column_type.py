"""Tests for customizing the type of the per-row numeric value columns.

These build throwaway declarative classes and only inspect the resulting table
metadata, so they need no database.
"""

from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.orm import declarative_base, declared_attr

from timely_beliefs.beliefs.classes import TimedBeliefDBMixin

VALUE_COLUMNS = ("cumulative_probability", "event_value")


def build_belief_class(tablename: str, value_column_type=None):
    """Build a TimedBeliefDBMixin subclass on its own declarative base."""
    Base = declarative_base()
    namespace = {"__tablename__": tablename}

    def source_id(cls):
        return Column(Integer, ForeignKey("belief_source.id"), primary_key=True)

    namespace["source_id"] = declared_attr(source_id)
    if value_column_type is not None:
        namespace["value_column_type"] = value_column_type
    return type(tablename, (Base, TimedBeliefDBMixin), namespace)


def test_value_columns_default_to_double_precision():
    """By default the value columns are float8, i.e. Float without a precision."""
    cls = build_belief_class("default_belief")
    for column_name in VALUE_COLUMNS:
        column = cls.__table__.columns[column_name]
        assert isinstance(column.type, Float)
        assert column.type.precision is None
        assert column.nullable is False
    assert cls.__table__.columns["cumulative_probability"].primary_key is True
    assert cls.__table__.columns["event_value"].primary_key is False


def test_value_column_type_is_customizable():
    """Setting value_column_type narrows both value columns to float4."""
    cls = build_belief_class("float4_belief", value_column_type=Float(precision=24))
    for column_name in VALUE_COLUMNS:
        column = cls.__table__.columns[column_name]
        assert isinstance(column.type, Float)
        assert column.type.precision == 24
        # Narrowing the type must not silently drop the other column properties
        assert column.nullable is False
    assert cls.__table__.columns["cumulative_probability"].primary_key is True
    assert cls.__table__.columns["cumulative_probability"].default.arg == 0.5
    assert cls.__table__.columns["event_value"].primary_key is False


def test_customizing_value_column_type_preserves_column_and_pk_order():
    """Column and primary-key order must not depend on value_column_type.

    Reordering the primary key would silently change index behaviour for hosts
    that adopt a narrower type, and would make a fresh create_all() disagree with
    an existing migrated table. Redeclaring these columns in the subclass body
    (rather than using this hook) does exactly that, which is why the hook exists.
    """
    default_cls = build_belief_class("order_default_belief")
    float4_cls = build_belief_class(
        "order_float4_belief", value_column_type=Float(precision=24)
    )

    assert [c.name for c in default_cls.__table__.columns] == [
        c.name for c in float4_cls.__table__.columns
    ]
    assert [c.name for c in default_cls.__table__.primary_key.columns] == [
        c.name for c in float4_cls.__table__.primary_key.columns
    ]
