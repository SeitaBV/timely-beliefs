from typing import Optional

from sqlalchemy.orm import Query, Session
from sqlalchemy import and_, func


def query_unchanged_beliefs(
    session: Session,
    cls: "TimedBeliefDBMixin",  # noqa F821
    query: Optional[Query] = None,
) -> Query:
    """Match unchanged beliefs.

    Unchanged beliefs are beliefs that have already been recorded with an earlier belief time,
    and are otherwise the same.
    """
    if query is None:
        query = cls.query
    subq = (
        session.query(
            cls.event_start,
            cls.sensor_id,
            cls.source_id,
            cls.event_value,
            func.max(cls.belief_horizon).label("original_belief_horizon"),
        )
        .group_by(cls.event_start, cls.sensor_id, cls.source_id, cls.event_value)
        .subquery()
    )
    q = query.join(
        subq,
        and_(
            cls.event_start == subq.c.event_start,
            cls.sensor_id == subq.c.sensor_id,
            cls.source_id == subq.c.source_id,
            cls.event_value == subq.c.event_value,
            cls.belief_horizon != subq.c.original_belief_horizon,
        ),
    )
    return q
