from datetime import timedelta
from typing import Optional

from sqlalchemy import and_, func
from sqlalchemy.orm import Query, Session, aliased


def query_unchanged_beliefs(
    session: Session,
    cls: Optional["TimedBeliefDBMixin"] = None,  # noqa F821
    query: Optional[Query] = None,
    include_positive_horizons: bool = True,  # include belief horizons > 0 (i.e. forecasts)
    include_non_positive_horizons: bool = True,  # include belief horizons <= 0 (i.e. measurements, nowcasts and backcasts)
) -> Query:
    """Match unchanged beliefs.

    Unchanged beliefs are beliefs that have not changed with respect to the preceding belief,
    other than their belief time.
    """
    if cls is None:
        # Avoid circular import
        from timely_beliefs import DBTimedBelief as cls
    if query is None:
        query = session.query(cls)

    # Set up aliases
    tb1 = cls  # the DBTimedBelief class mapped to the timed_beliefs table
    tb2 = aliased(cls)  # alias for holding each preceding belief
    tb3 = aliased(cls)  # alias from which to select the preceding belief

    # Set up / copy criteria for the tb3 alias
    # todo: if query has filter criteria on tb1, those should be applied to tb2 and tb3, too. Hint, combine visitors.iterate(query.whereclause) with replace_selectable
    tb3_criteria = []
    if include_positive_horizons and not include_non_positive_horizons:
        tb3_criteria.append(tb3.belief_horizon > timedelta(0))
    elif include_non_positive_horizons and not include_positive_horizons:
        tb3_criteria.append(tb3.belief_horizon <= timedelta(0))

    q = query.join(
        tb2,
        and_(
            tb2.event_start == tb1.event_start,
            tb2.sensor_id == tb1.sensor_id,
            tb2.source_id == tb1.source_id,
            tb2.event_value == tb1.event_value,
            # next higher belief horizon for a given event for a given source, i.e. the preceding belief
            tb2.belief_horizon
            == session.query(func.min(tb3.belief_horizon))
            .where(
                tb3.belief_horizon > tb1.belief_horizon,
                tb3.event_start == tb1.event_start,
                tb3.sensor_id == tb1.sensor_id,
                tb3.source_id == tb1.source_id,
                *tb3_criteria,
            )
            .scalar_subquery(),
        ),
        # NB: to query changed beliefs instead, adjust the join clauses as follows to not lose the very oldest belief:
        # or_(tb2.sensor_id == tb1.sensor_id, tb2.sensor_id is None),
        # or_(tb2.source_id == tb1.source_id, tb2.source_id is None),
        # or_(tb2.event_value != tb1.event_value, tb2.event_value is None),
    )
    return q
