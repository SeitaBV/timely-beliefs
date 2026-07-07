"""Optional support for a Postgres materialized view that caches the most recent belief horizon
per event and source, per sensor.

Selecting the most recent beliefs normally requires a MIN(belief_horizon) GROUP BY subquery over
the beliefs table, which is expensive on large datasets. Hosts can pre-compute that subquery in a
materialized view and pass it to TimedBelief.search_session (or rely on its default lookup) to
speed up such queries considerably.

Responsibilities left to the host application:
- Creating the view (and its indexes), e.g. in a database migration, using the DDL generators here.
- Refreshing the view periodically (e.g. from a cron job), using refresh_mview_ddl().
  Note that a unique index is required to refresh concurrently (i.e. without locking reads),
  which is why create_mview_indexes_ddl() includes one.

Staleness semantics: the view only knows about beliefs present at the time of its last refresh.
To avoid missing events recorded since then, pass mview_cutoff to search_session: events starting
before the cutoff are looked up in the view, while later events are looked up in the beliefs table.
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, Interval, MetaData, Table, text
from sqlalchemy.orm import Session

MVIEW_NAME = "most_recent_beliefs_mview"

# Define the mview Table once; its structure matches the DDL from create_mview_ddl()
MOST_RECENT_BELIEFS_MVIEW = Table(
    MVIEW_NAME,
    MetaData(),
    Column("sensor_id", Integer),
    Column("event_start", DateTime(timezone=True)),
    Column("source_id", Integer),
    Column("most_recent_belief_horizon", Interval),
)


def create_mview_ddl(cls) -> str:
    """Return DDL for creating the materialized view over the given TimedBelief model class.

    :param cls: a TimedBeliefDBMixin subclass; its __tablename__ determines the source table
    """
    tablename = cls.__tablename__
    return f"""
CREATE MATERIALIZED VIEW {MVIEW_NAME} AS
SELECT
    sensor_id,
    event_start,
    source_id,
    MIN(belief_horizon) AS most_recent_belief_horizon
FROM {tablename}
GROUP BY
    sensor_id,
    event_start,
    source_id;
"""


def create_mview_indexes_ddl() -> str:
    """Return DDL for creating the materialized view's indexes.

    The unique index is required for REFRESH MATERIALIZED VIEW CONCURRENTLY.
    """
    return f"""
CREATE INDEX idx_{MVIEW_NAME}_sensor_event
    ON {MVIEW_NAME} (sensor_id, event_start);

CREATE INDEX idx_{MVIEW_NAME}_event_start
    ON {MVIEW_NAME} (event_start);

CREATE UNIQUE INDEX idx_{MVIEW_NAME}_unique
    ON {MVIEW_NAME} (sensor_id, event_start, source_id);
"""


def drop_mview_ddl() -> str:
    """Return DDL for dropping the materialized view (and its indexes with it)."""
    return f"DROP MATERIALIZED VIEW IF EXISTS {MVIEW_NAME};"


def refresh_mview_ddl(concurrently: bool = False) -> str:
    """Return DDL for refreshing the materialized view.

    :param concurrently: if True, avoid locking reads during the refresh,
                         at the cost of higher resource usage
                         (requires the unique index and cannot run inside a transaction block)
    """
    return f"REFRESH MATERIALIZED VIEW {'CONCURRENTLY ' if concurrently else ''}{MVIEW_NAME};"


def get_most_recent_beliefs_mview(session: Session) -> Table | None:
    """Return the mview Table if the materialized view exists in the database, else None."""
    row = session.execute(
        text("SELECT 1 FROM pg_matviews WHERE matviewname = :name"),
        {"name": MVIEW_NAME},
    ).fetchone()
    return MOST_RECENT_BELIEFS_MVIEW if row else None
