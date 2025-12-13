CREATE_MVIEW_SQL = """
CREATE MATERIALIZED VIEW most_recent_beliefs_mview AS
SELECT *
FROM (
    SELECT
        my_timed_belief.sensor_id,
        my_timed_belief.event_start,
        my_timed_belief.source_id,
        MIN(my_timed_belief.belief_horizon) AS most_recent_belief_horizon
    FROM my_timed_belief
    INNER JOIN belief_source
        ON belief_source.id = my_timed_belief.source_id
    GROUP BY
        my_timed_belief.sensor_id,
        my_timed_belief.event_start,
        my_timed_belief.source_id
) AS belief_mins
GROUP BY
    sensor_id,
    event_start,
    source_id,
    most_recent_belief_horizon;
"""

DROP_MVIEW_SQL = """
DROP MATERIALIZED VIEW IF EXISTS most_recent_beliefs_mview CASCADE;
"""

CREATE_INDEXES_SQL = """
CREATE INDEX idx_most_recent_beliefs_mview_sensor_event
    ON most_recent_beliefs_mview(sensor_id, event_start);

CREATE INDEX idx_most_recent_beliefs_mview_event_start
    ON most_recent_beliefs_mview(event_start);

CREATE UNIQUE INDEX idx_most_recent_beliefs_mview_unique
    ON most_recent_beliefs_mview(sensor_id, event_start, source_id);
"""

REFRESH_MVIEW_SQL = """
REFRESH MATERIALIZED VIEW most_recent_beliefs_mview
"""
