from timely_beliefs import DBTimedBelief
from timely_beliefs.beliefs.materialized_views import (
    create_mview_ddl,
    create_mview_indexes_ddl,
    drop_mview_ddl,
    refresh_mview_ddl,
)

CREATE_MVIEW_SQL = create_mview_ddl(DBTimedBelief)
CREATE_INDEXES_SQL = create_mview_indexes_ddl()
DROP_MVIEW_SQL = drop_mview_ddl()
REFRESH_MVIEW_SQL = refresh_mview_ddl()
