from typing import List

from timely_beliefs.sensors.classes import Sensor
from timely_beliefs.sources.classes import BeliefSource
from timely_beliefs.beliefs.classes import TimedBelief, BeliefsDataFrame
from pandas.api.extensions import register_dataframe_accessor


@register_dataframe_accessor("lineage")
class SensorAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def sources(self) -> List[int]:
        """Return the unique sources for this BeliefsDataFrame."""
        source_id = self._obj.index.get_level_values(level="source_id")
        return source_id.unique().values

    @property
    def number_of_sources(self):
        """Return the number of unique sources for this BeliefsDataFrame."""
        return len(self.sources)
