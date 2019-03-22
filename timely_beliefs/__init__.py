from typing import List

import numpy as np

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

    @property
    def number_of_beliefs(self) -> int:
        """Return the number of beliefs in the BeliefsDataFrame."""
        gr = self._obj.groupby(["event_start", "belief_time", "source_id"])
        return int(np.prod(gr.dtypes.index.levshape))

    @property
    def number_of_probabilistic_beliefs(self) -> int:
        """Return the number of beliefs in the BeliefsDataFrame that are probabilistic (more than 1 unique value)."""
        gr = self._obj.groupby(["event_start", "belief_time", "source_id"])
        df = gr.nunique(dropna=True)
        return len(df[df>1].dropna())

    @property
    def number_of_deterministic_beliefs(self) -> int :
        """Return the number of beliefs in the BeliefsDataFrame that are deterministic (1 unique value)."""
        return len(self._obj) - self.number_of_probabilistic_beliefs

    @property
    def percentage_of_probabilistic_beliefs(self) -> float:
        """Return the percentage of beliefs in the BeliefsDataFrame that are probabilistic (more than 1 unique value).
        """
        return self.number_of_probabilistic_beliefs / self.number_of_beliefs

    @property
    def percentage_of_deterministic_beliefs(self) -> float :
        """Return the percentage of beliefs in the BeliefsDataFrame that are deterministic (1 unique value).
        """
        return 1 - self.number_of_probabilistic_beliefs / self.number_of_beliefs

    @property
    def probabilistic_accuracy(self) -> float:
        """Return the average number of probabilistic values per belief.
        As a rule of thumb:
        deterministic = 1
              uniform = 2
               normal = 2 or 3
          skew normal = 3
             quartile = 3
          percentile = 99
        """
        return len(self._obj) / float(self.number_of_beliefs)
