"""Our main class, the BeliefsDataFrame, is an extension on a pandas DataFrame, following recommendations from:

    https://pandas.pydata.org/pandas-docs/stable/development/extending.html

Below, we register customer accessors.
"""

from datetime import datetime, timedelta
from typing import List

from pandas.api.extensions import register_dataframe_accessor


@register_dataframe_accessor("lineage")
class BeliefsAccessor(object):
    """Add a `lineage` namespace to BeliefsDataFrame objects with convenient attributes providing additional information
    about the data, such as the number of sources and the probabilistic accuracy. This is a very basic approach to add
    some form of data provenance.

    :Example:

    >>> import timely_beliefs
    >>> df = timely_beliefs.tests.get_example_df()
    >>> df.lineage.number_of_sources
    2

    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if all(key not in obj.index.names for key in ["event_start", "event_end"]):
            raise AttributeError("Must have index level 'event_start' or 'event_end'.")
        if all(key not in obj.index.names for key in ["belief_time", "belief_horizon"]):
            raise AttributeError(
                "Must have index level 'belief_time' or 'belief_horizon'."
            )
        if "source" not in obj.index.names:
            raise AttributeError("Must have index level 'source'.")
        if "cumulative_probability" not in obj.index.names:
            raise AttributeError("Must have index level 'cumulative_probability'.")

    @property
    def events(self) -> List[int]:
        """Return the unique events described in this BeliefsDataFrame."""
        event_start = self._obj.index.get_level_values(level="event_start")
        return event_start.unique().to_numpy(dtype="object")

    @property
    def number_of_events(self):
        """Return the number of unique event described in this BeliefsDataFrame."""
        return len(self.events)

    @property
    def belief_horizons(self) -> List[timedelta]:
        """Return the unique belief horizons in this BeliefsDataFrame."""
        if "belief_horizon" in self._obj.index.names:
            return (
                self._obj.index.get_level_values(level="belief_horizon")
                .unique()
                .to_pytimedelta()
            )
        else:
            return (
                self._obj.convert_index_from_belief_time_to_horizon()
                .index.get_level_values(level="belief_horizon")
                .unique()
                .to_pytimedelta()
            )

    @property
    def number_of_belief_horizons(self):
        """Return the number of unique belief horizons described in this BeliefsDataFrame."""
        return len(self.belief_horizons)

    @property
    def belief_times(self) -> List[datetime]:
        """Return the unique belief times in this BeliefsDataFrame."""
        if "belief_time" in self._obj.index.names:
            return (
                self._obj.index.get_level_values(level="belief_time")
                .unique()
                .to_pydatetime()
            )
        else:
            return (
                self._obj.convert_index_from_belief_horizon_to_time()
                .index.get_level_values(level="belief_time")
                .unique()
                .to_pydatetime()
            )

    @property
    def number_of_belief_times(self):
        """Return the number of unique belief times described in this BeliefsDataFrame."""
        return len(self.belief_times)

    @property
    def number_of_beliefs(self) -> int:
        """Return the total number of beliefs in the BeliefsDataFrame, including both deterministic beliefs (which
        require a single row) and probabilistic beliefs (which require multiple rows)."""
        return len(self._obj.probabilistic_depth_per_belief)

    @property
    def sources(self) -> List[int]:
        """Return the unique sources for this BeliefsDataFrame."""
        source = self._obj.index.get_level_values(level="source")
        return source.unique().to_numpy(dtype="object")

    @property
    def number_of_sources(self):
        """Return the number of unique sources for this BeliefsDataFrame."""
        return len(self.sources)

    @property
    def number_of_probabilistic_beliefs(self) -> int:
        """Return the number of beliefs in the BeliefsDataFrame that are probabilistic (more than 1 unique value)."""
        return self.number_of_beliefs - self.number_of_deterministic_beliefs

    @property
    def number_of_deterministic_beliefs(self) -> int:
        """Return the number of beliefs in the BeliefsDataFrame that are deterministic (1 unique value)."""
        return self._obj.probabilistic_depth_count.get(1, 0)

    @property
    def percentage_of_probabilistic_beliefs(self) -> float:
        """Return the percentage of beliefs in the BeliefsDataFrame that are probabilistic (more than 1 unique value).

        Empty frames default to 0.
        """
        return 1 - self.percentage_of_deterministic_beliefs

    @property
    def percentage_of_deterministic_beliefs(self) -> float:
        """Return the percentage of beliefs in the BeliefsDataFrame that are deterministic (1 unique value).

        Empty frames default to 1.
        """
        if len(self._obj) == 0:
            return 1
        return self.number_of_deterministic_beliefs / self.number_of_beliefs

    @property
    def unique_beliefs_per_event_per_source(self) -> bool:
        """Return whether or not the BeliefsDataFrame contains at most 1 belief per event per source."""
        return len(
            self._obj.groupby(
                level=["event_start", "source", "cumulative_probability"],
                group_keys=False,
            )
        ) == len(self._obj)

    @property
    def probabilistic_depth(self) -> float:
        """Return the average number of probabilistic values per belief.
        As a rule of thumb:
        deterministic = 1
              uniform = 2
               normal = 2 or 3
          skew normal = 3
             quartile = 3
          percentile = 99

        Empty frames default to 1.
        """
        if len(self._obj) == 0:
            return 1
        return len(self._obj) / float(self.number_of_beliefs)
