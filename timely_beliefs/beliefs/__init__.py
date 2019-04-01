"""Our main class, the BeliefsDataFrame, is an extension on a pandas DataFrame, following recommendations from:

    https://pandas.pydata.org/pandas-docs/stable/development/extending.html

Below, we register customer accessors.
"""

from typing import List

from pandas.api.extensions import register_dataframe_accessor


@register_dataframe_accessor("lineage")
class BeliefsAccessor(object):
    """Add a `lineage` namespace to BeliefsDataFrame objects with convenient attributes providing additional information
    about the data, such as the number of sources and the probabilistic accuracy. This is a very basic approach to add
    some form of data provenance.

    :Example:

    >>> import timely_beliefs
    >>> df = timely_beliefs.tests.example_df
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
            raise AttributeError("Must have index level 'belief_time' or 'belief_horizon'.")
        if "source" not in obj.index.names:
            raise AttributeError("Must have index level 'source'.")
        if "cumulative_probability" not in obj.index.names:
            raise AttributeError("Must have index level 'cumulative_probability'.")
        if "event_value" not in obj.columns:
            raise AttributeError("Must have column 'event_value'.")

    @property
    def events(self) -> List[int] :
        """Return the unique events described in this BeliefsDataFrame."""
        event_start = self._obj.index.get_level_values(level="event_start")
        return event_start.unique().values

    @property
    def number_of_events(self) :
        """Return the number of unique event described in this BeliefsDataFrame."""
        return len(self.events)

    @property
    def belief_times(self) -> List[int] :
        """Return the unique belief times in this BeliefsDataFrame."""
        belief_time = self._obj.index.get_level_values(level="belief_time")
        return belief_time.unique().values

    @property
    def number_of_belief_times(self) :
        """Return the number of unique belief times described in this BeliefsDataFrame."""
        return len(self.belief_times)

    @property
    def number_of_beliefs(self) -> int:
        """Return the total number of beliefs in the BeliefsDataFrame, including both deterministic beliefs (which
        require a single row) and probabilistic beliefs (which require multiple rows)."""
        index_names = []
        index_names.append("event_start") if "event_start" in self._obj.index.names else index_names.append("event_end")
        index_names.append("belief_time") if "belief_time" in self._obj.index.names else index_names.append("belief_horizon")
        index_names.append("source")
        gr = self._obj.groupby(index_names)
        return len(gr)

    @property
    def sources(self) -> List[int]:
        """Return the unique sources for this BeliefsDataFrame."""
        source = self._obj.index.get_level_values(level="source")
        return source.unique().values

    @property
    def number_of_sources(self):
        """Return the number of unique sources for this BeliefsDataFrame."""
        return len(self.sources)

    @property
    def number_of_probabilistic_beliefs(self) -> int:
        """Return the number of beliefs in the BeliefsDataFrame that are probabilistic (more than 1 unique value)."""
        index_names = []
        index_names.append("event_start") if "event_start" in self._obj.index.names else index_names.append("event_end")
        index_names.append("belief_time") if "belief_time" in self._obj.index.names else index_names.append("belief_horizon")
        index_names.append("source")
        gr = self._obj.groupby(index_names)
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
