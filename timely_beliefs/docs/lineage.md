# Lineage

Get the (number of) sources contributing to the BeliefsDataFrame:

    >>> df.lineage.sources
    array([<BeliefSource Source A>, <BeliefSource Source B>], dtype=object)
    >>> df.lineage.number_of_sources
    2

Many more convenient properties can be found in `df.lineage`.
