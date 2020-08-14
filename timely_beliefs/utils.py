from datetime import datetime
from typing import Optional, Sequence

import pandas as pd


def enforce_tz(dt: datetime, name: Optional[str]) -> datetime:
    if dt.tzinfo is None:
        raise TypeError(
            f"The timely-beliefs package does not work with timezone-naive datetimes. Please localize your {name if name else 'datetime'} {dt}."
        )
    return dt


def all_of_type(seq: Sequence, element_type) -> bool:
    """Return true if all elements in sequence are of the same type."""
    for item in seq:
        if type(item) != element_type:
            return False
    return True


def replace_multi_index_level(
    df: "classes.BeliefsDataFrame",
    level: str,
    index: pd.Index,
    intersection: bool = False,  # noqa: F821
) -> "classes.BeliefsDataFrame":
    """Replace one of the index levels of the multi-indexed DataFrame. Returns a new DataFrame object.
    :param df: a BeliefsDataFrame (or just a multi-indexed DataFrame).
    :param level: the name of the index level to replace.
    :param index: the new index.
    :param intersection: policy for replacing the index level.
    If intersection is False then simply replace (note that the new index should have the same length as the old index).
    If intersection is True then add indices not contained in the old index and delete indices not contained in the new
    index. New rows have nan columns values and copies of the first row for other index levels (note that the resulting
    index is usually longer and contains values that were both in the old and new index, i.e. the intersection).
    """
    # Todo: check whether timezone information is copied over correctly

    # Check input
    if intersection is False and len(index) != len(df.index):
        raise ValueError(
            "Cannot simply replace multi-index level with an index of different length than the original. "
            "Use intersection instead?"
        )
    if index.name is None:
        index.name = level

    new_index_values = []
    new_index_names = []
    if intersection is True:
        contained_in_old = index.isin(df.index.get_level_values(level))
        new_index_not_in_old = index[~contained_in_old]
        contained_in_new = df.index.get_level_values(level).isin(index)
        for i in df.index.names:
            if i == level:  # For the index level that should be replaced
                # Copy old values that the new index contains, and add new values that the old index does not contain
                new_index_values.append(
                    df.index.get_level_values(i)[contained_in_new].append(
                        new_index_not_in_old
                    )
                )
                new_index_names.append(index.name)
            else:  # For the other index levels
                # Copy old values that the new index contains, and add the first value to the new rows
                new_row_values = pd.Index(
                    [df.index.get_level_values(i)[0]] * len(new_index_not_in_old)
                )
                new_index_values.append(
                    df.index.get_level_values(i)[contained_in_new].append(
                        new_row_values
                    )
                )
                new_index_names.append(i)
    else:
        for i in df.index.names:
            if i == level:  # For the index level that should be replaced
                # Replace with new index
                new_index_values.append(index)
                new_index_names.append(index.name)
            else:  # For the other index levels
                # Copy all old values
                new_index_values.append(df.index.get_level_values(i))
                new_index_names.append(i)

    # Construct new MultiIndex
    mux = pd.MultiIndex.from_arrays(new_index_values, names=new_index_names)

    df = df.copy(deep=True)
    # Apply new MultiIndex
    if intersection is True:
        # Reindex such that new rows get nan column values
        df = df.reindex(mux)
    else:
        # Replace the index
        df.index = mux
    return df.sort_index()
