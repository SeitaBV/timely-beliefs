import warnings
from datetime import datetime, timedelta
from typing import Optional, Sequence, Union

import pandas as pd


def parse_timedelta_like(
    td: Union[timedelta, str, pd.Timedelta],
    variable_name: Optional[str] = None,
) -> timedelta:
    """Parse timedelta like objects as a datetime.timedelta object.

    The interpretation of M (month), Y and y (year) units as a timedelta is ambiguous (e.g. td="1Y").
    For these cases, Pandas (since 0.25) gives a FutureWarning, warning that support for these units will be removed.
    This function throws a ValueError in case Pandas gives a FutureWarning.
    https://pandas.pydata.org/docs/whatsnew/v0.25.0.html#deprecations

    :param td: timedelta-like object
    :param variable_name: used to give a better error message in case the variable failed to parse
    :return: timedelta
    """
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            if isinstance(td, str):
                td = pd.Timedelta(td)
            if isinstance(td, pd.Timedelta):
                td = td.to_pytimedelta()
    except (ValueError, FutureWarning) as e:
        raise ValueError(
            f"Could not parse {variable_name if variable_name else 'timedelta'} {td}, because {e}"
        )
    return td


def parse_datetime_like(
    dt: Union[datetime, str, pd.Timestamp], variable_name: Optional[str] = None
) -> datetime:
    """Parse datetime-like objects as a datetime.datetime object.

    :param dt: datetime-like object
    :param variable_name: used to give a better error message in case the variable failed to parse
    :return: timezone-aware datetime
    """
    try:
        if isinstance(dt, str):
            dt = pd.Timestamp(dt)
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
    except ValueError as e:
        raise ValueError(
            f"Could not parse {variable_name if variable_name else 'datetime'} {dt}, because {e}"
        )
    return enforce_tz(dt, variable_name)


def enforce_tz(dt: datetime, variable_name: Optional[str] = None) -> datetime:
    """Raise exception in case of a timezone-naive datetime.

    :param dt: datetime
    :param variable_name: used to give a better error message in case the variable contained a timezone-naive datetime
    :return: timezone-aware datetime
    """
    if not hasattr(dt, "tzinfo") or dt.tzinfo is None:
        raise TypeError(
            f"The timely-beliefs package does not work with timezone-naive datetimes. Please localize your {variable_name if variable_name else 'datetime'} {dt}."
        )
    return dt


def all_of_type(seq: Sequence, element_type) -> bool:
    """Return true if all elements in sequence are of the same type."""
    for item in seq:
        if type(item) != element_type:
            return False
    return True


def replace_multi_index_level(
    df: "classes.BeliefsDataFrame",  # noqa: F821
    level: str,
    index: pd.Index,
    intersection: bool = False,
) -> "classes.BeliefsDataFrame":  # noqa: F821
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


def append_doc_of(fun):
    def decorator(f):
        if f.__doc__:
            f.__doc__ += fun.__doc__
        else:
            f.__doc__ = fun.__doc__
        return f

    return decorator


def replace_deprecated_argument(
    deprecated_arg_name: str,
    deprecated_arg_val: any,
    new_arg_name: str,
    new_arg_val: any,
) -> any:
    """Util function for replacing a deprecated argument in favour of a new argument.
    If new_arg_val was not already set, it is set to deprecated_arg_val together with a FutureWarning.
    """
    if new_arg_val is None and deprecated_arg_val is None:
        raise ValueError(f"Missing argument: {new_arg_name}.")
    elif new_arg_val is not None:
        pass
    else:
        import warnings

        warnings.warn(
            f"Argument '{deprecated_arg_name}' will be replaced by '{new_arg_name}'. Replace '{deprecated_arg_name}' with '{new_arg_name}' to suppress this warning.",
            FutureWarning,
        )
        new_arg_val = deprecated_arg_val
    return new_arg_val


def remove_class_init_kwargs(cls, kwargs: dict) -> dict:
    """Remove kwargs used to initialize the given class."""
    params = list(cls.__init__.__code__.co_varnames)
    params.remove("self")
    for param in params:
        kwargs.pop(param, None)
    return kwargs
