from pandas import Index, DatetimeIndex

from timely_beliefs.beliefs import classes


def select_most_recent_belief(df: "classes.BeliefsDataFrame") -> "classes.BeliefsDataFrame":

    # Remember original index levels
    indices = df.index.names

    # Convert index levels to columns
    df = df.reset_index()

    # Drop all but most recent belief
    if "belief_horizon" in indices:
        df = df.sort_values(by=["belief_horizon"], ascending=True).drop_duplicates(subset=["event_start"], keep="first").sort_values(by=["event_start"])
    elif "belief_time" in indices:
        df = df.sort_values(by=["belief_time"], ascending=True).drop_duplicates(subset=["event_start"], keep="last").sort_values(by=["event_start"])
    else:
        raise KeyError("No belief_horizon or belief_time index level found in DataFrame.")

    # Convert columns to index levels (only columns that represent index levels)
    return df.set_index(indices)


def replace_multi_index_level(df: "classes.BeliefsDataFrame", level: str, index: Index) -> "classes.BeliefsDataFrame":

    # Remember original index levels and the new index
    indices = []
    for i in df.index.names:
        if i == level:
            indices.append(index.name)
        else:
            indices.append(i)

    # Convert index levels to columns
    df = df.reset_index()

    # Replace desired index level (as a column)
    if isinstance(index, DatetimeIndex):
        df[level] = index.to_series(keep_tz=True, name="").reset_index()[index.name]
    else:
        df[level] = index.to_series(name="").reset_index()[index.name]
    df = df.rename({level: index.name}, axis="columns")

    # Convert columns to index levels (only columns that represent index levels)
    return df.set_index(indices)
