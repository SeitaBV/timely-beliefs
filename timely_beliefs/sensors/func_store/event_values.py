"""Function store for aggregating event values."""
import numpy as np

from timely_beliefs.beliefs import classes  # noqa: F401


def nan_mean(
    df: "classes.BeliefsDataFrame", output_resolution, input_resolution
) -> "classes.BeliefsDataFrame":
    """Calculate the mean value while ignoring nan values."""

    first_row = df.iloc[0:1]
    for col in df:
        first_row[col] = np.nanmean(df[col].values)
    return first_row
