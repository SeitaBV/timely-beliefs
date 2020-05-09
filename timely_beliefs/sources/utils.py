from typing import Optional, Union
import warnings

from timely_beliefs import BeliefSource


def ensure_source(
    source: Optional[Union[BeliefSource, str, int]], allow_none: bool = False
):
    """Creates a BeliefSource if source is not already a BeliefSource, with a warning.
    By default, if a source is None, it doesn't fail."""
    if isinstance(source, BeliefSource) or (source is None and allow_none):
        return source
    source_created = BeliefSource(source)  # throws error if source is None
    warnings.warn(f"{source_created.__repr__()} created from {source.__repr__()}.")
    return source_created
