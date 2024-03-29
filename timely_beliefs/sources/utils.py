from __future__ import annotations

import warnings

from timely_beliefs import BeliefSource


def ensure_source_exists(
    source: BeliefSource | str | int | None, allow_none: bool = False
) -> BeliefSource | None:
    """Creates a BeliefSource if source is not already a BeliefSource, with a warning.
    By default, if a source is None, it fails."""
    if isinstance(source, BeliefSource) or (source is None and allow_none):
        return source
    source_created = BeliefSource(source)  # throws error if source is None
    warnings.warn(f"{source_created.__repr__()} created from {source.__repr__()}.")
    return source_created
