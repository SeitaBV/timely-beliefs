from __future__ import annotations

from functools import total_ordering

import numpy as np
from sqlalchemy import Column, Integer, String

from timely_beliefs.db_base import Base


@total_ordering
class BeliefSource(object):
    """
    A belief source is any data-creating entity such as a user, a ML model or a script.
    """

    name: str

    def __init__(self, name: str | int):
        """Initialize with a name (string or integer identifier)."""
        if not isinstance(name, str):
            if isinstance(name, (int, np.int64)):
                name = str(name)
            else:
                raise TypeError("Please give this source a name to be identifiable.")
        self.name = name

    def __repr__(self):
        return "<BeliefSource %s>" % self.name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        """Set a strict total order, tiebreaking on object identity when names are equal."""
        if not isinstance(other, BeliefSource):
            return NotImplemented
        return (str(self), id(self)) < (str(other), id(other))


@total_ordering
class BeliefSourceDBMixin(BeliefSource):
    """
    Mixin class for a table with belief sources.
    """

    id = Column(Integer, primary_key=True)
    # overwriting name as db field
    name = Column(String(120), nullable=False, default="")

    def __init__(self, name: str):
        BeliefSource.__init__(self, name)

    def __eq__(self, other):
        """Two database-backed sources are equal if they share a persisted primary key."""
        if not isinstance(other, BeliefSourceDBMixin):
            return NotImplemented
        if self.id is not None and other.id is not None:
            return self.id == other.id
        return self is other

    def __hash__(self):
        """Hash based on name so hash is stable before and after flushing an id."""
        return hash(self.name)

    def __lt__(self, other):
        """Order by name, then primary key, falling back to identity for unflushed instances."""
        if not isinstance(other, BeliefSource):
            return NotImplemented
        other_id = getattr(other, "id", None)
        self_key = (
            str(self),
            self.id is None,
            self.id or 0,
            0 if self.id is not None else id(self),
        )
        other_key = (
            str(other),
            other_id is None,
            other_id or 0,
            0 if other_id is not None else id(other),
        )
        return self_key < other_key


class DBBeliefSource(Base, BeliefSourceDBMixin):
    """
    Database class for a table with belief sources.
    """

    __tablename__ = "belief_source"

    def __init__(self, name: str):
        BeliefSourceDBMixin.__init__(self, name)
        Base.__init__(self)

    def __repr__(self):
        return "<DBBeliefSource %s (%s)>" % (self.id, self.name)
