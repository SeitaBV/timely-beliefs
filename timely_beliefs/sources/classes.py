from __future__ import annotations

from functools import total_ordering

import numpy as np
from sqlalchemy import Column, Integer, String, inspect

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
        """Two database-backed sources are equal if they stand for the same row.

        Rows are told apart by SQLAlchemy's identity key (the mapped class and primary key),
        which is known without loading anything from the database,
        so comparing expired or detached sources does not query or raise.
        A source that was not flushed yet has no identity key, so it only equals itself.
        """
        if not isinstance(other, BeliefSourceDBMixin):
            return NotImplemented
        if self is other:
            return True
        identity_key = inspect(self).key
        return identity_key is not None and identity_key == inspect(other).key

    def __hash__(self):
        """Hash the identity key, or the object itself for a source that was not flushed yet.

        The first hash is kept for the object's lifetime,
        so a source used as a dict key or set member before a flush is still found after it.
        The price is that such a source hashes differently from another object loaded later for the same row,
        although the two compare equal.
        """
        cached_hash = self.__dict__.get("_belief_source_hash")
        if cached_hash is None:
            identity_key = inspect(self).key
            cached_hash = hash(identity_key) if identity_key is not None else id(self)
            self.__dict__["_belief_source_hash"] = cached_hash
        return cached_hash

    def __lt__(self, other):
        """Order by name, then by row, with sources that were not flushed yet after flushed ones, ordered by object identity."""
        if not isinstance(other, BeliefSource):
            return NotImplemented
        return _order_key(self) < _order_key(other)


def _order_key(source: BeliefSource) -> tuple:
    """Key to order sources by, consistent with how `BeliefSourceDBMixin` and `BeliefSource` define equality.

    A flushed database-backed source is ordered by its mapped class name and primary key, read from its identity key.
    Any other source is ordered by object identity.
    """
    identity_key = (
        inspect(source).key if isinstance(source, BeliefSourceDBMixin) else None
    )
    if identity_key is None:
        return str(source), True, "", (), id(source)
    mapped_class, primary_key, _ = identity_key
    return str(source), False, mapped_class.__name__, primary_key, 0


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
