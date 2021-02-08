from functools import total_ordering
from typing import Union

from sqlalchemy import Column, Integer, String

from timely_beliefs.db_base import Base


@total_ordering
class BeliefSource(object):

    """
    A belief source is any data-creating entity such as a user, a ML model or a script.
    """

    name: str

    def __init__(self, name: Union[str, int]):
        """Initialize with a name (string or integer identifier)."""
        if not isinstance(name, str):
            if isinstance(name, int):
                name = str(name)
            else:
                raise TypeError("Please give this source a name to be identifiable.")
        self.name = name

    def __repr__(self):
        return "<BeliefSource %s>" % self.name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        """Set a rule for ordering."""
        return self.name < other.name


class BeliefSourceDBMixin(BeliefSource):
    """
    Mixin class for a table with belief sources.
    """

    id = Column(Integer, primary_key=True)
    # overwriting name as db field
    name = Column(String(120), nullable=False, default="")

    def __init__(self, name: str):
        BeliefSource.__init__(self, name)


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
