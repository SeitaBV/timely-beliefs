from typing import Union

from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declared_attr

from timely_beliefs.db_base import Base


class BeliefSource(object):

    """
    A belief source is any data-creating entitiy such as a user, a ML model or a script.
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

    def __repr__(self):
        return "<BeliefSource %s (%s)>" % (self.id, self.name)


class InheritableMixin(BeliefSource):
    """
    Mixin class which allows your db class to be extended (via inheritance)
    TODO: maybe we can leave this away!
    """

    # Supporting single-table inheritance, so db classes can be extended easily.
    # (https://docs.sqlalchemy.org/en/13/orm/inheritance.html#single-table-inheritance)
    # type_ is required for polymorphic inheritance
    type_ = Column(String(50), nullable=False)

    @declared_attr
    def __mapper_args__(self):
        if self.__name__ == "DBBeliefSource":
            return {
                "polymorphic_on": self.type_,
                "polymorphic_identity": "DBBeliefSource",
            }
        else:
            return {"polymorphic_identity": self.__name__}


class DBBeliefSource(Base, BeliefSourceDBMixin):
    """
    Database class for a table with belief sources.
    """

    __tablename__ = "belief_source"

    def __init__(self, name: str):
        BeliefSourceDBMixin.__init__(self, name)
        Base.__init__(self)

