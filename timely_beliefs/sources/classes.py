from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declared_attr

from timely_beliefs.db_base import Base


class BeliefSource(object):

    name: str

    def __init__(self, name: str = ""):
        if name == "":
            raise Exception("Please give this source a name to be identifiable.")
        self.name = name

    def __repr__(self):
        return "<BeliefSource %s>" % self.name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        """Set a rule for ordering."""
        return self.name < other.name


class DBBeliefSource(Base):
    """Mixin class for a table with belief sources, i.e. data-creating entities such as users or scripts."""

    __tablename__ = "belief_source"

    # two columns for db purposes: id is a row identifier
    id = Column(Integer, primary_key=True)
    # type is useful so we can use polymorphic inheritance
    # (https://docs.sqlalchemy.org/en/13/orm/inheritance.html#single-table-inheritance)
    type = Column(String(50), nullable=False)

    name = Column(String(120), nullable=False, default="")

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return "<DBBeliefSource %s (%s)>" % (self.id, self.name)

    def __str__(self):
        return self.name

    def __lt__(self, other):
        """Set a rule for ordering."""
        return self.name < other.name

    @declared_attr
    def __mapper_args__(self):
        if self.__name__ == "DBBeliefSource":
            return {
                "polymorphic_on": self.type,
                "polymorphic_identity": "DBBeliefSource",
            }
        else:
            return {"polymorphic_identity": self.__name__}
