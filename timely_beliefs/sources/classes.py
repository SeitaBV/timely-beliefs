from sqlalchemy import Column, Integer, String

from timely_beliefs.base import Base


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

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False, default="")

    def __repr__(self):
        return "<DBBeliefSource %s (%s)>" % (self.id, self.name)

    def __str__(self):
        return self.name

    def __lt__(self, other):
        """Set a rule for ordering."""
        return self.name < other.name
