from sqlalchemy import Column, Integer

from timely_beliefs.base import Base


class BeliefSource(Base):
    """Mixin class for a table with belief sources, i.e. data-creating entities such as users or scripts."""

    __tablename__ = "belief_source"

    id = Column(Integer, primary_key=True)
