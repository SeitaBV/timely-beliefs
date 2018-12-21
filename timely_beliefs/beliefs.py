from datetime import datetime

from flask_sqlalchemy.model import Model
from sqlalchemy.orm import Query


class TimedBelief(Model):
    """"""

    event_start = db.Column(db.DateTime(timezone=True), primary_key=True)
    belief_horizon = db.Column(db.Interval(), nullable=False, primary_key=True)
    event_value = db.Column(db.Float, nullable=False)
    sensor_id = db.Column(
        db.Integer(), db.ForeignKey("sensor.id", ondelete="CASCADE"), primary_key=True
    )
    source_id = db.Column(db.Integer, db.ForeignKey("sources.id"), primary_key=True)
    sensor = db.relationship(
        "Sensor",
        backref=db.backref(
            "beliefs",
            lazy=True,
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )
    source = db.relationship(
        "Source",
        backref=db.backref(
            "beliefs",
            lazy=True,
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )

    @property
    def event_end(self) -> datetime:
        return self.event_start + self.sensor.event_resolution

    @property
    def knowledge_time(self) -> datetime:
        return self.sensor.knowledge_time(self.event_end)

    @property
    def belief_time(self) -> datetime:
        return self.knowledge_time - self.belief_horizon

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_query(self) -> Query:
        pass
