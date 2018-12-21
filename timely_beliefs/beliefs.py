from flask_sqlalchemy.model import Model
from sqlalchemy.orm import Query


class TimedBelief(Model):
    """"""

    datetime = db.Column(db.DateTime(timezone=True), primary_key=True)
    horizon = db.Column(db.Interval(), nullable=False, primary_key=True)
    value = db.Column(db.Float, nullable=False)
    sensor_id = db.Column(
        db.Integer(), db.ForeignKey("asset.id", ondelete="CASCADE"), primary_key=True
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_query(self) -> Query:
        pass
